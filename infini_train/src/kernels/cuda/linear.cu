#include "cublas_v2.h"
#include "glog/logging.h"
#include <cub/block/block_reduce.cuh>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

// restrict 承诺：写目标（output）必须与读源（input/other）不重叠，原地调用属未定义行为  // Fix CR#L26-L28
__global__ void MatmulForwardKernel(const float *__restrict__ input, const float *__restrict__ other,
                                    float *__restrict__ output, int rows, int in_features, int out_features,
                                    int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int col = idx % out_features;
    int tmp = idx / out_features;
    int row = tmp % rows;
    int batch = tmp / rows;

    // 每个线程串行累加 k（k 升序，固定累加顺序保证 GPU 侧运行位级确定；
    // 与 CPU 版在容差内一致——nvcc 默认 FMA 融合（--fmad=true），存在 1 ULP 级差异）  // Fix CR#L38
    float sum = 0.0f;
    const float *input_row = input + (batch * rows + row) * in_features;
    const float *other_base = other + (batch * in_features) * out_features + col;
    for (int k = 0; k < in_features; ++k) {
        sum += input_row[k] * other_base[k * out_features];
    }
    output[idx] = sum;
}

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    // input[batch..., rows, in_features] × other[batch..., in_features, out_features] -> output[batch..., rows, out_features]
    // 逐 batch 严格相等（batch 维不支持 torch.matmul 的广播语义），语义与 CPU 版一致
    const int64_t batch_ndim = input_dims.size() - 2;
    const int64_t batch =
        std::accumulate(input_dims.begin(), input_dims.begin() + batch_ndim, 1, std::multiplies<int64_t>{});
    const int64_t rows = input_dims[batch_ndim];
    const int64_t in_features = *input_dims.rbegin();
    const int64_t out_features = *other_dims.rbegin();
    for (int64_t dim = 0; dim < batch_ndim; ++dim) {
        CHECK_EQ(input_dims[dim], other_dims[dim]);
    }
    CHECK_EQ(in_features, *(other_dims.rbegin() + 1));
    CHECK_EQ(static_cast<int>(input->GetDevice().Type()), static_cast<int>(other->GetDevice().Type()));  // Fix CR#L160-L161

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    // 一元素一线程 + 边界检查（网格按 CEIL_DIV 划分，与官方 kernel 风格一致）
    const int64_t total = batch * rows * out_features;
    // Fix CR#L76-L82：32 位索引范围防护（total 超出 int 范围时快速失败，避免溢出为负导致静默空输出）
    CHECK_LE(total, std::numeric_limits<int>::max());
    int threads_per_block = 256;
    int num_blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);
    MatmulForwardKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(other->DataPtr()),
        static_cast<float *>(output->DataPtr()), static_cast<int>(rows), static_cast<int>(in_features),
        static_cast<int>(out_features), static_cast<int>(total));
    return output;
}

// restrict 承诺：写目标（grad_input）必须与读源（grad_output/other）不重叠，原地调用属未定义行为  // Fix CR#L26-L28
__global__ void MatmulBackwardGradInputKernel(const float *__restrict__ grad_output, const float *__restrict__ other,
                                              float *__restrict__ grad_input, int rows, int in_features,
                                              int out_features, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int k = idx % in_features;
    int tmp = idx / in_features;
    int row = tmp % rows;
    int batch = tmp / rows;

    // grad_input[batch][row][k] = Σ_c grad_output[batch][row][c] * other[batch][k][c]
    // c 串行累加保证 GPU 侧运行位级确定；warp 内相邻线程（k 连续）读 other 为按列访问（非 coalesced）——
    // A×B^T 结构在无共享内存的 naive 形态下两操作数无法同时行主序访问，当前映射已保证输出写 coalesced
    // 与 grad_output 广播读，教学定位保持现状  // Fix CR#L86-L106, L98
    float sum = 0.0f;
    const float *grad_output_row = grad_output + (batch * rows + row) * out_features;
    const float *other_base = other + (batch * in_features + k) * out_features;
    for (int c = 0; c < out_features; ++c) {
        sum += grad_output_row[c] * other_base[c];
    }
    grad_input[idx] = sum;
}

// restrict 承诺：写目标（grad_other）必须与读源（input/grad_output）不重叠，原地调用属未定义行为  // Fix CR#L26-L28
__global__ void MatmulBackwardGradOtherKernel(const float *__restrict__ input, const float *__restrict__ grad_output,
                                              float *__restrict__ grad_other, int rows, int in_features,
                                              int out_features, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int col = idx % out_features;
    int tmp = idx / out_features;
    int k = tmp % in_features;
    int batch = tmp / in_features;

    // grad_other[batch][k][col] = Σ_r input[batch][row][k] * grad_output[batch][row][col]
    // r 串行累加保证 GPU 侧运行位级确定；warp 内相邻线程（col 连续）grad_output 与写均 coalesced  // Fix CR#L120
    float sum = 0.0f;
    const float *input_base = input + (batch * rows) * in_features + k;
    const float *grad_output_base = grad_output + (batch * rows) * out_features + col;
    for (int r = 0; r < rows; ++r) {
        sum += input_base[r * in_features] * grad_output_base[r * out_features];
    }
    grad_other[idx] = sum;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    // grad_input = grad_output × other^T，grad_other = input^T × grad_output
    // 逐 batch 严格相等（batch 维不支持 torch.matmul 的广播语义），语义与 CPU 版一致
    const int64_t batch_ndim = input_dims.size() - 2;
    const int64_t batch =
        std::accumulate(input_dims.begin(), input_dims.begin() + batch_ndim, 1, std::multiplies<int64_t>{});
    const int64_t rows = input_dims[batch_ndim];
    const int64_t in_features = *input_dims.rbegin();
    const int64_t out_features = *other_dims.rbegin();
    for (int64_t dim = 0; dim < batch_ndim; ++dim) {
        CHECK_EQ(input_dims[dim], other_dims[dim]);
    }
    CHECK_EQ(in_features, *(other_dims.rbegin() + 1));
    // Fix CR#L160-L161：设备一致性校验（跨设备误用快速失败而非异步运行时错误）
    CHECK_EQ(static_cast<int>(input->GetDevice().Type()), static_cast<int>(other->GetDevice().Type()));
    CHECK_EQ(static_cast<int>(input->GetDevice().Type()), static_cast<int>(grad_output->GetDevice().Type()));

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    CHECK(grad_output_dims == output_dims);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32, grad_output->GetDevice());

    // 两个 kernel 均为一元素一线程 + 边界检查，串行内层循环保证 GPU 侧运行位级确定
    const int64_t grad_input_total = batch * rows * in_features;
    // Fix CR#L164-L177：32 位索引范围防护（total 超出 int 范围时快速失败，避免静默错误）
    CHECK_LE(grad_input_total, std::numeric_limits<int>::max());
    int threads_per_block = 256;
    int num_blocks = static_cast<int>((grad_input_total + threads_per_block - 1) / threads_per_block);
    MatmulBackwardGradInputKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(grad_output->DataPtr()), static_cast<const float *>(other->DataPtr()),
        static_cast<float *>(grad_input->DataPtr()), static_cast<int>(rows), static_cast<int>(in_features),
        static_cast<int>(out_features), static_cast<int>(grad_input_total));

    const int64_t grad_other_total = batch * in_features * out_features;
    CHECK_LE(grad_other_total, std::numeric_limits<int>::max());
    num_blocks = static_cast<int>((grad_other_total + threads_per_block - 1) / threads_per_block);
    MatmulBackwardGradOtherKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_other->DataPtr()), static_cast<int>(rows), static_cast<int>(in_features),
        static_cast<int>(out_features), static_cast<int>(grad_other_total));
    return {grad_input, grad_other};
}

__global__ void BiasCopyKernel(float *output, const float *bias, int bs, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * out_features) {
        return;
    }
    int j = idx % out_features;
    output[idx] = bias[j];
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {

    /*
        !transpose: output = input * weight + bias
        output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]

        transpose:  output = input * weight^T + bias
        output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);

    // As for cublas:
    // C = alpha * op(B) * op(A) + beta * C
    // Dimensions:
    //   input:  (bs, in_features)
    //   weight: (in_features, out_features) or (out_features, in_features) if transposed
    //   output: (bs, out_features)
    const int64_t out_features = weight_dims[transpose ? 0 : 1];

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_features);
        int threads_per_block = 256;
        int num_blocks = (bs * out_features + threads_per_block - 1) / threads_per_block;
        BiasCopyKernel<<<num_blocks, threads_per_block>>>(
            static_cast<float *>(output->DataPtr()), static_cast<const float *>(bias->DataPtr()), bs, out_features);
    } else {
        output->Fill<float>(0.0f);
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    if (transpose) {
        // weight is [out_features, in_features] here

        // output = input * weight.T --> output.T = weight * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[in_features, out_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    } else {
        // output = input * weight --> output.T =  weight.T * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[out_features, in_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}

template <int BLOCK_SIZE>
__global__ void ReduceColumnsKernel(const float *__restrict__ input, float *__restrict__ output, int num_rows,
                                    int num_cols) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int row = blockIdx.x;
    float sum = 0.0f;

    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) { sum += input[row * num_cols + col]; }

    float reduced = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        output[row] = reduced;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_weight->Fill<float>(0.0f);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32,
                                             grad_output->GetDevice());
        grad_bias->Fill<float>(0.0f);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (transpose) {
        // weight is [out_features, in_features] here

        // d_input = d_output * weight --> d_input.T = weight.T * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[in_features, out_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = d_output.T * input --> d_weight.T = input.T * d_output
        // C = d_weight.T[in_features, out_features]
        // A = input.T[in_features, bs]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features, out_features, bs, &alpha,
                                 static_cast<const float *>(input->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), in_features));
    } else {
        // weight is [in_features, out_features] here

        // d_input = d_output * weight.T --> d_input.T = weight * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[out_features, in_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = input.T * d_output --> d_weight.T = d_output.T * input
        // C = d_weight.T[out_features, in_features]
        // A = d_output.T[out_features, bs]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_features, in_features, bs, &alpha,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), out_features));
    }

    // d_bias = \sum_i(i=0, bs-1) d_output[i]
    if (bias) {
        constexpr int BLOCK_SIZE = 256;
        int threads_per_block = BLOCK_SIZE;
        int num_blocks = out_features;
        ReduceColumnsKernel<BLOCK_SIZE>
            <<<num_blocks, threads_per_block>>>(static_cast<const float *>(grad_output->DataPtr()),
                                                static_cast<float *>(grad_bias->DataPtr()), out_features, bs);
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LINEAR_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LINEAR_KERNEL(MatmulForward)
REGISTER_CUDA_LINEAR_KERNEL(MatmulBackward)
REGISTER_CUDA_LINEAR_KERNEL(LinearForward)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CUDA_LINEAR_KERNEL
