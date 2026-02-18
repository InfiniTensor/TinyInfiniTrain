#include "cublas_v2.h"
#include "glog/logging.h"
#include <cub/block/block_reduce.cuh>
#include <limits>
#include <numeric>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

namespace {
const char *CublasStatusToString(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
#ifdef CUBLAS_STATUS_NOT_SUPPORTED
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#ifdef CUBLAS_STATUS_LICENSE_ERROR
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}
} // namespace

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
            LOG(FATAL) << "CUBLAS Error: " << CublasStatusToString(status) << " (" << static_cast<int>(status)        \
                       << ") at " << __FILE__ << ":" << __LINE__;                                                      \
        }                                                                                                              \
    } while (0)

namespace {
struct MatmulMeta {
    int64_t batch = 1;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    std::vector<int64_t> output_dims;
};

MatmulMeta BuildMatmulMeta(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(other->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(input->GetDevice().Type()), static_cast<int>(other->GetDevice().Type()));

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    const int64_t ndim = input_dims.size();
    for (int64_t i = 0; i < ndim - 2; ++i) {
        // 当前作业要求下，批次维逐维一致，不启用广播。
        CHECK_EQ(input_dims[i], other_dims[i]);
    }

    const int64_t m = input_dims[ndim - 2];
    const int64_t k = input_dims[ndim - 1];
    const int64_t other_k = other_dims[ndim - 2];
    const int64_t n = other_dims[ndim - 1];
    CHECK_EQ(k, other_k);

    MatmulMeta meta;
    meta.batch = std::accumulate(input_dims.begin(), input_dims.end() - 2, int64_t{1}, std::multiplies<int64_t>{});
    meta.m = m;
    meta.k = k;
    meta.n = n;
    meta.output_dims = input_dims;
    meta.output_dims[ndim - 1] = n;
    return meta;
}
} // namespace

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    const auto meta = BuildMatmulMeta(input, other);
    CHECK_LE(meta.batch, static_cast<int64_t>(std::numeric_limits<int>::max()));
    CHECK_LE(meta.m, static_cast<int64_t>(std::numeric_limits<int>::max()));
    CHECK_LE(meta.k, static_cast<int64_t>(std::numeric_limits<int>::max()));
    CHECK_LE(meta.n, static_cast<int64_t>(std::numeric_limits<int>::max()));

    auto output = std::make_shared<Tensor>(meta.output_dims, DataType::kFLOAT32, input->GetDevice());
    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *other_ptr = static_cast<const float *>(other->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    const int m = static_cast<int>(meta.m);
    const int k = static_cast<int>(meta.k);
    const int n = static_cast<int>(meta.n);
    const int batch = static_cast<int>(meta.batch);
    const long long int stride_input = static_cast<long long>(meta.m * meta.k);
    const long long int stride_other = static_cast<long long>(meta.k * meta.n);
    const long long int stride_output = static_cast<long long>(meta.m * meta.n);

    // Row-major 乘法 C = A * B，转换为 cuBLAS 视角中的 C^T = B^T * A^T。
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    if (batch == 1) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, other_ptr, n, input_ptr, k, &beta,
                                 output_ptr, n));
    } else {
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, other_ptr, n,
                                               stride_other, input_ptr, k, stride_input, &beta, output_ptr, n,
                                               stride_output, batch));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================
    const auto meta = BuildMatmulMeta(input, other);
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_EQ(grad_output_dims.size(), meta.output_dims.size());
    for (size_t dim_idx = 0; dim_idx < grad_output_dims.size(); ++dim_idx) {
        CHECK_EQ(grad_output_dims[dim_idx], meta.output_dims[dim_idx]);
    }
    CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(grad_output->GetDevice().Type()), static_cast<int>(input->GetDevice().Type()));
    CHECK_LE(meta.batch, static_cast<int64_t>(std::numeric_limits<int>::max()));
    CHECK_LE(meta.m, static_cast<int64_t>(std::numeric_limits<int>::max()));
    CHECK_LE(meta.k, static_cast<int64_t>(std::numeric_limits<int>::max()));
    CHECK_LE(meta.n, static_cast<int64_t>(std::numeric_limits<int>::max()));

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other->Dims(), DataType::kFLOAT32, other->GetDevice());

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *other_ptr = static_cast<const float *>(other->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    float *grad_other_ptr = static_cast<float *>(grad_other->DataPtr());

    const int m = static_cast<int>(meta.m);
    const int k = static_cast<int>(meta.k);
    const int n = static_cast<int>(meta.n);
    const int batch = static_cast<int>(meta.batch);
    const long long int stride_input = static_cast<long long>(meta.m * meta.k);
    const long long int stride_other = static_cast<long long>(meta.k * meta.n);
    const long long int stride_output = static_cast<long long>(meta.m * meta.n);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    if (batch == 1) {
        // dInput = dOut * Other^T
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, other_ptr, n, grad_output_ptr, n,
                                 &beta, grad_input_ptr, k));
        // dOther = Input^T * dOut
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, grad_output_ptr, n, input_ptr, k,
                                 &beta, grad_other_ptr, n));
    } else {
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha, other_ptr, n,
                                               stride_other, grad_output_ptr, n, stride_output, &beta, grad_input_ptr,
                                               k, stride_input, batch));
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha, grad_output_ptr, n,
                                               stride_output, input_ptr, k, stride_input, &beta, grad_other_ptr, n,
                                               stride_other, batch));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
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
