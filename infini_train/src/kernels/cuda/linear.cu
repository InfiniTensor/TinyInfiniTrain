#include "cublas_v2.h"
#include "glog/logging.h"

#include <algorithm>
#include <cstdlib>
#include <cub/block/block_reduce.cuh>
#include <limits>
#include <string>

#include "infini_train/include/cuda_check.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
namespace {

const char *CublasOperationName(cublasOperation_t op) {
    switch (op) {
    case CUBLAS_OP_N:
        return "N";
    case CUBLAS_OP_T:
        return "T";
    case CUBLAS_OP_C:
        return "C";
    default:
        return "unknown";
    }
}

void CheckFitsInt(const char *context, const char *name, int64_t value) {
    CHECK_GE(value, static_cast<int64_t>(0)) << context << " invalid " << name << "=" << value;
    CHECK_LE(value, static_cast<int64_t>(std::numeric_limits<int>::max()))
        << context << " " << name << " exceeds cuBLAS int range: " << value;
}

void CheckNonNegativeStride(const char *context, const char *name, int64_t value) {
    CHECK_GE(value, static_cast<int64_t>(0)) << context << " invalid " << name << "=" << value;
}

void CheckGemmArgs(const char *context, cublasOperation_t trans_a, cublasOperation_t trans_b, int64_t m, int64_t n,
                   int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
    CHECK(trans_a == CUBLAS_OP_N || trans_a == CUBLAS_OP_T || trans_a == CUBLAS_OP_C)
        << context << " invalid transA=" << CublasOperationName(trans_a);
    CHECK(trans_b == CUBLAS_OP_N || trans_b == CUBLAS_OP_T || trans_b == CUBLAS_OP_C)
        << context << " invalid transB=" << CublasOperationName(trans_b);

    CheckFitsInt(context, "m", m);
    CheckFitsInt(context, "n", n);
    CheckFitsInt(context, "k", k);
    CheckFitsInt(context, "lda", lda);
    CheckFitsInt(context, "ldb", ldb);
    CheckFitsInt(context, "ldc", ldc);

    const int64_t min_lda = std::max<int64_t>(1, trans_a == CUBLAS_OP_N ? m : k);
    const int64_t min_ldb = std::max<int64_t>(1, trans_b == CUBLAS_OP_N ? k : n);
    const int64_t min_ldc = std::max<int64_t>(1, m);

    CHECK_GE(lda, min_lda) << context << " invalid lda=" << lda << " for transA=" << CublasOperationName(trans_a)
                           << ", m=" << m << ", k=" << k;
    CHECK_GE(ldb, min_ldb) << context << " invalid ldb=" << ldb << " for transB=" << CublasOperationName(trans_b)
                           << ", n=" << n << ", k=" << k;
    CHECK_GE(ldc, min_ldc) << context << " invalid ldc=" << ldc << " for output m=" << m << ", n=" << n;
}

void CheckStridedBatchedGemmArgs(const char *context, cublasOperation_t trans_a, cublasOperation_t trans_b, int64_t m,
                                 int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc, int64_t stride_a,
                                 int64_t stride_b, int64_t stride_c, int64_t batch_count) {
    CheckGemmArgs(context, trans_a, trans_b, m, n, k, lda, ldb, ldc);
    CheckFitsInt(context, "batch_count", batch_count);
    CheckNonNegativeStride(context, "strideA", stride_a);
    CheckNonNegativeStride(context, "strideB", stride_b);
    CheckNonNegativeStride(context, "strideC", stride_c);
}

bool EnvFlagEnabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && std::string(value) == "1";
}

void LogCublasSgemmDiagnostics(const char *context, cublasHandle_t handle, cublasOperation_t trans_a,
                               cublasOperation_t trans_b, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
                               int64_t ldc, float alpha, float beta) {
    if (!EnvFlagEnabled("TINY_LOG_CUBLAS_GEMM")) {
        return;
    }

    cudaStream_t stream = nullptr;
    cublasMath_t math_mode{};
    cublasAtomicsMode_t atomics_mode{};
    cublasPointerMode_t pointer_mode{};
    CUBLAS_CHECK(cublasGetStream(handle, &stream));
    CUBLAS_CHECK(cublasGetMathMode(handle, &math_mode));
    CUBLAS_CHECK(cublasGetAtomicsMode(handle, &atomics_mode));
    CUBLAS_CHECK(cublasGetPointerMode(handle, &pointer_mode));
    const char *workspace_config = std::getenv("CUBLAS_WORKSPACE_CONFIG");

    LOG(INFO) << "CUBLAS_GEMM_DIAG context=" << context
              << " api=cublasSgemm"
              << " handle=" << reinterpret_cast<const void *>(handle)
              << " handle_created_per_call=1"
              << " stream=" << reinterpret_cast<const void *>(stream)
              << " default_stream=" << (stream == nullptr)
              << " cublas_get_stream_status=success"
              << " math_mode=" << static_cast<int>(math_mode)
              << " atomics_mode=" << static_cast<int>(atomics_mode)
              << " pointer_mode=" << static_cast<int>(pointer_mode)
              << " m=" << m
              << " n=" << n
              << " k=" << k
              << " trans_a=" << CublasOperationName(trans_a)
              << " trans_b=" << CublasOperationName(trans_b)
              << " lda=" << lda
              << " ldb=" << ldb
              << " ldc=" << ldc
              << " alpha=" << alpha
              << " beta=" << beta
              << " data_type=float32"
              << " compute_type=implicit_fp32_cublasSgemm"
              << " algorithm=default_cublasSgemm_not_queryable"
              << " workspace_config=" << (workspace_config == nullptr ? "unset" : workspace_config)
              << " multiple_streams=not_observed_default_stream_callsite";
}

} // namespace

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();

    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());
    CHECK_EQ(
        input_dims.back(),
        other_dims[other_dims.size() - 2]
    );

    for (size_t i = 0; i + 2 < input_dims.size(); ++i) {
        CHECK_EQ(input_dims[i], other_dims[i]);
    }

    const int64_t M =
        input_dims[input_dims.size() - 2];
    const int64_t K =
        input_dims.back();
    const int64_t N =
        other_dims.back();

    const int64_t batch_count =
        static_cast<int64_t>(input->NumElements()) /
        (M * K);

    CHECK_EQ(
        static_cast<int64_t>(other->NumElements()),
        batch_count * K * N
    );

    auto output_dims = input_dims;
    output_dims.back() = N;

    auto output = std::make_shared<Tensor>(
        output_dims,
        DataType::kFLOAT32,
        input->GetDevice()
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const int64_t stride_other = K * N;
    const int64_t stride_input = M * K;
    const int64_t stride_output = M * N;

    CheckStridedBatchedGemmArgs("MatmulForward cublasSgemmStridedBatched", CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, N, K,
                                N, stride_other, stride_input, stride_output, batch_count);

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        static_cast<const float *>(other->DataPtr()),
        static_cast<int>(N),
        stride_other,
        static_cast<const float *>(input->DataPtr()),
        static_cast<int>(K),
        stride_input,
        &beta,
        static_cast<float *>(output->DataPtr()),
        static_cast<int>(N),
        stride_output,
        static_cast<int>(batch_count)
    ));

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
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &grad_dims = grad_output->Dims();

    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    CHECK_GE(grad_dims.size(), 2);

    const int64_t M =
        input_dims[input_dims.size() - 2];
    const int64_t K =
        input_dims.back();
    const int64_t N =
        other_dims.back();

    CHECK_EQ(
        other_dims[other_dims.size() - 2],
        K
    );
    CHECK_EQ(
        grad_dims[grad_dims.size() - 2],
        M
    );
    CHECK_EQ(
        grad_dims.back(),
        N
    );

    const int64_t batch_count =
        static_cast<int64_t>(input->NumElements()) /
        (M * K);

    CHECK_EQ(
        static_cast<int64_t>(other->NumElements()),
        batch_count * K * N
    );
    CHECK_EQ(
        static_cast<int64_t>(grad_output->NumElements()),
        batch_count * M * N
    );

    auto grad_input = std::make_shared<Tensor>(
        input_dims,
        DataType::kFLOAT32,
        input->GetDevice()
    );

    auto grad_other = std::make_shared<Tensor>(
        other_dims,
        DataType::kFLOAT32,
        other->GetDevice()
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const int64_t stride_input = M * K;
    const int64_t stride_other = K * N;
    const int64_t stride_grad_output = M * N;

    CheckStridedBatchedGemmArgs("MatmulBackward grad_input cublasSgemmStridedBatched", CUBLAS_OP_T, CUBLAS_OP_N, K,
                                M, N, N, N, K, stride_other, stride_grad_output, stride_input, batch_count);

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(K),
        static_cast<int>(M),
        static_cast<int>(N),
        &alpha,
        static_cast<const float *>(other->DataPtr()),
        static_cast<int>(N),
        stride_other,
        static_cast<const float *>(grad_output->DataPtr()),
        static_cast<int>(N),
        stride_grad_output,
        &beta,
        static_cast<float *>(grad_input->DataPtr()),
        static_cast<int>(K),
        stride_input,
        static_cast<int>(batch_count)
    ));

    CheckStridedBatchedGemmArgs("MatmulBackward grad_other cublasSgemmStridedBatched", CUBLAS_OP_N, CUBLAS_OP_T, N,
                                K, M, N, K, N, stride_grad_output, stride_input, stride_other, batch_count);

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        static_cast<int>(N),
        static_cast<int>(K),
        static_cast<int>(M),
        &alpha,
        static_cast<const float *>(grad_output->DataPtr()),
        static_cast<int>(N),
        stride_grad_output,
        static_cast<const float *>(input->DataPtr()),
        static_cast<int>(K),
        stride_input,
        &beta,
        static_cast<float *>(grad_other->DataPtr()),
        static_cast<int>(N),
        stride_other,
        static_cast<int>(batch_count)
    ));

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
        CUDA_KERNEL_CHECK();
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
        CheckGemmArgs("LinearForward transpose cublasSgemm", CUBLAS_OP_T, CUBLAS_OP_N, out_features, bs,
                      in_features, in_features, in_features, out_features);
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    } else {
        // output = input * weight --> output.T =  weight.T * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[out_features, in_features]
        // B = input.T[in_features, bs]
        CheckGemmArgs("LinearForward cublasSgemm", CUBLAS_OP_N, CUBLAS_OP_N, out_features, bs, in_features,
                      out_features, in_features, out_features);
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
        CheckGemmArgs("LinearBackward transpose grad_input cublasSgemm", CUBLAS_OP_N, CUBLAS_OP_N, in_features, bs,
                      out_features, in_features, out_features, in_features);
        LogCublasSgemmDiagnostics("LinearBackward transpose grad_input cublasSgemm", handle, CUBLAS_OP_N,
                                  CUBLAS_OP_N, in_features, bs, out_features, in_features, out_features, in_features,
                                  alpha, beta);
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = d_output.T * input --> d_weight.T = input.T * d_output
        // C = d_weight.T[in_features, out_features]
        // A = input.T[in_features, bs]
        // B = d_output.T[out_features, bs]
        CheckGemmArgs("LinearBackward transpose grad_weight cublasSgemm", CUBLAS_OP_N, CUBLAS_OP_T, in_features,
                      out_features, bs, in_features, out_features, in_features);
        LogCublasSgemmDiagnostics("LinearBackward transpose grad_weight cublasSgemm", handle, CUBLAS_OP_N,
                                  CUBLAS_OP_T, in_features, out_features, bs, in_features, out_features, in_features,
                                  alpha, beta);
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
        CheckGemmArgs("LinearBackward grad_input cublasSgemm", CUBLAS_OP_T, CUBLAS_OP_N, in_features, bs,
                      out_features, out_features, out_features, in_features);
        LogCublasSgemmDiagnostics("LinearBackward grad_input cublasSgemm", handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  in_features, bs, out_features, out_features, out_features, in_features, alpha, beta);
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = input.T * d_output --> d_weight.T = d_output.T * input
        // C = d_weight.T[out_features, in_features]
        // A = d_output.T[out_features, bs]
        // B = input.T[in_features, bs]
        CheckGemmArgs("LinearBackward grad_weight cublasSgemm", CUBLAS_OP_N, CUBLAS_OP_T, out_features, in_features,
                      bs, out_features, in_features, out_features);
        LogCublasSgemmDiagnostics("LinearBackward grad_weight cublasSgemm", handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  out_features, in_features, bs, out_features, in_features, out_features, alpha, beta);
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
        CUDA_KERNEL_CHECK();
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
