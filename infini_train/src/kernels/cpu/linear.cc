#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();

    // 矩阵至少需要二维
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    const size_t rank = input_dims.size();

    // A: [..., M, K]
    // B: [..., K, N]
    const int64_t M = input_dims[rank - 2];
    const int64_t K = input_dims[rank - 1];
    const int64_t N = other_dims[rank - 1];

    // 检查矩阵乘法的中间维度
    CHECK_EQ(K, other_dims[rank - 2]);

    // 计算 batch 数量，并检查批次维度一致
    int64_t batch_size = 1;
    for (size_t dim = 0; dim + 2 < rank; ++dim) {
        CHECK_EQ(input_dims[dim], other_dims[dim]);
        batch_size *= input_dims[dim];
    }

    // 输出形状：[..., M, N]
    std::vector<int64_t> output_dims = input_dims;
    output_dims[rank - 1] = N;

    auto output = std::make_shared<Tensor>(
        output_dims,
        input->Dtype(),
        input->GetDevice()
    );

    const auto *input_data =
        static_cast<const float *>(input->DataPtr());
    const auto *other_data =
        static_cast<const float *>(other->DataPtr());
    auto *output_data =
        static_cast<float *>(output->DataPtr());

    // 每一个 batch 分别进行矩阵乘法
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int64_t input_offset = batch * M * K;
        const int64_t other_offset = batch * K * N;
        const int64_t output_offset = batch * M * N;

        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;

                for (int64_t k = 0; k < K; ++k) {
                    sum +=
                        input_data[input_offset + i * K + k] *
                        other_data[other_offset + k * N + j];
                }

                output_data[output_offset + i * N + j] = sum;
            }
        }
    }
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &grad_dims = grad_output->Dims();

    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());
    CHECK_EQ(input_dims.size(), grad_dims.size());

    const size_t rank = input_dims.size();

    // input:       [..., M, K]
    // other:       [..., K, N]
    // grad_output: [..., M, N]
    const int64_t M = input_dims[rank - 2];
    const int64_t K = input_dims[rank - 1];
    const int64_t N = other_dims[rank - 1];

    CHECK_EQ(other_dims[rank - 2], K);
    CHECK_EQ(grad_dims[rank - 2], M);
    CHECK_EQ(grad_dims[rank - 1], N);

    int64_t batch_size = 1;
    for (size_t dim = 0; dim + 2 < rank; ++dim) {
        CHECK_EQ(input_dims[dim], other_dims[dim]);
        CHECK_EQ(input_dims[dim], grad_dims[dim]);
        batch_size *= input_dims[dim];
    }

    auto grad_input = std::make_shared<Tensor>(
        input_dims,
        input->Dtype(),
        input->GetDevice()
    );

    auto grad_other = std::make_shared<Tensor>(
        other_dims,
        other->Dtype(),
        other->GetDevice()
    );

    const auto *input_data =
        static_cast<const float *>(input->DataPtr());
    const auto *other_data =
        static_cast<const float *>(other->DataPtr());
    const auto *grad_output_data =
        static_cast<const float *>(grad_output->DataPtr());

    auto *grad_input_data =
        static_cast<float *>(grad_input->DataPtr());
    auto *grad_other_data =
        static_cast<float *>(grad_other->DataPtr());

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int64_t input_offset = batch * M * K;
        const int64_t other_offset = batch * K * N;
        const int64_t grad_output_offset = batch * M * N;

        /*
         * grad_input = grad_output × other^T
         *
         * [M, N] × [N, K] = [M, K]
         */
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t k = 0; k < K; ++k) {
                float sum = 0.0f;

                for (int64_t j = 0; j < N; ++j) {
                    sum +=
                        grad_output_data[
                            grad_output_offset + i * N + j
                        ] *
                        other_data[
                            other_offset + k * N + j
                        ];
                }

                grad_input_data[
                    input_offset + i * K + k
                ] = sum;
            }
        }

        /*
         * grad_other = input^T × grad_output
         *
         * [K, M] × [M, N] = [K, N]
         */
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;

                for (int64_t i = 0; i < M; ++i) {
                    sum +=
                        input_data[
                            input_offset + i * K + k
                        ] *
                        grad_output_data[
                            grad_output_offset + i * N + j
                        ];
                }

                grad_other_data[
                    other_offset + k * N + j
                ] = sum;
            }
        }
    }
    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
