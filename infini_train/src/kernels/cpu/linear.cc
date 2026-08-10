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
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    // input[batch..., rows, in_features] × other[batch..., in_features, out_features] -> output[batch..., rows, out_features]
    // 逐 batch 严格相等（batch 维不支持 torch.matmul 的广播语义）  // Fix CR#L30-L33
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

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    const float *input_data = static_cast<const float *>(input->DataPtr());
    const float *other_data = static_cast<const float *>(other->DataPtr());
    float *output_data = static_cast<float *>(output->DataPtr());
    // 循环次序 r->k->c（内层 c）：other 与 output 沿连续方向访问（缓存友好），output 先清零再累积  // Fix CR#L42-L53
    const int64_t output_size = batch * rows * out_features;
    for (int64_t idx = 0; idx < output_size; ++idx) {
        output_data[idx] = 0.0f;
    }
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t k = 0; k < in_features; ++k) {
                const float a = input_data[(b * rows + r) * in_features + k];
                for (int64_t c = 0; c < out_features; ++c) {
                    output_data[(b * rows + r) * out_features + c]
                        += a * other_data[(b * in_features + k) * out_features + c];
                }
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
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    // grad_input = grad_output × other^T，grad_other = input^T × grad_output
    // 逐 batch 严格相等（batch 维不支持 torch.matmul 的广播语义）  // Fix CR#L77-L80
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

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    CHECK(grad_output_dims == output_dims);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);

    const float *input_data = static_cast<const float *>(input->DataPtr());
    const float *other_data = static_cast<const float *>(other->DataPtr());
    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_data = static_cast<float *>(grad_input->DataPtr());
    float *grad_other_data = static_cast<float *>(grad_other->DataPtr());
    // grad_other[b][k][c] = Σ_r input[b][r][k] * grad_output[b][r][c]
    // 循环次序 k->r->c（内层 c）：grad_output 与 grad_other 沿连续方向访问（缓存友好），grad_other 先清零  // Fix CR#L94-L115
    const int64_t grad_other_size = batch * in_features * out_features;
    for (int64_t idx = 0; idx < grad_other_size; ++idx) {
        grad_other_data[idx] = 0.0f;
    }
    for (int64_t b = 0; b < batch; ++b) {
        // grad_input[b][r][k] = Σ_c grad_output[b][r][c] * other[b][k][c]（c 最内层，两操作数连续访问）
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t k = 0; k < in_features; ++k) {
                float sum = 0.0f;
                for (int64_t c = 0; c < out_features; ++c) {
                    sum += grad_output_data[(b * rows + r) * out_features + c]
                           * other_data[(b * in_features + k) * out_features + c];
                }
                grad_input_data[(b * rows + r) * in_features + k] = sum;
            }
        }
        for (int64_t k = 0; k < in_features; ++k) {
            for (int64_t r = 0; r < rows; ++r) {
                const float a = input_data[(b * rows + r) * in_features + k];
                for (int64_t c = 0; c < out_features; ++c) {
                    grad_other_data[(b * in_features + k) * out_features + c]
                        += a * grad_output_data[(b * rows + r) * out_features + c];
                }
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
