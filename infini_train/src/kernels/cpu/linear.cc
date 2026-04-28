#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

using RowMajorMatf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);
    const int64_t k = *input_dims.rbegin();
    const int64_t n = *other_dims.rbegin();
    CHECK_EQ(k, *(other_dims.rbegin() + 1));

    if (other_dims.size() == 2) {
        // Broadcast: input[..., K] @ other[K, N]
        auto output_dims = input_dims;
        *output_dims.rbegin() = n;
        auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);
        output->EigenMatrix() = input->EigenMatrix() * other->EigenMatrix();
        return output;
    }

    // Batched: input[..., M, K] @ other[..., K, N]
    CHECK_EQ(input_dims.size(), other_dims.size());
    const int64_t m = *(input_dims.rbegin() + 1);
    int64_t batch = 1;
    for (size_t i = 0; i < input_dims.size() - 2; ++i) {
        CHECK_EQ(input_dims[i], other_dims[i]);
        batch *= input_dims[i];
    }
    auto output_dims = input_dims;
    *output_dims.rbegin() = n;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);
    const float *a = static_cast<const float *>(input->DataPtr());
    const float *b = static_cast<const float *>(other->DataPtr());
    float *c = static_cast<float *>(output->DataPtr());
    for (int64_t i = 0; i < batch; ++i) {
        Eigen::Map<RowMajorMatf>(c + i * m * n, m, n).noalias() =
            Eigen::Map<const RowMajorMatf>(a + i * m * k, m, k) *
            Eigen::Map<const RowMajorMatf>(b + i * k * n, k, n);
    }
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const int64_t k = *input_dims.rbegin();
    const int64_t n = *other_dims.rbegin();

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);

    if (other_dims.size() == 2) {
        // grad_input = grad_output @ other^T
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * other->EigenMatrix().transpose();
        // grad_other = input^T @ grad_output
        grad_other->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
        return {grad_input, grad_other};
    }

    // Batched backward
    const int64_t m = *(input_dims.rbegin() + 1);
    int64_t batch = 1;
    for (size_t i = 0; i < input_dims.size() - 2; ++i) { batch *= input_dims[i]; }
    const float *go = static_cast<const float *>(grad_output->DataPtr());
    const float *a = static_cast<const float *>(input->DataPtr());
    const float *b = static_cast<const float *>(other->DataPtr());
    float *ga = static_cast<float *>(grad_input->DataPtr());
    float *gb = static_cast<float *>(grad_other->DataPtr());
    for (int64_t i = 0; i < batch; ++i) {
        Eigen::Map<RowMajorMatf>(ga + i * m * k, m, k).noalias() =
            Eigen::Map<const RowMajorMatf>(go + i * m * n, m, n) *
            Eigen::Map<const RowMajorMatf>(b + i * k * n, k, n).transpose();
        Eigen::Map<RowMajorMatf>(gb + i * k * n, k, n).noalias() =
            Eigen::Map<const RowMajorMatf>(a + i * m * k, m, k).transpose() *
            Eigen::Map<const RowMajorMatf>(go + i * m * n, m, n);
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
