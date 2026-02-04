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
    const auto &a_dims = input->Dims();
    const auto &b_dims = other->Dims();

    CHECK_GE(a_dims.size(), 2);
    CHECK_GE(b_dims.size(), 2);

    const int64_t m = a_dims[a_dims.size() - 2];
    const int64_t k = a_dims[a_dims.size() - 1];
    const int64_t n = b_dims[b_dims.size() - 1];

    std::vector<int64_t> batch_dims(a_dims.begin(), a_dims.end() - 2);
    std::vector<int64_t> b_batch_dims(b_dims.begin(), b_dims.end() - 2);
    CHECK_EQ(batch_dims.size(), b_batch_dims.size());
    for (size_t i = 0; i < batch_dims.size(); ++i) CHECK_EQ(batch_dims[i], b_batch_dims[i]);
    std::vector<int64_t> out_dims = batch_dims;
    out_dims.push_back(m);
    out_dims.push_back(n);
    auto output = std::make_shared<Tensor>(out_dims, DataType::kFLOAT32);
    int64_t batch_count = 1;
    for (auto d : batch_dims) batch_count *= d;

    const float *a_ptr = static_cast<const float *>(input->DataPtr());
    const float *b_ptr = static_cast<const float *>(other->DataPtr());
    float *out_ptr = static_cast<float *>(output->DataPtr());

    const int64_t a_block = m * k;
    const int64_t b_block = k * n;
    const int64_t out_block = m * n;

    for (int64_t batch = 0; batch < batch_count; ++batch) {
        const float *a_block_ptr = a_ptr + batch * a_block;
        const float *b_block_ptr = b_ptr + batch * b_block;
        float *out_block_ptr = out_ptr + batch * out_block;

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(
            reinterpret_cast<const float *>(a_block_ptr), m, k);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B(
            reinterpret_cast<const float *>(b_block_ptr), k, n);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C(
            reinterpret_cast<float *>(out_block_ptr), m, n);

        C.noalias() = A * B;
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
    const auto &a_dims = input->Dims();
    const auto &b_dims = other->Dims();

    CHECK_GE(a_dims.size(), 2);
    CHECK_GE(b_dims.size(), 2);

    const int64_t m = a_dims[a_dims.size() - 2];
    const int64_t k = a_dims[a_dims.size() - 1];
    const int64_t n = b_dims[b_dims.size() - 1];

    std::vector<int64_t> batch_dims(a_dims.begin(), a_dims.end() - 2);
    std::vector<int64_t> b_batch_dims(b_dims.begin(), b_dims.end() - 2);
    CHECK_EQ(batch_dims.size(), b_batch_dims.size());
    for (size_t i = 0; i < batch_dims.size(); ++i) CHECK_EQ(batch_dims[i], b_batch_dims[i]);

    auto grad_input = std::make_shared<Tensor>(a_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(b_dims, DataType::kFLOAT32);

    int64_t batch_count = 1;
    for (auto d : batch_dims) batch_count *= d;

    const float *a_ptr = static_cast<const float *>(input->DataPtr());
    const float *b_ptr = static_cast<const float *>(other->DataPtr());
    const float *g_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *gi_ptr = static_cast<float *>(grad_input->DataPtr());
    float *go_ptr = static_cast<float *>(grad_other->DataPtr());

    const int64_t a_block = m * k;
    const int64_t b_block = k * n;
    const int64_t g_block = m * n;

    for (int64_t batch = 0; batch < batch_count; ++batch) {
        const float *a_block_ptr = a_ptr + batch * a_block;
        const float *b_block_ptr = b_ptr + batch * b_block;
        const float *g_block_ptr = g_ptr + batch * g_block;
        float *gi_block_ptr = gi_ptr + batch * a_block;
        float *go_block_ptr = go_ptr + batch * b_block;

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(
            reinterpret_cast<const float *>(a_block_ptr), m, k);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B(
            reinterpret_cast<const float *>(b_block_ptr), k, n);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> G(
            reinterpret_cast<const float *>(g_block_ptr), m, n);

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> GI(
            reinterpret_cast<float *>(gi_block_ptr), m, k);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> GO(
            reinterpret_cast<float *>(go_block_ptr), k, n);

        GI.noalias() = G * B.transpose();
        GO.noalias() = A.transpose() * G;
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
