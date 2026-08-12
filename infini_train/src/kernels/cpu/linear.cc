#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {
using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct MatmulShape {
    int64_t batch_count;
    int64_t m;
    int64_t k;
    int64_t n;
    std::vector<int64_t> output_dims;
};

MatmulShape ValidateMatmulInputs(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    CHECK(input);
    CHECK(other);
    CHECK(input->GetDevice().IsCPU());
    CHECK(input->GetDevice() == other->GetDevice());
    CHECK(input->Dtype() == DataType::kFLOAT32);
    CHECK(other->Dtype() == DataType::kFLOAT32);

    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());
    for (size_t i = 0; i + 2 < input_dims.size(); ++i) { CHECK_EQ(input_dims[i], other_dims[i]); }

    const int64_t m = input_dims[input_dims.size() - 2];
    const int64_t k = input_dims.back();
    const int64_t n = other_dims.back();
    CHECK_EQ(k, other_dims[other_dims.size() - 2]);

    const int64_t batch_count
        = std::accumulate(input_dims.begin(), input_dims.end() - 2, int64_t{1}, std::multiplies<int64_t>{});
    auto output_dims = input_dims;
    output_dims.back() = n;
    return {batch_count, m, k, n, std::move(output_dims)};
}

void ValidateGradOutput(const std::shared_ptr<Tensor> &grad_output, const MatmulShape &shape,
                        const Device &device) {
    CHECK(grad_output);
    CHECK(grad_output->GetDevice() == device);
    CHECK(grad_output->Dtype() == DataType::kFLOAT32);
    CHECK(grad_output->Dims() == shape.output_dims);
}
} // namespace

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // 逐batch执行row-major矩阵乘法
    // =================================== 作业 ===================================

    const auto shape = ValidateMatmulInputs(input, other);
    auto output = std::make_shared<Tensor>(shape.output_dims, DataType::kFLOAT32, input->GetDevice());

    const auto *input_data = static_cast<const float *>(input->DataPtr());
    const auto *other_data = static_cast<const float *>(other->DataPtr());
    auto *output_data = static_cast<float *>(output->DataPtr());
    for (int64_t batch = 0; batch < shape.batch_count; ++batch) {
        Eigen::Map<const RowMajorMatrix> input_matrix(input_data + batch * shape.m * shape.k, shape.m, shape.k);
        Eigen::Map<const RowMajorMatrix> other_matrix(other_data + batch * shape.k * shape.n, shape.k, shape.n);
        Eigen::Map<RowMajorMatrix> output_matrix(output_data + batch * shape.m * shape.n, shape.m, shape.n);
        output_matrix.noalias() = input_matrix * other_matrix;
    }
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // 逐batch计算grad_input和grad_other
    // =================================== 作业 ===================================

    const auto shape = ValidateMatmulInputs(input, other);
    ValidateGradOutput(grad_output, shape, input->GetDevice());

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other->Dims(), DataType::kFLOAT32, other->GetDevice());

    const auto *input_data = static_cast<const float *>(input->DataPtr());
    const auto *other_data = static_cast<const float *>(other->DataPtr());
    const auto *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    auto *grad_input_data = static_cast<float *>(grad_input->DataPtr());
    auto *grad_other_data = static_cast<float *>(grad_other->DataPtr());
    for (int64_t batch = 0; batch < shape.batch_count; ++batch) {
        Eigen::Map<const RowMajorMatrix> input_matrix(input_data + batch * shape.m * shape.k, shape.m, shape.k);
        Eigen::Map<const RowMajorMatrix> other_matrix(other_data + batch * shape.k * shape.n, shape.k, shape.n);
        Eigen::Map<const RowMajorMatrix> grad_output_matrix(grad_output_data + batch * shape.m * shape.n, shape.m,
                                                            shape.n);
        Eigen::Map<RowMajorMatrix> grad_input_matrix(grad_input_data + batch * shape.m * shape.k, shape.m, shape.k);
        Eigen::Map<RowMajorMatrix> grad_other_matrix(grad_other_data + batch * shape.k * shape.n, shape.k, shape.n);
        grad_input_matrix.noalias() = grad_output_matrix * other_matrix.transpose();
        grad_other_matrix.noalias() = input_matrix.transpose() * grad_output_matrix;
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
