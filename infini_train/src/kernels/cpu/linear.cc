#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
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
        // 当前作业场景下，批次维必须逐维一致（不做广播）。
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
    // TODO：实现CPU上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    const auto meta = BuildMatmulMeta(input, other);
    auto output = std::make_shared<Tensor>(meta.output_dims, DataType::kFLOAT32, input->GetDevice());

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *other_ptr = static_cast<const float *>(other->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    // 以 batch 为粒度并行，单个 batch 内使用 Eigen GEMM，兼顾可读性与性能。
#pragma omp parallel for if (meta.batch > 1)
    for (int64_t batch_idx = 0; batch_idx < meta.batch; ++batch_idx) {
        const float *input_batch_ptr = input_ptr + batch_idx * meta.m * meta.k;
        const float *other_batch_ptr = other_ptr + batch_idx * meta.k * meta.n;
        float *output_batch_ptr = output_ptr + batch_idx * meta.m * meta.n;

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input_matrix(
            input_batch_ptr, meta.m, meta.k);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> other_matrix(
            other_batch_ptr, meta.k, meta.n);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_matrix(
            output_batch_ptr, meta.m, meta.n);

        output_matrix.noalias() = input_matrix * other_matrix;
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
    const auto meta = BuildMatmulMeta(input, other);
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_EQ(grad_output_dims.size(), meta.output_dims.size());
    for (size_t dim_idx = 0; dim_idx < grad_output_dims.size(); ++dim_idx) {
        CHECK_EQ(grad_output_dims[dim_idx], meta.output_dims[dim_idx]);
    }
    CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(DataType::kFLOAT32));

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other->Dims(), DataType::kFLOAT32, other->GetDevice());

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *other_ptr = static_cast<const float *>(other->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    float *grad_other_ptr = static_cast<float *>(grad_other->DataPtr());

    // dInput = dOut * Other^T，dOther = Input^T * dOut。
#pragma omp parallel for if (meta.batch > 1)
    for (int64_t batch_idx = 0; batch_idx < meta.batch; ++batch_idx) {
        const float *input_batch_ptr = input_ptr + batch_idx * meta.m * meta.k;
        const float *other_batch_ptr = other_ptr + batch_idx * meta.k * meta.n;
        const float *grad_output_batch_ptr = grad_output_ptr + batch_idx * meta.m * meta.n;
        float *grad_input_batch_ptr = grad_input_ptr + batch_idx * meta.m * meta.k;
        float *grad_other_batch_ptr = grad_other_ptr + batch_idx * meta.k * meta.n;

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input_matrix(
            input_batch_ptr, meta.m, meta.k);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> other_matrix(
            other_batch_ptr, meta.k, meta.n);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> grad_output_matrix(
            grad_output_batch_ptr, meta.m, meta.n);

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> grad_input_matrix(
            grad_input_batch_ptr, meta.m, meta.k);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> grad_other_matrix(
            grad_other_batch_ptr, meta.k, meta.n);

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
