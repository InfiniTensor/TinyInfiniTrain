#include <cmath>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

__global__ void AdamAccumulateGradKernel(const float *grad_ptr, float *param_ptr, float *m_ptr, float *v_ptr,
                                         float learning_rate, float beta1, float beta2, float eps,
                                         float bias_correction1, float bias_correction2, size_t num_elements) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    const float gradient = grad_ptr[idx];
    const float first_moment = beta1 * m_ptr[idx] + (1.0f - beta1) * gradient;
    const float second_moment = beta2 * v_ptr[idx] + (1.0f - beta2) * gradient * gradient;
    m_ptr[idx] = first_moment;
    v_ptr[idx] = second_moment;

    const float m_hat = first_moment / bias_correction1;
    const float v_hat = second_moment / bias_correction2;
    param_ptr[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    CHECK(grad);
    CHECK(param);
    CHECK(m);
    CHECK(v);
    CHECK(grad->Dims() == param->Dims());
    CHECK(m->Dims() == param->Dims());
    CHECK(v->Dims() == param->Dims());
    CHECK(grad->Dtype() == DataType::kFLOAT32);
    CHECK(param->Dtype() == DataType::kFLOAT32);
    CHECK(m->Dtype() == DataType::kFLOAT32);
    CHECK(v->Dtype() == DataType::kFLOAT32);
    CHECK(param->GetDevice().IsCUDA());
    CHECK(grad->GetDevice() == param->GetDevice());
    CHECK(m->GetDevice() == param->GetDevice());
    CHECK(v->GetDevice() == param->GetDevice());
    CHECK_GT(t, 0);

    const size_t num_elements = grad->NumElements();
    if (num_elements == 0) {
        return;
    }

    const float bias_correction1 = 1.0f - std::pow(beta1, t);
    const float bias_correction2 = 1.0f - std::pow(beta2, t);
    constexpr int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(grad->DataPtr()), static_cast<float *>(param->DataPtr()),
        static_cast<float *>(m->DataPtr()), static_cast<float *>(v->DataPtr()), learning_rate, beta1, beta2, eps,
        bias_correction1, bias_correction2, num_elements);
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
