#include <cmath>
#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
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
    CHECK(param->GetDevice().IsCPU());
    CHECK(grad->GetDevice() == param->GetDevice());
    CHECK(m->GetDevice() == param->GetDevice());
    CHECK(v->GetDevice() == param->GetDevice());
    CHECK_GT(t, 0);

    const float bias_correction1 = 1.0f - std::pow(beta1, t);
    const float bias_correction2 = 1.0f - std::pow(beta2, t);

    const auto *grad_ptr = static_cast<const float *>(grad->DataPtr());
    auto *param_ptr = static_cast<float *>(param->DataPtr());
    auto *m_ptr = static_cast<float *>(m->DataPtr());
    auto *v_ptr = static_cast<float *>(v->DataPtr());

    for (size_t idx = 0; idx < grad->NumElements(); ++idx) {
        const float gradient = grad_ptr[idx];
        m_ptr[idx] = beta1 * m_ptr[idx] + (1.0f - beta1) * gradient;
        v_ptr[idx] = beta2 * v_ptr[idx] + (1.0f - beta2) * gradient * gradient;

        const float m_hat = m_ptr[idx] / bias_correction1;
        const float v_hat = v_ptr[idx] / bias_correction2;
        param_ptr[idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
