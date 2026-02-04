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
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    const auto n = grad->NumElements();
    const float *g_ptr = static_cast<const float *>(grad->DataPtr());
    float *m_ptr = static_cast<float*>(m->DataPtr());
    float *v_ptr = static_cast<float*>(v->DataPtr());
    float *p_ptr = static_cast<float *>(param->DataPtr());

    const float bias_correction1 = 1.0f - std::pow(beta1,t);
    const float bias_correction2 = 1.0f - std::pow(beta2,t);
    
    for(size_t i = 0;i<n;i++){
        m_ptr[i] = beta1 * m_ptr[i] + (1-beta1) * g_ptr[i];
        v_ptr[i] = beta2 * v_ptr[i] + (1-beta2) * g_ptr[i] * g_ptr[i];
        float m_hat = m_ptr[i] / bias_correction1;
        float v_hat = v_ptr[i] / bias_correction2;
        p_ptr[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
