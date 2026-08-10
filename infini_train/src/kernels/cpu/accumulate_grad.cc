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
    // 标准 Adam（含偏差校正）：m = β1*m + (1-β1)*g，v = β2*v + (1-β2)*g²，
    // param -= lr * m̂ / (√v̂ + eps)，其中 m̂ = m/(1-β1^t)，v̂ = v/(1-β2^t)，t 从 1 开始
    for (int64_t idx = 0; idx < param->NumElements(); ++idx) {
        const float g = static_cast<const float *>(grad->DataPtr())[idx];
        float &param_elem = static_cast<float *>(param->DataPtr())[idx];
        float &m_elem = static_cast<float *>(m->DataPtr())[idx];
        float &v_elem = static_cast<float *>(v->DataPtr())[idx];

        m_elem = beta1 * m_elem + (1.0f - beta1) * g;
        v_elem = beta2 * v_elem + (1.0f - beta2) * g * g;
        const float m_hat = m_elem / (1.0f - std::pow(beta1, t));
        const float v_hat = v_elem / (1.0f - std::pow(beta2, t));
        param_elem -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
