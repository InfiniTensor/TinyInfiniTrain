#include <cstddef>
#include <cmath>
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
    CHECK_EQ(static_cast<int>(grad->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(param->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(m->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(v->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(grad->GetDevice().Type()), static_cast<int>(DeviceType::kCPU));
    CHECK_EQ(static_cast<int>(param->GetDevice().Type()), static_cast<int>(DeviceType::kCPU));
    CHECK_EQ(static_cast<int>(m->GetDevice().Type()), static_cast<int>(DeviceType::kCPU));
    CHECK_EQ(static_cast<int>(v->GetDevice().Type()), static_cast<int>(DeviceType::kCPU));
    CHECK_EQ(grad->NumElements(), param->NumElements());
    CHECK_EQ(m->NumElements(), param->NumElements());
    CHECK_EQ(v->NumElements(), param->NumElements());

    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    // 偏置校正项只与步数 t 有关，提前计算避免循环内重复开销。
    const float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(t));

    for (int64_t idx = 0; idx < static_cast<int64_t>(grad->NumElements()); ++idx) {
        const float g = grad_ptr[idx];
        const float m_new = beta1 * m_ptr[idx] + (1.0f - beta1) * g;
        const float v_new = beta2 * v_ptr[idx] + (1.0f - beta2) * g * g;

        m_ptr[idx] = m_new;
        v_ptr[idx] = v_new;

        const float m_hat = m_new / bias_correction1;
        const float v_hat = v_new / bias_correction2;

        // Adam 参数更新：沿负梯度方向更新参数。
        param_ptr[idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
