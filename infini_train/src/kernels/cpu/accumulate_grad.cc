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
      CHECK_EQ(grad->NumElements(), param->NumElements());
    CHECK_EQ(m->NumElements(), param->NumElements());
    CHECK_EQ(v->NumElements(), param->NumElements());

    const auto *grad_data =
        static_cast<const float *>(grad->DataPtr());

    auto *param_data =
        static_cast<float *>(param->DataPtr());

    auto *m_data =
        static_cast<float *>(m->DataPtr());

    auto *v_data =
        static_cast<float *>(v->DataPtr());

    const float bias_correction1 =
        1.0f - std::pow(beta1, static_cast<float>(t));

    const float bias_correction2 =
        1.0f - std::pow(beta2, static_cast<float>(t));

    for (int64_t idx = 0; idx < param->NumElements(); ++idx) {
        const float gradient = grad_data[idx];

        // 更新一阶动量
        m_data[idx] =
            beta1 * m_data[idx]
            + (1.0f - beta1) * gradient;

        // 更新二阶动量
        v_data[idx] =
            beta2 * v_data[idx]
            + (1.0f - beta2) * gradient * gradient;

        // 偏差修正
        const float m_hat =
            m_data[idx] / bias_correction1;

        const float v_hat =
            v_data[idx] / bias_correction2;

        // 更新参数
        param_data[idx] -=
            learning_rate * m_hat /
            (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
