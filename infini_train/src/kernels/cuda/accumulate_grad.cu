#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

__global__ void AdamAccumulateGradKernel(
    const float *grad_data,
    float *param_data,
    float *m_data,
    float *v_data,
    size_t num_elements,
    float learning_rate,
    float beta1,
    float beta2,
    float eps,
    float bias_correction_m,
    float bias_correction_v
) {
    size_t idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        const float gradient = grad_data[idx];

        m_data[idx] =
            beta1 * m_data[idx]
            + (1.0f - beta1) * gradient;

        v_data[idx] =
            beta2 * v_data[idx]
            + (1.0f - beta2)
                * gradient * gradient;

        const float m_hat =
            m_data[idx] / bias_correction_m;

        const float v_hat =
            v_data[idx] / bias_correction_v;

        param_data[idx] -=
            learning_rate * m_hat /
            (__fsqrt_rn(v_hat) + eps);
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

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
     size_t num_elements = grad->NumElements();

    /*
     * 手动计算 beta1^t 和 beta2^t，
     * 避免使用 std::pow 和添加 <cmath>。
     */
    float beta1_power = 1.0f;
    float beta2_power = 1.0f;

    for (int64_t step = 0; step < t; ++step) {
        beta1_power *= beta1;
        beta2_power *= beta2;
    }

    const float bias_correction_m =
        1.0f - beta1_power;

    const float bias_correction_v =
        1.0f - beta2_power;

    const float *grad_data =
        static_cast<const float *>(grad->DataPtr());

    float *param_data =
        static_cast<float *>(param->DataPtr());

    float *m_data =
        static_cast<float *>(m->DataPtr());

    float *v_data =
        static_cast<float *>(v->DataPtr());

    int threads_per_block = 256;

    int num_blocks =
        (num_elements + threads_per_block - 1)
        / threads_per_block;

    AdamAccumulateGradKernel<<<
        num_blocks,
        threads_per_block
    >>>(
        grad_data,
        param_data,
        m_data,
        v_data,
        num_elements,
        learning_rate,
        beta1,
        beta2,
        eps,
        bias_correction_m,
        bias_correction_v
    );
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
