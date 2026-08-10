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

__global__ void AdamAccumulateGradKernel(const float *__restrict__ grad_ptr, float *__restrict__ param_ptr,
                                         float *__restrict__ m_ptr, float *__restrict__ v_ptr, float learning_rate,
                                         float beta1, float beta2, float eps, double beta1_t, double beta2_t,
                                         size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        const float g = grad_ptr[idx];
        float &p = param_ptr[idx];
        float &m = m_ptr[idx];
        float &v = v_ptr[idx];

        // 标准 Adam（含偏差校正），公式与 CPU 版一致：
        // m = β1*m + (1-β1)*g，v = β2*v + (1-β2)*g²，p -= lr * m̂ / (√v̂ + eps)
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
        const float m_hat = static_cast<float>(m / (1.0 - beta1_t));
        const float v_hat = static_cast<float>(v / (1.0 - beta2_t));
        p -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
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
    // 偏差校正因子仅依赖步数，与元素无关，host 侧计算一次（double，与 CPU 版一致）
    const double beta1_t = std::pow(static_cast<double>(beta1), static_cast<double>(t));
    const double beta2_t = std::pow(static_cast<double>(beta2), static_cast<double>(t));

    size_t num_elements = param->NumElements();
    // Fix CR#L29-L30：32 位索引范围防护（超过 int 范围时快速失败，避免后半元素静默不处理）
    CHECK_LE(num_elements, static_cast<size_t>(std::numeric_limits<int>::max()));
    // Fix CR#L25-L28：kernel 的 restrict 承诺要求写目标（param/m/v）与读源（grad）互不重叠（原地调用属未定义行为）
    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, param_ptr, m_ptr, v_ptr, learning_rate,
                                                                beta1, beta2, eps, beta1_t, beta2_t, num_elements);
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
