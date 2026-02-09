#include <cmath>

#include "cuda_runtime_api.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

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
    CUDA_CHECK(cudaGetLastError());
}

__global__ void AdamAccumulateGradKernel(const float *grad_ptr, float *param_ptr, float *m_ptr, float *v_ptr,
                                         float learning_rate, float beta1, float beta2, float eps,
                                         float bias_correction1, float bias_correction2, size_t num_elements) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) {
        return;
    }

    const float g = grad_ptr[idx];
    const float m_new = beta1 * m_ptr[idx] + (1.0f - beta1) * g;
    const float v_new = beta2 * v_ptr[idx] + (1.0f - beta2) * g * g;

    m_ptr[idx] = m_new;
    v_ptr[idx] = v_new;

    const float m_hat = m_new / bias_correction1;
    const float v_hat = v_new / bias_correction2;
    param_ptr[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
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
    CHECK_EQ(static_cast<int>(grad->GetDevice().Type()), static_cast<int>(DeviceType::kCUDA));
    CHECK_EQ(static_cast<int>(param->GetDevice().Type()), static_cast<int>(DeviceType::kCUDA));
    CHECK_EQ(static_cast<int>(m->GetDevice().Type()), static_cast<int>(DeviceType::kCUDA));
    CHECK_EQ(static_cast<int>(v->GetDevice().Type()), static_cast<int>(DeviceType::kCUDA));
    CHECK_EQ(grad->NumElements(), param->NumElements());
    CHECK_EQ(m->NumElements(), param->NumElements());
    CHECK_EQ(v->NumElements(), param->NumElements());

    const size_t num_elements = grad->NumElements();
    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    // 偏置校正只依赖 t，放在 host 侧一次性计算。
    const float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(t));

    const int threads_per_block = 256;
    const int num_blocks = static_cast<int>((num_elements + threads_per_block - 1) / threads_per_block);
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(
        grad_ptr, param_ptr, m_ptr, v_ptr, learning_rate, beta1, beta2, eps, bias_correction1, bias_correction2,
        num_elements);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
#undef CUDA_CHECK
