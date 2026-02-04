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

__global__ void AdamAccumulateGradKernel(const float *grad_ptr,float *p_ptr,float *m_ptr,float *v_ptr,float learning_rate,
                                         float bias_correction1, float bias_correction2,float eps, int64_t t,size_t n,float beta1,float beta2){
                                            int idx = blockDim.x * blockIdx.x + threadIdx.x;
                                            if(idx < n)
                                            {
                                                m_ptr[idx] = beta1 * m_ptr[idx] + (1 - beta1) * grad_ptr[idx];
                                                v_ptr[idx] = beta2 * v_ptr[idx] + (1 - beta2) * grad_ptr[idx] * grad_ptr[idx];
                                                float m_hat = m_ptr[idx] / bias_correction1;
                                                float v_hat = v_ptr[idx] / bias_correction2;
                                                p_ptr[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
                                            }
                                            }

// keep function inside namespace so `Tensor` (in namespace infini_train) is found unqualified
void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    const auto n = grad->NumElements();
    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *p_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    float bias_correction1 = 1.0f - powf(beta1, t);
    float bias_correction2 = 1.0f - powf(beta2, t);
    AdamAccumulateGradKernel<<<num_blocks,threads_per_block>>>(grad_ptr,p_ptr,m_ptr,v_ptr,learning_rate,bias_correction1,bias_correction2,eps,t,n,beta1,beta2);

}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
