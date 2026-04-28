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

__global__ void AdamAccumulateGradKernel(const float *g, float *p, float *m, float *v, float lr, float beta1,
                                         float beta2, float eps, float bc1, float bc2, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
    p[i] -= lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + eps);
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    const size_t n = grad->NumElements();
    const float bc1 = 1.0f - powf(beta1, static_cast<float>(t));
    const float bc2 = 1.0f - powf(beta2, static_cast<float>(t));
    const int threads_per_block = 256;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(grad->DataPtr()), static_cast<float *>(param->DataPtr()),
        static_cast<float *>(m->DataPtr()), static_cast<float *>(v->DataPtr()), learning_rate, beta1, beta2, eps, bc1,
        bc2, n);
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
