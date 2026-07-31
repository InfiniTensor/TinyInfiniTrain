#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "glog/logging.h"

namespace infini_train::cuda_detail {

inline void CudaCheck(cudaError_t status, const char *expression, const char *file, int line) {
    if (status != cudaSuccess) {
        LOG(FATAL) << "CUDA error while evaluating `" << expression << "` at " << file << ":" << line
                   << ": code=" << static_cast<int>(status) << " (" << cudaGetErrorString(status) << ")";
    }
}

inline void CublasCheck(cublasStatus_t status, const char *expression, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        LOG(FATAL) << "cuBLAS error while evaluating `" << expression << "` at " << file << ":" << line
                   << ": code=" << static_cast<int>(status) << " (" << cublasGetStatusString(status) << ")";
    }
}

inline void CudaKernelCheck(const char *file, int line) {
    CudaCheck(cudaGetLastError(), "cudaGetLastError()", file, line);

#if defined(TINY_DEBUG_CUDA_SYNC) && TINY_DEBUG_CUDA_SYNC
    CudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", file, line);
#endif
}

} // namespace infini_train::cuda_detail

#define CUDA_CHECK(expression)                                                                                         \
    do {                                                                                                               \
        ::infini_train::cuda_detail::CudaCheck((expression), #expression, __FILE__, __LINE__);                         \
    } while (0)

#define CUBLAS_CHECK(expression)                                                                                       \
    do {                                                                                                               \
        ::infini_train::cuda_detail::CublasCheck((expression), #expression, __FILE__, __LINE__);                       \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                                                            \
    do {                                                                                                               \
        ::infini_train::cuda_detail::CudaKernelCheck(__FILE__, __LINE__);                                              \
    } while (0)
