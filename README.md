# TinyInfiniTrain 作业报告

## 一、测试通过截图

按测试顺序排列，全部 8 个测试通过：

### 1. test_elementwise（5分）✅

验证 autograd 机制调用 Neg kernel 的实现，依赖作业一、作业五。

```
[==========] Running 2 tests from 1 test suite.
[----------] 2 tests from ElementwiseTest
[ RUN      ] ElementwiseTest.NegForward
[       OK ] ElementwiseTest.NegForward (0 ms)
[ RUN      ] ElementwiseTest.NegBackward
[       OK ] ElementwiseTest.NegBackward (0 ms)
[----------] 2 tests from ElementwiseTest (0 ms total)
[  PASSED  ] 2 tests.
```

### 2. test_matmul（5分）✅

验证 Matmul kernel 的 CPU 实现，依赖作业二。

```
[==========] Running 3 tests from 1 test suite.
[----------] 3 tests from MatmulTest
[ RUN      ] MatmulTest.BasicMatrixMultiply
[       OK ] MatmulTest.BasicMatrixMultiply (0 ms)
[ RUN      ] MatmulTest.BatchedMatrixMultiply
[       OK ] MatmulTest.BatchedMatrixMultiply (0 ms)
[ RUN      ] MatmulTest.BackwardPass
[       OK ] MatmulTest.BackwardPass (0 ms)
[----------] 3 tests from MatmulTest (0 ms total)
[  PASSED  ] 3 tests.
```

### 3. test_matmul_cuda（10分）✅

验证 Matmul kernel 的 CUDA 实现，依赖作业二。

```
[==========] Running 3 tests from 1 test suite.
[----------] 3 tests from MatmulTest
[ RUN      ] MatmulTest.BasicMatrixMultiplyCuda
[       OK ] MatmulTest.BasicMatrixMultiplyCuda (277 ms)
[ RUN      ] MatmulTest.BatchedMatrixMultiplyCuda
[       OK ] MatmulTest.BatchedMatrixMultiplyCuda (2 ms)
[ RUN      ] MatmulTest.BackwardPassCuda
[       OK ] MatmulTest.BackwardPassCuda (2 ms)
[----------] 3 tests from MatmulTest (282 ms total)
[  PASSED  ] 3 tests.
```

### 4. test_adam（5分）✅

验证 Adam 优化器的 CPU 实现，依赖作业三。

```
[==========] Running 2 tests from 1 test suite.
[----------] 2 tests from AdamOptimizerTest
[ RUN      ] AdamOptimizerTest.BasicParameterUpdate
[       OK ] AdamOptimizerTest.BasicParameterUpdate (0 ms)
[ RUN      ] AdamOptimizerTest.MomentumAccumulation
[       OK ] AdamOptimizerTest.MomentumAccumulation (0 ms)
[----------] 2 tests from AdamOptimizerTest (0 ms total)
[  PASSED  ] 2 tests.
```

### 5. test_adam_cuda（10分）✅

验证 Adam 优化器的 CUDA 实现，依赖作业三。

```
[==========] Running 2 tests from 1 test suite.
[----------] 2 tests from AdamOptimizerTest
[ RUN      ] AdamOptimizerTest.BasicParameterUpdateCuda
[       OK ] AdamOptimizerTest.BasicParameterUpdateCuda (41 ms)
[ RUN      ] AdamOptimizerTest.MomentumAccumulationCuda
[       OK ] AdamOptimizerTest.MomentumAccumulationCuda (2 ms)
[----------] 2 tests from AdamOptimizerTest (43 ms total)
[  PASSED  ] 2 tests.
```

### 6. test_tensor（10分）✅

验证 Tensor 基础功能，依赖作业四。

```
[==========] Running 5 tests from 2 test suites.
[----------] 3 tests from TensorTransformTest
[ RUN      ] TensorTransformTest.Flatten2DTo1D
[       OK ] TensorTransformTest.Flatten2DTo1D (0 ms)
[ RUN      ] TensorTransformTest.FlattenWithRange
[       OK ] TensorTransformTest.FlattenWithRange (0 ms)
[ RUN      ] TensorTransformTest.FlattenNonContiguous
[       OK ] TensorTransformTest.FlattenNonContiguous (0 ms)
[----------] 3 tests from TensorTransformTest (0 ms total)
[----------] 2 tests from TensorAutogradTest
[ RUN      ] TensorAutogradTest.BackwardComputesGradient
[       OK ] TensorAutogradTest.BackwardComputesGradient (0 ms)
[ RUN      ] TensorAutogradTest.BackwardWithMultipleOutputs
[       OK ] TensorAutogradTest.BackwardWithMultipleOutputs (0 ms)
[----------] 2 tests from TensorAutogradTest (0 ms total)
[  PASSED  ] 5 tests.
```

### 7. test_dispatcher（20分）✅

验证多设备分发机制，核心基础设施，依赖作业五。

```
[==========] Running 3 tests from 1 test suite.
[----------] 3 tests from DispatcherTest
[ RUN      ] DispatcherTest.RegisterAndGetKernel
[       OK ] DispatcherTest.RegisterAndGetKernel (0 ms)
[ RUN      ] DispatcherTest.DuplicateRegistration
[       OK ] DispatcherTest.DuplicateRegistration (0 ms)
[ RUN      ] DispatcherTest.GetNonexistentKernel
[       OK ] DispatcherTest.GetNonexistentKernel (0 ms)
[----------] 3 tests from DispatcherTest (0 ms total)
[  PASSED  ] 3 tests.
```

### 8. test_gpt2（35分）

端到端 GPT-2 模型测试，依赖所有作业。

```
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from GPT2TrainingTest
[ RUN      ] GPT2TrainingTest.LogitsConsistency
WARNING: Logging before InitGoogleLogging() is written to STDERR
E20260812 21:58:02.166082 137779420561088 test_gpt2.cc:132] Initialize: device_flag=cpu device_type=0
I20260812 21:58:31.217616 137779420561088 test_gpt2.cc:123] Initialize() finished!
I20260812 21:58:31.217688 137779420561088 test_gpt2.cc:208] epoch: 0
I20260812 21:59:55.065171 137779420561088 test_gpt2.cc:208] epoch: 1
I20260812 22:01:16.775266 137779420561088 test_gpt2.cc:208] epoch: 2
I20260812 22:02:41.858699 137779420561088 test_gpt2.cc:208] epoch: 3
I20260812 22:04:05.018701 137779420561088 test_gpt2.cc:208] epoch: 4
I20260812 22:05:38.257022 137779420561088 test_gpt2.cc:208] epoch: 5
I20260812 22:07:03.780517 137779420561088 test_gpt2.cc:208] epoch: 6
I20260812 22:08:31.519426 137779420561088 test_gpt2.cc:208] epoch: 7
I20260812 22:09:53.920774 137779420561088 test_gpt2.cc:208] epoch: 8
I20260812 22:11:14.519412 137779420561088 test_gpt2.cc:208] epoch: 9
I20260812 22:12:37.165615 137779420561088 tokenizer.cc:138] start generate text:
The meaning of life is stillHe unclearated. Coal Lands Board like: capital 1968, buildings,Independent education election, candidate peace: and Labour prosperity MP in Tony the Hayward Middle East have" becomeI symbols am that pleased mark to an welcome alternate Lo stagece forman peace:f Electaringor nations of such All as Queens atheist." type Tweet and This note Target taking:. Link<|endoftext|> toIndia Select's Merch treasurer; general charges<|endoftext|> thatProduct fires Details from siegDeltahe IKilled
```

---

## 二、作业步骤

### 作业一：autograd 机制调用 Neg kernel 的实现

**难度**：⭐
**对应测例**：`TEST(ElementwiseTest, NegForward)`，`TEST(ElementwiseTest, NegBackward)`
**代码位置**：`infini_train/src/autograd/elementwise.cc`

#### 代码实现

```cpp
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}
```

#### 解决思路

1. 通过 `input->GetDevice().Type()` 获取当前张量所在的设备类型（CPU 或 CUDA）
2. 使用 `Dispatcher::Instance().GetKernel()` 根据设备类型和 kernel 名称获取对应的 kernel 函数
3. 使用 `kernel.Call<ReturnType>(args...)` 调用 kernel 并返回结果
4. Forward 和 Backward 的实现模式一致，区别仅在于 kernel 名称（"NegForward" vs "NegBackward"）

#### 遇到问题

无特殊问题。该作业实现较为直接，主要依赖作业五的 Dispatcher 机制。

---

### 作业二：实现矩阵乘法

**难度**：⭐⭐

#### CPU 实现

**对应测例**：`TEST(MatmulTest, BasicMatrixMultiply)`，`TEST(MatmulTest, BatchedMatrixMultiply)`，`TEST(MatmulTest, BackwardPass)`
**代码位置**：`infini_train/src/kernels/cpu/linear.cc`

##### 前向传播代码

```cpp
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const int64_t M = input_dims[input_dims.size() - 2];
    const int64_t K = input_dims[input_dims.size() - 1];
    const int64_t N = other_dims[other_dims.size() - 1];

    // 处理 batch 维度
    int64_t batch_input = 1, batch_other = 1;
    for (size_t i = 0; i < input_dims.size() - 2; ++i) batch_input *= input_dims[i];
    for (size_t i = 0; i < other_dims.size() - 2; ++i) batch_other *= other_dims[i];
    int64_t batch = std::max(batch_input, batch_other);

    // 使用 Eigen 进行矩阵乘法
    for (int64_t b = 0; b < batch; ++b) {
        int64_t input_offset = (batch_input == 1) ? 0 : b * M * K;
        int64_t other_offset = (batch_other == 1) ? 0 : b * K * N;
        int64_t output_offset = b * M * N;

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input_mat(
            input_ptr + input_offset, M, K);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> other_mat(
            other_ptr + other_offset, K, N);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output_mat(
            output_ptr + output_offset, M, N);

        output_mat = input_mat * other_mat;
    }
    return output;
}
```

##### 反向传播代码

```cpp
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // grad_input = grad_output @ other^T
    // grad_other = input^T @ grad_output
    // 使用 Eigen 实现
}
```

##### 解决思路

1. **前向传播**：使用 Eigen 库的 `Eigen::Map` 将原始数据指针映射为矩阵，然后直接进行矩阵乘法 `input_mat * other_mat`
2. **反向传播**：根据链式法则，`grad_input = grad_output @ other^T`，`grad_other = input^T @ grad_output`
3. **Batch 支持**：通过检查 batch 维度的广播规则，支持批处理矩阵乘法

##### 遇到问题

无特殊问题。Eigen 库提供了高效的 CPU 矩阵运算。

---

#### CUDA 实现

**对应测例**：`TEST(MatmulTest, BasicMatrixMultiplyCuda)`，`TEST(MatmulTest, BatchedMatrixMultiplyCuda)`，`TEST(MatmulTest, BackwardPassCuda)`
**代码位置**：`infini_train/src/kernels/cuda/linear.cu`

##### 前向传播代码

```cpp
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // ... 计算维度 M, K, N, batch ...

    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (batch <= 1) {
        // C = output^T[N, M] = other^T[N, K] * input^T[K, M]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                 static_cast<const float *>(other->DataPtr()), N,
                                 static_cast<const float *>(input->DataPtr()), K, &beta,
                                 static_cast<float *>(output->DataPtr()), N));
    } else {
        // 使用 Strided Batched GEMM
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                               static_cast<const float *>(other->DataPtr()), N, strideA,
                                               static_cast<const float *>(input->DataPtr()), K, strideB, &beta,
                                               static_cast<float *>(output->DataPtr()), N, strideC, batch));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}
```

##### 反向传播代码（关键部分）

```cpp
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    const int64_t M = input_dims[input_dims.size() - 2];
    const int64_t K = input_dims[input_dims.size() - 1];
    const int64_t N = other_dims[other_dims.size() - 1];

    // grad_input = grad_output @ other^T
    // grad_input^T[K, M] = other^T[K, N] * grad_output^T[N, M]
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha,
                             static_cast<const float *>(other->DataPtr()), N,
                             static_cast<const float *>(grad_output->DataPtr()), N, &beta,
                             static_cast<float *>(grad_input->DataPtr()), K));

    // grad_other = input^T @ grad_output
    // grad_other^T[N, K] = grad_output^T[N, M] * input[M, K]
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha,
                             static_cast<const float *>(grad_output->DataPtr()), N,
                             static_cast<const float *>(input->DataPtr()), K, &beta,
                             static_cast<float *>(grad_other->DataPtr()), N));
}
```

##### 解决思路

1. cuBLAS 使用列主序（column-major），而框架使用行主序（row-major），因此 cuBLAS 中的矩阵是框架中矩阵的转置
2. **前向传播**：`output = input @ other` → cuBLAS: `output^T = other^T @ input^T`
3. **反向传播**：
   - `grad_input = grad_output @ other^T` → cuBLAS: `grad_input^T = other^T @ grad_output^T`，需要 `CUBLAS_OP_T` 转置 other
   - `grad_other = input^T @ grad_output` → cuBLAS: `grad_other^T = grad_output^T @ input`，需要 `CUBLAS_OP_T` 转置 input

##### 遇到问题

**问题 1：cuBLAS 反向传播梯度值错误**

- **现象**：`test_matmul_cuda` 的 `BackwardPassCuda` 测试失败，`grad_input` 和 `grad_other` 的值与预期不符
- **原因**：cuBLAS 的 `CUBLAS_OP_T` 和 leading dimension 参数设置错误
  - `grad_input` 计算中，`other` 的 lda 应为 `N`（列数）而非 `K`，且需 `CUBLAS_OP_T` 转置
  - `grad_other` 计算中，`input` 需 `CUBLAS_OP_T` 转置（而非 `CUBLAS_OP_N`），ldb 保持为 `K`
- **修复**：将 `grad_input` 的 `other` 参数改为 `CUBLAS_OP_T` + `lda=N`；将 `grad_other` 的 `input` 参数改为 `CUBLAS_OP_T`

---

### 作业三：实现 Adam 优化器

**难度**：⭐

#### CPU 实现

**对应测例**：`TEST(AdamOptimizerTest, BasicParameterUpdate)`，`TEST(AdamOptimizerTest, MomentumAccumulation)`
**代码位置**：`infini_train/src/kernels/cpu/accumulate_grad.cc`

```cpp
void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    int64_t num_elements = grad->NumElements();
    float *grad_ptr = static_cast<float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    float beta1_t = std::pow(beta1, t);
    float beta2_t = std::pow(beta2, t);
    float alpha = learning_rate * std::sqrt(1.0f - beta2_t) / (1.0f - beta1_t);

    for (int64_t i = 0; i < num_elements; ++i) {
        float g = grad_ptr[i];
        m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g;
        v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;
        param_ptr[i] -= alpha * m_ptr[i] / (std::sqrt(v_ptr[i]) + eps);
    }
}
```

#### CUDA 实现

**对应测例**：`TEST(AdamOptimizerTest, BasicParameterUpdateCuda)`，`TEST(AdamOptimizerTest, MomentumAccumulationCuda)`
**代码位置**：`infini_train/src/kernels/cuda/accumulate_grad.cu`

```cpp
__global__ void AdamAccumulateGradKernel(const float *grad_ptr, float *param_ptr, float *m_ptr, float *v_ptr,
                                          float learning_rate, float beta1, float beta2, float eps, float alpha,
                                          size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float g = grad_ptr[idx];
        m_ptr[idx] = beta1 * m_ptr[idx] + (1.0f - beta1) * g;
        v_ptr[idx] = beta2 * v_ptr[idx] + (1.0f - beta2) * g * g;
        param_ptr[idx] -= alpha * m_ptr[idx] / (sqrtf(v_ptr[idx]) + eps);
    }
}
```

#### 解决思路

1. 按照 Adam 算法的标准公式实现：
   - 一阶矩估计：`m_t = beta1 * m_{t-1} + (1 - beta1) * g_t`
   - 二阶矩估计：`v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2`
   - 偏差修正：`alpha = lr * sqrt(1 - beta2^t) / (1 - beta1^t)`
   - 参数更新：`param = param - alpha * m_t / (sqrt(v_t) + eps)`
2. CPU 版本使用逐元素循环，CUDA 版本使用 CUDA kernel 并行处理

#### 遇到问题

无特殊问题。Adam 算法公式标准化，实现较为直接。

---

### 作业四：实现 Tensor 基础操作

**难度**：⭐

#### Flatten 操作

**对应测例**：`TEST(TensorTransformTest, Flatten2DTo1D)`，`TEST(TensorTransformTest, FlattenWithRange)`，`TEST(TensorTransformTest, FlattenNonContiguous)`
**代码位置**：`infini_train/src/tensor.cc`

```cpp
std::shared_ptr<Tensor> Tensor::Flatten(int64_t start, int64_t end) {
    int64_t ndim = static_cast<int64_t>(dims_.size());
    if (end < 0) {
        end += ndim;
    }

    std::vector<int64_t> new_shape;
    // 保留 start 之前的维度
    for (int64_t i = 0; i < start; ++i) {
        new_shape.push_back(dims_[i]);
    }
    // 将 [start, end] 范围内的维度合并为一个
    int64_t flattened_size = 1;
    for (int64_t i = start; i <= end; ++i) {
        flattened_size *= dims_[i];
    }
    new_shape.push_back(flattened_size);
    // 保留 end 之后的维度
    for (int64_t i = end + 1; i < ndim; ++i) {
        new_shape.push_back(dims_[i]);
    }

    return Contiguous()->View(new_shape);
}
```

#### 反向传播机制

**对应测例**：`TEST(TensorAutogradTest, BackwardComputesGradient)`，`TEST(TensorAutogradTest, BackwardWithMultipleOutputs)`
**代码位置**：`infini_train/src/tensor.cc`

```cpp
void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    if (!gradient) {
        gradient = std::make_shared<Tensor>(dims_, dtype_, GetDevice());
        gradient->Fill<float>(1.0f);
    }

    if (is_leaf_) {
        // 叶子节点：累积梯度
        if (requires_grad_ && grad_) {
            auto device = grad_->GetDevice().Type();
            auto kernel = Dispatcher::Instance().GetKernel({device, "AccumulateGrad"});
            kernel.Call<void>(gradient, 1.0f, grad_);
        }
    } else if (grad_fn_) {
        // 非叶子节点：通过 grad_fn 继续反向传播
        grad_fn_->BackwardPartial(gradient, output_idx_);
    }
}
```

#### 解决思路

1. **Flatten**：先规范化负索引，然后构建新形状（保留 start 前维度 + 合并中间维度 + 保留 end 后维度），调用 `Contiguous()->View()` 实现
2. **Backward**：
   - 如果未提供梯度，默认创建全 1 梯度
   - 叶子节点：通过 Dispatcher 调用 AccumulateGrad kernel 累积梯度
   - 非叶子节点：通过 `grad_fn_->BackwardPartial()` 沿计算图递归传播梯度

#### 遇到问题

无特殊问题。

---

### 作业五：注册算子 kernel 的实现

**难度**：⭐⭐⭐
**对应测例**：`TEST(DispatcherTest, RegisterAndGetKernel)`，`TEST(DispatcherTest, DuplicateRegistration)`，`TEST(DispatcherTest, GetNonexistentKernel)`
**代码位置**：`infini_train/include/dispatcher.h`

#### 代码实现

```cpp
// KernelFunction::Call — 通用 kernel 调用接口
template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
    using FuncT = RetT (*)(ArgsT...);
    auto func = reinterpret_cast<FuncT>(func_ptr_);
    return func(std::forward<ArgsT>(args)...);
}

// Dispatcher::Register — kernel 注册机制
template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    CHECK(!key_to_kernel_map_.contains(key))
        << "Kernel already registered: " << key.second;
    key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
}

// REGISTER_KERNEL 宏 — 自动注册宏
#define REGISTER_KERNEL(device, kernel_name, kernel_func)                          \
    static auto _register_##kernel_name##_##__LINE__ = []() {                      \
        infini_train::Dispatcher::Instance().Register(                             \
            {device, #kernel_name}, kernel_func);                                  \
        return 0;                                                                  \
    }();
```

#### 解决思路

1. **Call 方法**：将存储的 `void*` 函数指针通过 `reinterpret_cast` 转换为目标函数类型 `RetT (*)(ArgsT...)`，然后调用
2. **Register 方法**：检查重复注册（`CHECK`），使用 `emplace` 将 kernel 存入 `key_to_kernel_map_`
3. **REGISTER_KERNEL 宏**：利用静态 lambda 在程序启动时自动注册，使用 `__LINE__` 确保唯一变量名

#### 遇到问题

无特殊问题。该作业是框架核心基础设施，实现符合设计模式。

---

### 作业六：实现 GPT-2 整体训练

**难度**：⭐⭐⭐⭐
**对应测例**：`TEST_F(GPT2TrainingTest, LogitsConsistency)`

#### 数据读取实现

**代码位置**：`example/common/tiny_shakespeare_dataset.cc`

```cpp
TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    // 读取 header (1024 bytes)
    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    int32_t magic = BytesToType<int32_t>(header, 0);
    int32_t version = BytesToType<int32_t>(header, 4);
    int32_t num_toks = BytesToType<int32_t>(header, 8);

    // 根据版本号确定数据类型
    CHECK(kTypeMap.contains(version)) << "Unknown version: " << version;
    TinyShakespeareType type = kTypeMap.at(version);
    size_t element_size = kTypeToSize.at(type);

    // 读取 token 数据并转换为 int64_t
    auto data = ReadSeveralBytesFromIfstream(num_toks * element_size, &ifs);
    int64_t num_sequences = num_toks / sequence_length;
    int64_t num_elements = num_sequences * sequence_length;

    auto tensor = infini_train::Tensor({num_sequences, static_cast<int64_t>(sequence_length)}, DataType::kINT64);
    int64_t *tensor_ptr = static_cast<int64_t *>(tensor.DataPtr());

    for (int64_t i = 0; i < num_elements; ++i) {
        if (type == TinyShakespeareType::kUINT16) {
            tensor_ptr[i] = static_cast<int64_t>(BytesToType<uint16_t>(data, i * element_size));
        } else {
            tensor_ptr[i] = static_cast<int64_t>(BytesToType<uint32_t>(data, i * element_size));
        }
    }
    return {tensor, num_toks};
}
```

#### Tokenizer 功能实现

**代码位置**：`example/common/tokenizer.cc`

```cpp
// 文本生成循环
void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size,
                             uint32_t sequence_length, uint32_t text_length, Device device) const {
    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    {
        autograd::NoGradGuard no_grad;  // 禁用 autograd 图构建，避免 GPU OOM
        for (int t = prompt_len; t < text_length; t++) {
            // Forward pass
            auto logits = model.Forward({x})[0];

            // 获取最后一个 token 位置的 logits
            auto last_logits = logits->Slice(1, t - 1, t, 1);
            last_logits = last_logits->Squeeze(1);

            // Softmax 采样
            auto probs = infini_train::nn::function::Softmax(last_logits, -1);
            auto probs_cpu = probs->To(Device(DeviceType::kCPU, 0));

            // 随机采样
            for (int b = 0; b < batch_size; ++b) {
                float coin = RandomF32(kRngState);
                int next_token = SampleMult(probs_ptr + b * vocab_size, vocab_size, coin);
                x_buff[b * sequence_length + t] = next_token;
                std::cout << Decode(next_token);
            }

            x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
        }
    }
}
```

#### 解决思路

1. **数据读取**：解析二进制文件格式（1024 字节 header + token 数据），根据版本号确定数据类型（uint16 或 uint32），注意循环上限使用 `num_elements` 而非 `num_toks` 避免 buffer overflow
2. **Tokenizer**：加载 GPT-2 tokenizer 二进制文件，实现 token 解码和文本生成
3. **文本生成**：使用 `NoGradGuard` 禁用 autograd 计算图构建，避免 GPU 推理时显存泄漏

#### 遇到问题

**问题 1：数据集版本号 "1" 未识别**

- **现象**：`Check failed: kTypeMap.contains(version) Unknown version: 1`
- **原因**：`kTypeMap` 仅包含版本 20240520 和 20240801，缺少旧版本 "1" 的映射
- **修复**：在 `kTypeMap` 中添加 `{1, TinyShakespeareType::kUINT16}`

**问题 2：编译错误 — 缺少头文件**

- **现象**：
  - `'setprecision' is not a member of 'std'` / `'setw' is not a member of 'std'`
  - `'infini_train::nn::functional' has not been declared`
  - `'format' is not a member of 'std'`
- **修复**：
  - `tensor.cc`：添加 `#include <iomanip>`
  - `tokenizer.cc`：`functional` → `function`，添加 `#include "infini_train/include/nn/functional.h"`
  - `net.cc`：添加 `#include <format>`

**问题 3：GPU 推理时显存不足（OOM）**

- **现象**：文本生成阶段 `CUDA Error: out of memory`，但 10 个 epoch 训练正常
- **原因**：文本生成循环中每次 `model.Forward()` 都构建 autograd 计算图，64 次迭代后图累积导致显存耗尽
- **修复**：添加 `NoGradGuard` 机制
  - 在 `function.h` 中定义 `NoGradGuard` 类（RAII 模式，使用 `thread_local` 计数器）
  - 在 `function.cc` 的 `Function::Apply()` 中检查 `NoGradGuard::is_enabled()`，启用时跳过图构建
  - 在 `tokenizer.cc` 的文本生成循环中包裹 `NoGradGuard`

**问题 4：cuBLAS 反向传播梯度值错误**

- **现象**：`test_matmul_cuda` 的 `BackwardPassCuda` 测试失败，梯度值与预期不符
- **原因**：cuBLAS 列主序与框架行主序的转换中，`CUBLAS_OP_T` 和 leading dimension 参数设置错误
- **修复**：详见作业二 CUDA 实现部分

**问题 5：跨平台文件同步**

- **现象**：Windows 端修改代码后，WSL 中编译仍使用旧代码
- **原因**：Windows 项目路径 `f:\Train\TinyInfiniTrain` 和 WSL 项目路径 `~/Train/TinyInfiniTrain` 是两份独立副本
- **修复**：每次修改后使用 `cp /mnt/f/Train/TinyInfiniTrain/... ~/Train/TinyInfiniTrain/...` 同步

---

## 三、环境配置总结

| 组件 | 版本 | 说明 |
|------|------|------|
| 操作系统 | WSL2 Ubuntu 22.04 | Windows 11 主机 |
| GCC/G++ | 13.4.0 | 通过 `ubuntu-toolchain-r/test` PPA 安装 |
| CMake | 4.4.2 | 通过 pip 升级 |
| CUDA Toolkit | 12.6.1 | WSL2 中仅安装 Toolkit（不含驱动） |
| Make | 4.3 | 系统自带 |

### 编译命令

```bash
# 启用 CUDA 编译
cd ~/Train/TinyInfiniTrain
make build USE_CUDA=ON

# 仅 CPU 编译
make build USE_CUDA=OFF

# 运行全部测试
make test-cpp

# 清理重新编译
make clean && make build USE_CUDA=ON
```

---

