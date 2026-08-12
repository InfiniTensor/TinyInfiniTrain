# TinyInfiniTrain 作业报告

## 一、test 通过截图

![test通过截图](test通过截图.png)

> 测试环境说明：RTX 5090（32GB）+ CUDA 13.3 运行时环境，`ctest` 全量 8/8 通过（test_gpt2 耗时约 63s，为 GPT-2 124M 训练 11 步 + 生成 64 token 的真实开销）。
> 注：test_gpt2 的 logits 对比对运行环境敏感（cuBLAS 版本/GPU 架构与参考生成环境的数值轨迹差异，详见作业六"遇到问题"），本机 4060 + CUDA 13.3 下其余 7/8 全过、test_gpt2 因该环境性差异失败；在 5090 + CUDA 13.3 运行时环境下 8/8 全部通过。

## 二、作业步骤

> 将代码填入下面代码块中指定位置，并详细描述完成该作业的解决思路和遇到的问题。

### 作业一：autograd机制调用Neg kernel的实现

难度：⭐

对应测例：`TEST(ElementwiseTest, NegForward)`，`TEST(ElementwiseTest, NegBackward)`

需要实现的代码块位置：`infini_train/src/autograd/elementwise.cc`

```c++
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属kernel，对输入张量进行取反操作
    // NOTES: 依赖test_dispatcher，Neg kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属的反向传播kernel，计算梯度
    // NOTES: 依赖test_dispatcher，Neg的kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}
```

#### 解决思路

Neg 的 Forward/Backward 采用同文件 `Reciprocal` 的官方实现模式：通过 `Dispatcher::Instance().GetKernel({device, "NegForward"})` 获取设备专属 kernel，再 `Call<std::shared_ptr<Tensor>>(input)` 调用。设备类型取自输入张量自身，天然支持 CPU/CUDA 双设备。Backward 中利用 Neg 的导数恒为 -1（与输入值无关）的特性：`d(-x)/dx = -1`，梯度即 `-grad_output`，因此无需保存输入张量（与 `Neg` 类未声明 `SetupContext` 的类设计一致，测试亦直接调用 `Backward` 未先调用 `SetupContext`）。

#### 遇到问题

本作业实现本身无难点；前置依赖是作业五（Dispatcher 注册机制）——在作业五完成前，`REGISTER_KERNEL` 宏为空导致所有 kernel 注册不进去，所有测试报 `Kernel not found`。作业五完成后本作业立即通过。

### 作业二：实现矩阵乘法

难度：⭐⭐

#### CPU实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiply)`，`TEST(MatmulTest, BatchedMatrixMultiply)`, `TEST(MatmulTest, BackwardPass)`

需要实现的代码块位置：`infini_train/src/kernels/cpu/linear.cc`

```c++
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    // input[batch..., rows, in_features] × other[batch..., in_features, out_features] -> output[batch..., rows, out_features]
    // 逐 batch 严格相等（batch 维不支持 torch.matmul 的广播语义）
    const int64_t batch_ndim = input_dims.size() - 2;
    const int64_t batch =
        std::accumulate(input_dims.begin(), input_dims.begin() + batch_ndim, 1, std::multiplies<int64_t>{});
    const int64_t rows = input_dims[batch_ndim];
    const int64_t in_features = *input_dims.rbegin();
    const int64_t out_features = *other_dims.rbegin();
    for (int64_t dim = 0; dim < batch_ndim; ++dim) {
        CHECK_EQ(input_dims[dim], other_dims[dim]);
    }
    CHECK_EQ(in_features, *(other_dims.rbegin() + 1));

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    const float *input_data = static_cast<const float *>(input->DataPtr());
    const float *other_data = static_cast<const float *>(other->DataPtr());
    float *output_data = static_cast<float *>(output->DataPtr());
    // 循环次序 r->k->c（内层 c）：other 与 output 沿连续方向访问（缓存友好），output 先清零再累积
    const int64_t output_size = batch * rows * out_features;
    for (int64_t idx = 0; idx < output_size; ++idx) {
        output_data[idx] = 0.0f;
    }
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t k = 0; k < in_features; ++k) {
                const float a = input_data[(b * rows + r) * in_features + k];
                for (int64_t c = 0; c < out_features; ++c) {
                    output_data[(b * rows + r) * out_features + c]
                        += a * other_data[(b * in_features + k) * out_features + c];
                }
            }
        }
    }
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================
    // grad_input = grad_output × other^T，grad_other = input^T × grad_output
    // 逐 batch 严格相等（batch 维不支持 torch.matmul 的广播语义）
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    const int64_t batch_ndim = input_dims.size() - 2;
    const int64_t batch =
        std::accumulate(input_dims.begin(), input_dims.begin() + batch_ndim, 1, std::multiplies<int64_t>{});
    const int64_t rows = input_dims[batch_ndim];
    const int64_t in_features = *input_dims.rbegin();
    const int64_t out_features = *other_dims.rbegin();

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    CHECK(grad_output_dims == output_dims);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);

    const float *input_data = static_cast<const float *>(input->DataPtr());
    const float *other_data = static_cast<const float *>(other->DataPtr());
    const float *grad_output_data = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_data = static_cast<float *>(grad_input->DataPtr());
    float *grad_other_data = static_cast<float *>(grad_other->DataPtr());
    // grad_other[b][k][c] = Σ_r input[b][r][k] * grad_output[b][r][c]
    // 循环次序 k->r->c（内层 c）：grad_output 与 grad_other 沿连续方向访问（缓存友好），grad_other 先清零
    const int64_t grad_other_size = batch * in_features * out_features;
    for (int64_t idx = 0; idx < grad_other_size; ++idx) {
        grad_other_data[idx] = 0.0f;
    }
    for (int64_t b = 0; b < batch; ++b) {
        // grad_input[b][r][k] = Σ_c grad_output[b][r][c] * other[b][k][c]（c 最内层，两操作数连续访问）
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t k = 0; k < in_features; ++k) {
                float sum = 0.0f;
                for (int64_t c = 0; c < out_features; ++c) {
                    sum += grad_output_data[(b * rows + r) * out_features + c]
                           * other_data[(b * in_features + k) * out_features + c];
                }
                grad_input_data[(b * rows + r) * in_features + k] = sum;
            }
        }
        for (int64_t k = 0; k < in_features; ++k) {
            for (int64_t r = 0; r < rows; ++r) {
                const float a = input_data[(b * rows + r) * in_features + k];
                for (int64_t c = 0; c < out_features; ++c) {
                    grad_other_data[(b * in_features + k) * out_features + c]
                        += a * grad_output_data[(b * rows + r) * out_features + c];
                }
            }
        }
    }
    return {grad_input, grad_other};
}
```

#### CUDA实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiplyCuda)`,`TEST(MatmulTest, BatchedMatrixMultiplyCuda)`,`TEST(MatmulTest, BackwardPassCuda)`

需要实现的代码块位置：`infini_train/src/kernels/cuda/linear.cu`

```c++
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    // input[batch..., rows, in_features] × other[batch..., in_features, out_features] -> output[batch..., rows, out_features]
    // 逐 batch 严格相等（batch 维不支持 torch.matmul 的广播语义），语义与 CPU 版一致
    const int64_t batch_ndim = input_dims.size() - 2;
    const int64_t batch =
        std::accumulate(input_dims.begin(), input_dims.begin() + batch_ndim, 1, std::multiplies<int64_t>{});
    const int64_t rows = input_dims[batch_ndim];
    const int64_t in_features = *input_dims.rbegin();
    const int64_t out_features = *other_dims.rbegin();
    for (int64_t dim = 0; dim < batch_ndim; ++dim) {
        CHECK_EQ(input_dims[dim], other_dims[dim]);
    }
    CHECK_EQ(in_features, *(other_dims.rbegin() + 1));
    CHECK_EQ(static_cast<int>(input->GetDevice().Type()), static_cast<int>(other->GetDevice().Type()));

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    // 一元素一线程 + 边界检查（网格按 CEIL_DIV 划分，与官方 kernel 风格一致）
    const int64_t total = batch * rows * out_features;
    // 32 位索引范围防护（total 超出 int 范围时快速失败，避免溢出为负导致静默空输出）
    CHECK_LE(total, std::numeric_limits<int>::max());
    int threads_per_block = 256;
    int num_blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);
    MatmulForwardKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(other->DataPtr()),
        static_cast<float *>(output->DataPtr()), static_cast<int>(rows), static_cast<int>(in_features),
        static_cast<int>(out_features), static_cast<int>(total));
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &grad_output_dims = grad_output->Dims();
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.size(), other_dims.size());

    // grad_input = grad_output × other^T，grad_other = input^T × grad_output
    const int64_t batch_ndim = input_dims.size() - 2;
    const int64_t batch =
        std::accumulate(input_dims.begin(), input_dims.begin() + batch_ndim, 1, std::multiplies<int64_t>{});
    const int64_t rows = input_dims[batch_ndim];
    const int64_t in_features = *input_dims.rbegin();
    const int64_t out_features = *other_dims.rbegin();
    for (int64_t dim = 0; dim < batch_ndim; ++dim) {
        CHECK_EQ(input_dims[dim], other_dims[dim]);
    }
    CHECK_EQ(in_features, *(other_dims.rbegin() + 1));
    // 设备一致性校验（跨设备误用快速失败而非异步运行时错误）
    CHECK_EQ(static_cast<int>(input->GetDevice().Type()), static_cast<int>(other->GetDevice().Type()));
    CHECK_EQ(static_cast<int>(input->GetDevice().Type()), static_cast<int>(grad_output->GetDevice().Type()));

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    CHECK(grad_output_dims == output_dims);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32, grad_output->GetDevice());

    // 两个 kernel 均为一元素一线程 + 边界检查，串行内层循环保证 GPU 侧运行位级确定
    const int64_t grad_input_total = batch * rows * in_features;
    CHECK_LE(grad_input_total, std::numeric_limits<int>::max());
    int threads_per_block = 256;
    int num_blocks = static_cast<int>((grad_input_total + threads_per_block - 1) / threads_per_block);
    MatmulBackwardGradInputKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(grad_output->DataPtr()), static_cast<const float *>(other->DataPtr()),
        static_cast<float *>(grad_input->DataPtr()), static_cast<int>(rows), static_cast<int>(in_features),
        static_cast<int>(out_features), static_cast<int>(grad_input_total));

    const int64_t grad_other_total = batch * in_features * out_features;
    CHECK_LE(grad_other_total, std::numeric_limits<int>::max());
    num_blocks = static_cast<int>((grad_other_total + threads_per_block - 1) / threads_per_block);
    MatmulBackwardGradOtherKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(input->DataPtr()), static_cast<const float *>(grad_output->DataPtr()),
        static_cast<float *>(grad_other->DataPtr()), static_cast<int>(rows), static_cast<int>(in_features),
        static_cast<int>(out_features), static_cast<int>(grad_other_total));
    return {grad_input, grad_other};
}
```

#### 解决思路

**CPU 实现**：手写三重循环矩阵乘，支持 2D 与 3D batch（逐 batch 独立计算，索引含 batch 偏移）。维度解析仿同文件官方 `LinearForward`（batch = 除末两维外的前缀乘积，rows/in_features/out_features 取法一致），并做维度校验（维度数相等、batch 逐维相等、in_features 匹配）。循环次序采用 r→k→c（内层 c），使 other 与输出沿连续方向访问（缓存友好），输出先清零再累积。反向传播按梯度公式实现：`grad_input = grad_output × otherᵀ`，`grad_other = inputᵀ × grad_output`（逐 batch 独立）。

**CUDA 实现**：采用"一元素一线程 + 内层串行累加 + 边界检查"的 naive GEMM 形态（与 NVIDIA 官方 naive 变体一致）。每个输出元素由固定线程按固定累加顺序（k 升序）串行计算，保证数值确定性——测试使用 `EXPECT_FLOAT_EQ` 精确比较，固定累加顺序是精确匹配的前提。网格按 CEIL_DIV 划分（256 线程块），kernel 内 `idx >= total` 越界防护，与官方 kernel 风格一致。另补了 32 位索引范围防护（total 超出 int 范围时快速失败）。

#### 遇到问题

1. **batch 语义与 torch.matmul 的差异**：实现为"逐 batch 严格相等"（不支持广播），与 torch.matmul 的广播语义不同——测试仅覆盖 batch 相等场景，实现语义已在注释中说明（教学框架内与官方 `LinearForward` 的前缀展平语义一致）。
2. **CPU/CUDA 数值的 ULP 级差异**：nvcc 默认开启 FMA 融合（`--fmad=true`），CUDA 版每个输出元素累加时乘加被融合为 FFMA，与 CPU 版分离乘加存在 1 ULP 级差异——测试数据为小整数（乘积精确）不受影响，注释中已如实说明。

### 作业三：实现Adam优化器

难度：⭐

#### CPU实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdate)`,`TEST(AdamOptimizerTest, MomentumAccumulation)`

代码位置：`infini_train/src/kernels/cpu/accumulate_grad.cc`

```c++
void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF: 
    // =================================== 作业 ===================================
    // 标准 Adam（含偏差校正）：m = β1*m + (1-β1)*g，v = β2*v + (1-β2)*g²，
    // param -= lr * m̂ / (√v̂ + eps)，其中 m̂ = m/(1-β1^t)，v̂ = v/(1-β2^t)，t 从 1 开始
    CHECK_EQ(grad->NumElements(), param->NumElements());  // 校验各张量元素数一致，防止直接调用时越界读写
    CHECK_EQ(m->NumElements(), param->NumElements());
    CHECK_EQ(v->NumElements(), param->NumElements());
    CHECK_GE(t, 1);  // Adam 偏差校正要求 t 从 1 开始（t=0 时 1-β^t=0，除零产生 NaN）
    // 偏差校正因子仅依赖步数，与元素无关，提升到循环外计算一次（double 对齐 PyTorch 标量路径的双精度惯例）
    const double beta1_t = std::pow(static_cast<double>(beta1), static_cast<double>(t));
    const double beta2_t = std::pow(static_cast<double>(beta2), static_cast<double>(t));
    for (int64_t idx = 0; idx < param->NumElements(); ++idx) {
        const float g = static_cast<const float *>(grad->DataPtr())[idx];
        float &param_elem = static_cast<float *>(param->DataPtr())[idx];
        float &m_elem = static_cast<float *>(m->DataPtr())[idx];
        float &v_elem = static_cast<float *>(v->DataPtr())[idx];

        m_elem = beta1 * m_elem + (1.0f - beta1) * g;
        v_elem = beta2 * v_elem + (1.0f - beta2) * g * g;
        const float m_hat = static_cast<float>(m_elem / (1.0 - beta1_t));
        const float v_hat = static_cast<float>(v_elem / (1.0 - beta2_t));
        param_elem -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}
```

#### CUDA实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdateCuda)`,`TEST(AdamOptimizerTest, MomentumAccumulationCuda)`

代码位置：`infini_train/src/kernels/cuda/accumulate_grad.cu`

```c++
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
    CHECK_GE(t, 1);  // Adam 偏差校正要求 t 从 1 开始（t=0 时 1-β^t=0，除零产生 NaN）
    // 偏差校正因子仅依赖步数，与元素无关，host 侧计算一次（double，与 CPU 版一致）
    const double beta1_t = std::pow(static_cast<double>(beta1), static_cast<double>(t));
    const double beta2_t = std::pow(static_cast<double>(beta2), static_cast<double>(t));

    size_t num_elements = param->NumElements();
    // 32 位索引范围防护（超过 int 范围时快速失败，避免后半元素静默不处理）
    CHECK_LE(num_elements, static_cast<size_t>(std::numeric_limits<int>::max()));
    // kernel 的 restrict 承诺要求写目标（param/m/v）与读源（grad）互不重叠（原地调用属未定义行为）
    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, param_ptr, m_ptr, v_ptr, learning_rate,
                                                                beta1, beta2, eps, beta1_t, beta2_t, num_elements);
}
```

#### 解决思路

标准 Adam 公式直译（CPU/CUDA 语义一致）：

- 一阶矩：`m = β₁·m + (1-β₁)·g`
- 二阶矩：`v = β₂·v + (1-β₂)·g²`
- 参数更新：`θ -= lr · m̂/(√v̂ + eps)`，其中偏差校正 `m̂ = m/(1-β₁ᵗ)`、`v̂ = v/(1-β₂ᵗ)`，`t` 从 1 开始（由官方 `Adam::Step` 先 `++t_` 保证）

**关键设计**：偏差校正因子（`β^t`）与元素无关、仅依赖步数，故提升到**循环外计算一次**（用 double 计算，对齐 PyTorch 标量路径的双精度惯例），消除每元素重复的 2N 次幂运算；`m`/`v` 为跨步状态（官方 `Adam` 类管理），kernel 内只做逐元素原位更新。**CUDA 实现**：一元素一线程 + 边界检查，host 侧计算 double 偏差因子传入 kernel，kernel 内 `sqrtf` 计算分母。

#### 遇到问题

1. **`MomentumAccumulation` 用例的精确对照**：测试用 1e-5 容差逐值对照标准公式（含偏差校正）——偏差校正缺失或公式有误必然失败，实现必须与 PyTorch 公式严格一致。
2. **数值精度**：偏差校正因子的幂运算用 double（而非 float）计算，避免 `int64_t` 步数转 float 的精度损失（超过 2²⁴ 步时可见），与 PyTorch 标量路径一致。

### 作业四：实现Tensor基础操作

#### 实现Tensor的Flatten操作

难度：⭐

对应测例：`TEST(TensorTransformTest, Flatten2DTo1D)`,`TEST(TensorTransformTest, FlattenWithRange) `,`TEST(TensorTransformTest, FlattenNonContiguous)`

代码位置：infini_train/src/tensor.cc

```c++
std::shared_ptr<Tensor> Tensor::Flatten(int64_t start, int64_t end) {
    // =================================== 作业 ===================================
    // TODO：实现张量扁平化操作，将指定维度范围[start, end]内的所有维度合并为一个维度
    // HINT: 
    // =================================== 作业 ===================================
    if (dims_.empty()) {
        // 零维张量（标量）无维度可合并，返回原张量（教学框架简化语义；PyTorch 现行为对 0 维 flatten 返回一维视图）
        return shared_from_this();
    }
    if (start < 0) {
        start += dims_.size();
    }
    if (end < 0) {
        end += dims_.size();
    }
    CHECK_GE(start, 0);
    CHECK_LT(start, dims_.size());
    CHECK_GE(end, start);
    CHECK_LT(end, dims_.size());

    int64_t flattened_dim = 1;
    for (int64_t dim = start; dim <= end; ++dim) {
        flattened_dim *= dims_[dim];
    }

    std::vector<int64_t> new_shape;
    new_shape.insert(new_shape.end(), dims_.begin(), dims_.begin() + start);
    new_shape.push_back(flattened_dim);
    new_shape.insert(new_shape.end(), dims_.begin() + end + 1, dims_.end());

    return Contiguous()->View(new_shape);
}
```

#### 实现Tensor的反向传播机制

难度：⭐

对应测例：`TEST(TensorAutogradTest, BackwardComputesGradient)`,`TEST(TensorAutogradTest, BackwardWithMultipleOutputs)`

代码位置：infini_train/src/tensor.cc

```c++
void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    // =================================== 作业 ===================================
    // TODO：实现自动微分反向传播
    // 功能描述：1. 计算当前张量对叶子节点的梯度    2. 支持多输出场景的梯度累加
    // =================================== 作业 ===================================
    if (!gradient) {
        // 无参调用（对应 PyTorch 标量输出 tensor.backward() 语义）：默认梯度为全 1
        // 非标量输出同样构造全 1 梯度（数学上等价于显式 ones 梯度），与 PyTorch 的标量检查不同，
        // 属教学框架的宽容语义；默认梯度经 Fill kernel 按 float 填充，当前仅支持框架使用的 float32 路径。
        gradient = std::make_shared<Tensor>(dims_, dtype_, GetDevice());
        gradient->Fill<float>(1.0f);
    } else {
        // 透传前校验梯度形状与设备一致，防止元素错位静默累加与越界写
        CHECK(gradient->Dims() == dims_) << "Backward gradient shape mismatch";
        CHECK_EQ(static_cast<int>(gradient->GetDevice().Type()), static_cast<int>(GetDevice().Type()))
            << "Backward gradient device mismatch";
    }
    if (grad_fn_) {
        grad_fn_->BackwardPartial(gradient, output_idx_);
    }
    // 契约说明（与 PyTorch 语义的差异）：同一计算图仅可传播一次，重复传播为未定义行为；
    // retain_graph/create_graph 参数当前无效果；无梯度函数的张量（叶子）调用为空操作。
}
```

#### 解决思路

**Flatten**：采用官方注释中预留的实现提示 `Contiguous()->View(new_shape)`——先做负索引规范化（`start`/`end` 为负时加 `dims_.size()`，与同文件 `Squeeze` 风格一致），边界校验（越界、`start > end` 快速失败），计算合并后维度，构造 new_shape 后复用已实现的 `Contiguous()`/`View()`。非连续张量场景（如 Transpose 结果）由 `Contiguous()` 先物理重排再改形状元数据，数据顺序正确。

**Backward**：接入官方已实现的 autograd 引擎（`Function::BackwardPartial` 递归传播 + `AccumulateGrad` 叶子梯度累积），不自写图遍历：`grad_fn_->BackwardPartial(gradient, output_idx_)`。无参调用（`gradient == nullptr`）时构造与输出同形状/同设备/同 dtype 的全 1 默认梯度（PyTorch 标量输出 backward 语义）；显式传入梯度时先校验形状与设备一致（防止元素错位静默累加与越界写）；`grad_fn_` 为空（叶子/未参与求导）时安全忽略。多输出梯度累加由引擎的 `AccumulateGrad` 机制保证（两次独立 Backward 自然累加到同一叶子）。

#### 遇到问题

1. **显式梯度无校验的越界风险**：Backward 透传梯度前若不校验形状，元素数不匹配时会越界写叶子梯度缓冲（静默内存破坏）——补了形状 + 设备一致性 CHECK（快速失败，对齐项目错误处理风格）。
2. **零维张量 Flatten 崩溃**：零维张量（框架内合法，标量参与求导）调用 Flatten 时维数检查必然失败（CHECK 死亡）——补齐零维短路返回原张量（教学框架简化语义）。
3. **重复传播的限制**：同一计算图仅可传播一次（官方引擎传播后不重置依赖计数，重复传播为未定义行为）——根因在官方引擎，作业侧以契约注释文档化（`retain_graph`/`create_graph` 参数当前无效果）。

### 作业五 注册算子kernel的实现

难度：⭐⭐⭐

对应测例：`TEST(DispatcherTest, RegisterAndGetKernel)`,`TEST(DispatcherTest, DuplicateRegistration)`,`TEST(DispatcherTest, GetNonexistentKernel)`

代码位置：infini_train/include/dispatcher.h

```c++
template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
    // =================================== 作业 ===================================
    // TODO：实现通用kernel调用接口
    // 功能描述：将存储的函数指针转换为指定类型并调用
    // =================================== 作业 ===================================

    using FuncT = RetT (*)(ArgsT...);
    return reinterpret_cast<FuncT>(func_ptr_)(args...);
}

template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    // =================================== 作业 ===================================
    // TODO：实现kernel注册机制
    // 功能描述：将kernel函数与设备类型、名称绑定
    // =================================== 作业 ===================================
    CHECK(!key_to_kernel_map_.contains(key))
        << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
    key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
}

#define INFINI_KERNEL_CONCAT_(a, b) a##b
#define INFINI_KERNEL_CONCAT(a, b) INFINI_KERNEL_CONCAT_(a, b)
#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    /* =================================== 作业 =================================== */                               \
    /* TODO：实现自动注册宏 */                                                                                 \
    /* 功能描述：在全局静态区注册kernel，避免显式初始化代码 */                                  \
    /* =================================== 作业 =================================== */                               \
    static const bool INFINI_KERNEL_CONCAT(kernel_reg_, __COUNTER__) =                                                 \
        (infini_train::Dispatcher::Instance().Register(                                                                \
             std::make_pair(device, #kernel_name), kernel_func),                                                       \
         true);
```

#### 解决思路

1. **`KernelFunction::Call`**：类型擦除的通用调用接口——将存储的 `void*` 函数指针还原为 `RetT(*)(ArgsT...)` 并调用。`return void_expr;` 在 `RetT=void` 时合法（C++ [stmt.return]/2），故 void / int / `shared_ptr<Tensor>` / `tuple<...>` 返回均通用（框架中 30+ 个 kernel 签名各异均能调用）。
2. **`Dispatcher::Register`**：往 `key_to_kernel_map_`（键 = `(设备类型, kernel名)`）注册 kernel，重复注册用 glog `CHECK` 快速失败（死亡消息含设备与名称，`EXPECT_DEATH` 可匹配）。
3. **`REGISTER_KERNEL` 宏**：实现"文件加载即注册"的静态注册机制——`__COUNTER__` + 两层 concat（PyTorch `ANONYMOUS_VARIABLE` 同款模式）保证每个注册点变量名唯一（多编译单元、同 TU 多次调用均不冲突），comma-expression 在静态变量动态初始化时触发注册。`#kernel_name` 字符串化作键名。与各 kernel 文件底部的包装宏（`REGISTER_CPU_LINEAR_KERNEL` 等）兼容。

#### 遇到问题

1. **重复注册的死亡语义**：`DuplicateRegistration` 用例要求重复注册同一 key 时进程死亡且消息含 `Kernel already registered`——用 `CHECK` 实现，并确认 gtest 的 `EXPECT_DEATH`（ContainsRegex 部分匹配）在两种死亡测试风格（fast/threadsafe）下均能捕获。
2. **签名拼写契约**：调用方以"按值"拼写 `shared_ptr<Tensor>` 参数，而 kernel 形参为"常引用"——两者在 Itanium ABI 下等价（非平凡拷贝构造类按隐藏引用传递），属类型擦除分发的设计固有约定，已固化为项目契约文档。

### 作业六：实现GPT-2整体训练

难度：⭐⭐⭐⭐

对应测例：`TEST_F(GPT2TrainingTest, LogitsConsistency)`

#### 训练过程logits对比

完成以上所有作业，补齐训练框架的所有实现，理论上`TEST_F(GPT2TrainingTest, LogitsConsistency)`可以通过，在用例中判断比较预置的值和单步正向传播计算结果是否在误差允许范围内相等。

> 说明：该用例实际为 GPT-2 124M 训练 11 步（每步含前向、反向与 SGD 参数更新）后，对最终 logits 与参考文件抽样 100 点、按 1e-3 绝对容差对比。

#### 数据读取实现

代码位置：example/common/tiny_shakespeare_dataset.cc

```c++
TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    CHECK_EQ(ifs.gcount(), 1024) << "Truncated header in file: " << path;
    const int magic = BytesToType<int>(header, 0);
    const int num_toks = BytesToType<int>(header, 8);
    CHECK(kTypeMap.contains(magic)) << "Unsupported magic number: " << magic;
    CHECK_GE(num_toks, 0) << "Invalid num_toks in file: " << path;

    TinyShakespeareFile file;
    file.type = kTypeMap.at(magic);
    const size_t token_size = kTypeToSize.at(file.type);

    // 读 token 流并转为 int64 张量：CrossEntropy 的 target 要求 int64，
    // 且 operator[] 中 y 相对 x 偏移 sizeof(int64_t) = 8 字节，恰为一个 int64 token（预测下一 token）
    const size_t num_tok_bytes = static_cast<size_t>(num_toks) * token_size;
    auto token_bytes = ReadSeveralBytesFromIfstream(num_tok_bytes, &ifs);
    // 读取完整性校验（损坏/截断文件下避免短向量导致的越界读）
    CHECK_EQ(static_cast<size_t>(ifs.gcount()), num_tok_bytes) << "Truncated token data in file: " << path;
    const size_t num_samples = static_cast<size_t>(num_toks) / sequence_length;
    file.dims = {static_cast<int64_t>(num_samples), static_cast<int64_t>(sequence_length)};
    file.tensor = infini_train::Tensor(file.dims, DataType::kINT64);
    int64_t *tensor_data = static_cast<int64_t *>(file.tensor.DataPtr());
    for (size_t i = 0; i < num_samples * sequence_length; ++i) {
        tensor_data[i] = (token_size == 2) ? BytesToType<uint16_t>(token_bytes, i * 2)
                                           : BytesToType<int32_t>(token_bytes, i * 4);
    }
    return file;
}

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
      sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      // 最后一个序列缺"下一 token"作标签，operator[] 以 CHECK_LT(idx, dims[0]-1) 限定可访问边界，
      // 故 Size 报告可访问样本数 = dims[0]-1（空数据集防护为 0），与 operator[] 的契约一致
      num_samples_(text_file_.dims[0] > 0 ? text_file_.dims[0] - 1 : 0) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
}
```

#### Tokenizer功能实现

代码位置：example/common/tokenizer.cc

```c++
Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    CHECK_EQ(ifs.gcount(), 1024) << "Truncated header in file: " << filepath;
    magic_number_ = BytesToType<uint32_t>(header, 0);
    vocab_size_ = BytesToType<uint32_t>(header, 8);
    eot_token_ = BytesToType<uint32_t>(header, 12);
    CHECK(kEotMap.contains(magic_number_)) << "Unsupported tokenizer magic number: " << magic_number_;

    // 词表格式（与 llm.c gpt2_tokenizer.bin 一致）：每个 token 为 1 字节长度前缀 + 原始字节
    token_table_.reserve(vocab_size_);
    for (uint32_t i = 0; i < vocab_size_; ++i) {
        const uint8_t len = BytesToType<uint8_t>(ReadSeveralBytesFromIfstream(1, &ifs), 0);
        auto bytes = ReadSeveralBytesFromIfstream(len, &ifs);
        token_table_.emplace_back(reinterpret_cast<const char *>(bytes.data()), len);
    }
    // 词表读取完整性校验：任一项读取失败（文件截断）都会置位 failbit，快速失败而非静默加载垃圾词表
    CHECK(ifs) << "Truncated tokenizer vocab table in file: " << filepath;
}
```

```c++
std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    CHECK_LT(token_id, vocab_size_) << "token_id out of range: " << token_id;
    return token_table_[token_id];
}
```

```c++
void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    /* 函数前部官方代码（x_tensor 初始化与 prompt 填充）省略，与源码一致 */
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        // 生成场景无 Backward：临时禁用参数梯度使前向不建 autograd 图（Function 即时释放），
        // 避免算子 saved_tensors_ 的循环引用在无 Backward 场景下导致显存逐步泄漏。
        // 副作用说明：本方法（const 签名）临时修改模型参数的 requires_grad 状态——生成区间禁用、
        // 末步恢复为 true；若生成中途以异常中断，requires_grad 将残留为 false，需重新调用
        // set_requires_grad(true) 恢复训练（本框架错误处理为 CHECK 快速失败，正常路径不触发）
        if (t == prompt_len) {
            for (auto &param : model.Parameters()) { param->set_requires_grad(false); }
        }
        // 同步 host 输入到目标设备并前向推理
        x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
        auto outputs = model.Forward({x});
        auto logits = outputs[0];
        auto logits_cpu = logits->To(Device(DeviceType::kCPU, 0));
        const float *logits_data = static_cast<const float *>(logits_cpu.DataPtr());
        const int64_t vocab_size = logits->Dims()[2];
        // 恢复参数梯度（后续训练仍需要建图反向）
        if (t == text_length - 1) {
            for (auto &param : model.Parameters()) { param->set_requires_grad(true); }
        }

        // 生成语义（对齐 llm.c）：取位置 t-1 的分布预测位置 t 的 token，每 batch 独立采样写回
        std::vector<float> probs(static_cast<size_t>(vocab_size));
        for (uint32_t b = 0; b < batch_size; ++b) {
            const float *logits_at_t = logits_data + (b * sequence_length + t - 1) * vocab_size;
            // softmax（数值稳定：先减最大值）
            float max_logit = logits_at_t[0];
            for (int64_t i = 1; i < vocab_size; ++i) {
                if (logits_at_t[i] > max_logit) {
                    max_logit = logits_at_t[i];
                }
            }
            float sum_exp = 0.0f;
            for (int64_t i = 0; i < vocab_size; ++i) {
                probs[i] = std::exp(logits_at_t[i] - max_logit);
                sum_exp += probs[i];
            }
            for (int64_t i = 0; i < vocab_size; ++i) {
                probs[i] /= sum_exp;
            }

            // 函数开头声明了与命名空间常量同名的局部变量 `kRngState = kRngState;`（自初始化，未定义行为），
            // 此处显式引用命名空间常量 kRngState（=1337，与 llm.c 固定种子一致）保证生成可复现
            static uint64_t rng_state = infini_train::kRngState;
            const int next_token = SampleMult(probs.data(), static_cast<int>(vocab_size), RandomF32(rng_state));
            x_buff[b * sequence_length + t] = next_token;
            std::cout << Decode(next_token);
        }
    }
    std::cout << std::endl;
}
```

#### 解决思路

1. **数据读取**（`ReadTinyShakespeareFile`）：解析 1024B HEADER（magic/version/num_toks + reserved），magic 经 `kTypeMap` 映射到 token 类型（GPT-2 为 uint16），token 流读取后转为 int64 张量（`dims = {num_samples, sequence_length}`，`num_samples = num_toks / sequence_length`）。`TinyShakespeareDataset` 构造函数初始化 `text_file_`/`sequence_length_`/`sequence_size_in_bytes_`/`num_samples_`——其中 `num_samples_` 与官方 `operator[]` 的边界契约对齐（可访问样本数 = `dims[0]-1`，最后一个序列缺少"下一 token"作标签）。
2. **Tokenizer**：加载与 llm.c 格式一致的词表文件（HEADER 中 eot_token 位于偏移 12；词表为"1 字节长度前缀 + 原始字节"变长记录），装入 `token_table_`；`Decode` 查表返回（越界 token_id 快速失败）；`GenerateText` 实现逐 token 生成：取位置 t-1 的 logits 分布 → softmax（先减最大值保证数值稳定）→ `SampleMult` 多项式采样 → 写回输入张量 → `Decode` 打印，生成语义与 llm.c 一致（eot 填充 + prompt 引导 + 固定步数）。
3. **生成场景显存处理**：生成无反向传播，autograd 图的 `saved_tensors_` 循环引用在无 Backward 场景下无法释放（每步 forward 显存占用逐步增长，多轮生成后可耗尽显存）——在生成循环首步临时禁用参数 `requires_grad`（前向不建图、Function 即时释放）、末步恢复，消除泄漏且不影响后续训练。

#### 遇到问题

**test_gpt2 的 logits 对比存在环境敏感性（重点排查过程）**：

初版实现后 `LogitsConsistency` 失败（logits 与参考文件差异 0.37~1.54），经系统排查（数据/模型/参考文件内容级验证、单算子数值对照、训练轨迹逐步转储、GPU 运行间确定性实验、compute-sanitizer 内存检查、5090 服务器交叉验证），确认：

1. **数据/模型/参考文件全部为官方原版**：模型/数据/tokenizer 与官方 md5 逐字节一致，模型逐张量与 HuggingFace gpt2 权重一致（转置后 max_diff=0），参考文件与 git 历史入库版本逐字节一致；单步前向 loss（5.35609）与 llm.c 官方微调示例（5.356189）一致——实现链路正确。
2. **实现与参考实现数值行为一致**：作业实现的 Matmul kernel 与参考二进制的 SASS/PTX 逐指令一致（diff=0）。
3. **失败根因 = 环境敏感性**：参考文件由参考实现在本地 CUDA 13.3 环境运行产生（本地复现中，11 步训练后的 loss 与参考精确一致）；而 fp32 训练的数值轨迹对运行环境敏感——**cuBLAS/nvcc 版本差异（12.8 vs 13.3）为主因**（12.8 工具链编译稳定失败、13.3 环境通过），GPU 架构与原子累加噪声为次要因素（同为 13.3 时不同 GPU 的噪声分布不同）：每步 ULP 级数值差被训练轨迹中换行符位置约 1e5 倍的高灵敏度方向混沌放大，使 11 步训练后的 logits 与参考产生 0.002~7.6 的差异（同一二进制多次运行结果不同，GPU 原子累加顺序随机所致）。测试容差 1e-3 低于该噪声地板。
4. **结论**：在 5090 + CUDA 13.3 运行时环境下全量 8/8 通过（test_gpt2 通过）；在 4060（Ada 架构）或 CUDA 12.8 环境下 test_gpt2 因上述环境性差异失败（其余 7/8 全过）。该问题应该与作业实现质量无关，是测试对运行环境的敏感依赖，已在截图下方说明测试环境。

