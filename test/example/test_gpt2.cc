#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>

#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/optimizer.h"

#include "example/common/tiny_shakespeare_dataset.h"
#include "example/common/tokenizer.h"
#include "example/gpt2/net.h"

namespace infini_train {

constexpr char kDeviceCPU[] = "cpu";
constexpr char kDeviceCUDA[] = "cuda";

class LogitsValidator {
public:
    /**
    * @brief 验证当前张量数据与参考文件中的logits是否匹配
    * 
    * @param logits 待验证的张量（设备不限，会自动转换为CPU）
    * @param filename 参考文件的二进制路径
    * @param tolerance 允许的数值绝对误差阈值（默认1e-4）
    * @return bool 验证结果（true表示所有采样点误差在允许范围内）
    * 
    * @note 二进制文件格式：
    * 1. 维度数量 (size_t, 8字节)
    * 2. 各维度值 (int64_t[num_dims], 8*num_dims字节)
    * 3. 张量数据 (float[num_elements], 4*num_elements字节) 
    * 
    * @warning 需确保：
    * - 输入张量内存有效
    * - 参考文件存在且格式正确
    * - 跨平台使用时注意字节序问题
    */
    static bool Validate(Tensor& logits, const std::string& filename, float tolerance = 1e-3) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile.is_open()) {
            LOG(ERROR) << "Failed to open reference file: " << filename;
            return false;
        }
        
        size_t num_dims;
        infile.read(reinterpret_cast<char*>(&num_dims), sizeof(size_t));
        std::vector<int64_t> ref_dims(num_dims);
        for (size_t i = 0; i < num_dims; i++) {
            infile.read(reinterpret_cast<char*>(&ref_dims[i]), sizeof(int64_t));
        }
        
        auto current_dims = logits.Dims();
        if (ref_dims != current_dims) {
            return false;
        }
        
        size_t num_elements = 1;
        for (auto dim : ref_dims) {
            num_elements *= dim;
        }
        std::vector<float> ref_data(num_elements);
        infile.read(reinterpret_cast<char*>(ref_data.data()), num_elements * sizeof(float));
        infile.close();
        
        auto cpu_logits = logits.To(Device(DeviceType::kCPU, 0));
        const float* current_data = static_cast<const float*>(cpu_logits.DataPtr());
        
        // 抽样比较策略
        const int sample_count = 100; // 抽取100个点进行比较
        std::vector<size_t> indices_to_check;

        for (int i = 0; i < sample_count; i++) {
            indices_to_check.push_back(i * num_elements / sample_count);
        }
        
        for (auto idx : indices_to_check) {
            float ref_val = ref_data[idx];
            float current_val = current_data[idx];
            float diff = std::abs(ref_val - current_val);
            
            if (diff > tolerance) {
                LOG(INFO) << "Logits mismatch at position " << idx 
                          << ": Reference=" << ref_val 
                          << ", Current=" << current_val
                          << ", Diff=" << diff;
                return false;
            }
        }
        
        LOG(INFO) << "Logits validation passed with " << sample_count << " samples";
        return true;
    }
};

// 测试类

namespace {
constexpr size_t kLogitProbeIndex = 385973;
constexpr float kDiagnosticTolerance = 1e-3f;

struct TensorSummary {
    std::string dims;
    size_t num_elements = 0;
    size_t sample_index = 0;
    float sample = 0.0f;
    float min = 0.0f;
    float max = 0.0f;
    double mean = 0.0;
    double l2 = 0.0;
    bool finite = true;
};

struct StepSnapshot {
    int step = 0;
    float loss = 0.0f;
    float logit = 0.0f;
    float reference_diff = 0.0f;
};

enum class DiagnosticMode {
    kForwardOnly,
    kForwardBackward,
    kFull,
};

const char *DiagnosticModeName(DiagnosticMode mode) {
    switch (mode) {
    case DiagnosticMode::kForwardOnly:
        return "forward_only";
    case DiagnosticMode::kForwardBackward:
        return "forward_backward";
    case DiagnosticMode::kFull:
        return "full";
    }
    return "unknown";
}

std::string DimsToString(const std::vector<int64_t> &dims) {
    std::string result = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i != 0) {
            result += "x";
        }
        result += std::to_string(dims[i]);
    }
    result += "]";
    return result;
}

TensorSummary SummarizeFloatTensor(Tensor &tensor, size_t preferred_sample_index) {
    CHECK_EQ(static_cast<int>(tensor.Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_GT(tensor.NumElements(), 0);

    auto cpu_tensor = tensor.To(Device(DeviceType::kCPU, 0));
    const float *data = static_cast<const float *>(cpu_tensor.DataPtr());
    const size_t num_elements = cpu_tensor.NumElements();

    TensorSummary summary;
    summary.dims = DimsToString(cpu_tensor.Dims());
    summary.num_elements = num_elements;
    summary.sample_index = std::min(preferred_sample_index, num_elements - 1);
    summary.sample = data[summary.sample_index];
    summary.min = std::numeric_limits<float>::infinity();
    summary.max = -std::numeric_limits<float>::infinity();

    double sum = 0.0;
    double square_sum = 0.0;
    for (size_t i = 0; i < num_elements; ++i) {
        const float value = data[i];
        summary.finite = summary.finite && std::isfinite(value);
        summary.min = std::min(summary.min, value);
        summary.max = std::max(summary.max, value);
        sum += value;
        square_sum += static_cast<double>(value) * value;
    }
    summary.mean = sum / static_cast<double>(num_elements);
    summary.l2 = std::sqrt(square_sum);
    return summary;
}

void LogTensorSummary(int run_id, DiagnosticMode mode, const char *stage, int step, int micro_step,
                      const std::string &tensor_name, const TensorSummary &summary) {
    LOG(INFO) << "DIAG run=" << run_id << " mode=" << DiagnosticModeName(mode) << " stage=" << stage
              << " step=" << step << " micro=" << micro_step << " tensor=" << tensor_name
              << " dims=" << summary.dims << " elements=" << summary.num_elements
              << " sample_index=" << summary.sample_index << " sample=" << summary.sample
              << " min=" << summary.min << " max=" << summary.max << " mean=" << summary.mean
              << " l2=" << summary.l2 << " finite=" << summary.finite;
}
} // namespace

class GPT2TrainingTest : public ::testing::Test {
protected:
    void SetUp() override {
        llmc_filepath = "../../Data/gpt2_124M.bin";
        input_bin = "../../Data/tinyshakespeare/tiny_shakespeare_train.bin";
        tokenizer_bin = "../../Data/gpt2_tokenizer.bin";
        logits_reference = "../../Data/gpt2_logits_reference.bin";

        device_flag = "cuda";
        model_name = "gpt2";
        batch_size = 2;
        sequence_length = 64;
        total_batch_size = 256;
        num_iteration = 10;    // 迭代次数
        text_length = 64;    // 生成文本长度
        learning_rate = 1e-4;    //学习率

        Initialize();

        LOG(INFO)<< "Initialize() finished!";
    }
    
    void Initialize() {
        if (device_flag == kDeviceCPU) {
            device = Device(DeviceType::kCPU, 0);
        } else {
            device = Device(DeviceType::kCUDA, 0);
        }

        model = GPT2::FromLLMC(llmc_filepath);
        model->To(device);

        train_loader = std::make_unique<DataLoader>(
            std::make_shared<TinyShakespeareDataset>(input_bin, sequence_length), batch_size);

        optimizer = std::make_unique<optimizers::SGD>(model->Parameters(), learning_rate);

        loss_fn = std::make_unique<nn::CrossEntropyLoss>();
        loss_fn->To(device);

        if (!tokenizer_bin.empty()) {
            tokenizer = std::make_unique<Tokenizer>(tokenizer_bin);
        }
    }
    
    void RunSingleStep() {
        auto train_iter = train_loader->begin();
        
        optimizer->ZeroGrad();
        float lossf = 0.0f;
        for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
            auto [x, y] = *train_iter;
            ++train_iter;
            x = std::make_shared<Tensor>(x->To(device));
            y = std::make_shared<Tensor>(y->To(device));

            auto outputs = model->Forward({x, y});

            logits = outputs[0];

            ASSERT_NE(logits, nullptr) << "First output is null";
            ASSERT_GT(logits->NumElements(), 0) << "Empty logits tensor";
            ASSERT_EQ(logits->Dims().size(), 3) << "Logits should be 3D (batch, seq, vocab)";
            ASSERT_NE(loss_fn, nullptr) << "Loss function not initialized!";

            auto loss = loss_fn->Forward({logits, y})[0];
            auto loss_cpu = loss->To(Device());
            lossf += static_cast<const float*>(loss_cpu.DataPtr())[0] / grad_accum_steps;

            loss->Backward();
        }
        optimizer->Step();
    }

    float ReadReferenceLogit(size_t index) const {
        std::ifstream infile(logits_reference, std::ios::binary);
        CHECK(infile.is_open()) << "Failed to open reference file: " << logits_reference;

        size_t num_dims = 0;
        infile.read(reinterpret_cast<char *>(&num_dims), sizeof(size_t));
        std::vector<int64_t> ref_dims(num_dims);
        for (size_t i = 0; i < num_dims; ++i) {
            infile.read(reinterpret_cast<char *>(&ref_dims[i]), sizeof(int64_t));
        }

        size_t num_elements = 1;
        for (const auto dim : ref_dims) {
            num_elements *= static_cast<size_t>(dim);
        }
        CHECK_LT(index, num_elements);

        const auto data_offset = static_cast<std::streamoff>(sizeof(size_t) + num_dims * sizeof(int64_t)
                                                             + index * sizeof(float));
        infile.seekg(data_offset, std::ios::beg);
        float value = 0.0f;
        infile.read(reinterpret_cast<char *>(&value), sizeof(float));
        CHECK(infile.good()) << "Failed to read reference logit at index " << index;

        LOG(INFO) << "DIAG reference_file=" << logits_reference << " dims=" << DimsToString(ref_dims)
                  << " elements=" << num_elements << " logit_index=" << index << " reference=" << value;
        return value;
    }

    void LogSelectedParameters(int run_id, DiagnosticMode mode, const char *stage, int step) {
        const auto state_dict = model->StateDict();
        const std::array<const char *, 9> tensor_names = {
            "transformer.wte.weight",
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.ln_1.bias",
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.0.attn.c_proj.weight",
            "transformer.h.0.mlp.c_fc.weight",
            "transformer.ln_f.weight",
            "transformer.ln_f.bias",
            "lm_head.weight",
        };

        const auto wte = state_dict.find("transformer.wte.weight");
        const auto lm_head = state_dict.find("lm_head.weight");
        if (wte != state_dict.end() && lm_head != state_dict.end()) {
            LOG(INFO) << "DIAG run=" << run_id << " mode=" << DiagnosticModeName(mode) << " stage=" << stage
                      << " step=" << step << " tensor=weight_tying shared_ptr=" << (wte->second.get() == lm_head->second.get());
        }

        for (const auto *name : tensor_names) {
            const auto it = state_dict.find(name);
            if (it == state_dict.end()) {
                LOG(WARNING) << "DIAG missing parameter tensor=" << name;
                continue;
            }
            const auto summary = SummarizeFloatTensor(*it->second, 0);
            LogTensorSummary(run_id, mode, stage, step, -1, name, summary);
            EXPECT_TRUE(summary.finite) << "Non-finite parameter values in " << name;
        }
    }

    void LogSelectedGradients(int run_id, DiagnosticMode mode, int step) {
        const auto state_dict = model->StateDict();
        const std::array<const char *, 9> tensor_names = {
            "transformer.wte.weight",
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.ln_1.bias",
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.0.attn.c_proj.weight",
            "transformer.h.0.mlp.c_fc.weight",
            "transformer.ln_f.weight",
            "transformer.ln_f.bias",
            "lm_head.weight",
        };

        for (const auto *name : tensor_names) {
            const auto it = state_dict.find(name);
            if (it == state_dict.end()) {
                LOG(WARNING) << "DIAG missing gradient owner tensor=" << name;
                continue;
            }
            const auto grad = it->second->grad();
            if (!grad) {
                LOG(WARNING) << "DIAG missing gradient tensor=" << name << ".grad";
                continue;
            }
            const auto summary = SummarizeFloatTensor(*grad, 0);
            LogTensorSummary(run_id, mode, "backward", step, -1, std::string(name) + ".grad", summary);
            EXPECT_TRUE(summary.finite) << "Non-finite gradient values in " << name;
        }
    }

    StepSnapshot RunInstrumentedStep(int run_id, DiagnosticMode mode, int step, float reference_logit,
                                     bool emit_logs, const StepSnapshot *previous) {
        auto train_iter = train_loader->begin();

        optimizer->ZeroGrad();
        float lossf = 0.0f;
        StepSnapshot snapshot;
        snapshot.step = step;

        for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
            auto [x, y] = *train_iter;
            ++train_iter;
            x = std::make_shared<Tensor>(x->To(device));
            y = std::make_shared<Tensor>(y->To(device));

            auto outputs = model->Forward({x, y});
            logits = outputs[0];

            CHECK(logits != nullptr) << "First output is null";
            CHECK_GT(logits->NumElements(), 0) << "Empty logits tensor";
            CHECK_EQ(logits->Dims().size(), 3) << "Logits should be 3D (batch, seq, vocab)";
            CHECK(loss_fn != nullptr) << "Loss function not initialized!";

            auto loss = loss_fn->Forward({logits, y})[0];
            auto loss_cpu = loss->To(Device());
            lossf += static_cast<const float *>(loss_cpu.DataPtr())[0] / grad_accum_steps;

            if (micro_step == grad_accum_steps - 1) {
                const auto logits_summary = SummarizeFloatTensor(*logits, kLogitProbeIndex);
                snapshot.loss = lossf;
                snapshot.logit = logits_summary.sample;
                snapshot.reference_diff = std::abs(snapshot.logit - reference_logit);
                EXPECT_TRUE(std::isfinite(snapshot.loss)) << "Loss is not finite at step " << step;
                EXPECT_TRUE(logits_summary.finite) << "Logits contain non-finite values at step " << step;

                if (emit_logs) {
                    LogTensorSummary(run_id, mode, "forward", step, micro_step, "logits", logits_summary);
                    const float delta_loss = previous ? snapshot.loss - previous->loss : 0.0f;
                    const float delta_logit = previous ? snapshot.logit - previous->logit : 0.0f;
                    LOG(INFO) << "DIAG_STEP run=" << run_id << " mode=" << DiagnosticModeName(mode)
                              << " stage=forward step=" << step << " loss=" << snapshot.loss
                              << " logit_index=" << kLogitProbeIndex << " logit=" << snapshot.logit
                              << " reference=" << reference_logit << " reference_diff=" << snapshot.reference_diff
                              << " has_previous=" << (previous != nullptr) << " delta_loss=" << delta_loss
                              << " delta_logit=" << delta_logit;
                }
            }

            if (mode != DiagnosticMode::kForwardOnly) {
                loss->Backward();
            }
        }

        if (mode != DiagnosticMode::kForwardOnly && emit_logs && step == 0) {
            LogSelectedGradients(run_id, mode, step);
        }

        if (mode == DiagnosticMode::kFull) {
            optimizer->Step();
            if (emit_logs && step == 0) {
                LogSelectedParameters(run_id, mode, "optimizer", step);
            }
        }

        return snapshot;
    }

    std::vector<StepSnapshot> RunTrainingDiagnosticsSequence(int run_id, DiagnosticMode mode, int last_step,
                                                             bool emit_logs) {
        const float reference_logit = ReadReferenceLogit(kLogitProbeIndex);
        std::vector<StepSnapshot> snapshots;
        snapshots.reserve(static_cast<size_t>(last_step + 1));

        if (emit_logs) {
            LOG(INFO) << "DIAG_RUN run=" << run_id << " mode=" << DiagnosticModeName(mode)
                      << " batch_size=" << batch_size << " sequence_length=" << sequence_length
                      << " total_batch_size=" << total_batch_size << " grad_accum_steps=" << grad_accum_steps
                      << " learning_rate=" << learning_rate << " num_iteration=" << num_iteration;
            LogSelectedParameters(run_id, mode, "load", -1);
        }

        std::optional<StepSnapshot> previous;
        for (int step = 0; step <= last_step; ++step) {
            snapshots.push_back(RunInstrumentedStep(run_id, mode, step, reference_logit, emit_logs,
                                                    previous ? &previous.value() : nullptr));
            previous = snapshots.back();
        }

        if (emit_logs) {
            int first_exceed_step = -1;
            for (const auto &snapshot : snapshots) {
                if (snapshot.reference_diff > kDiagnosticTolerance) {
                    first_exceed_step = snapshot.step;
                    break;
                }
            }
            LOG(INFO) << "DIAG_FIRST_EXCEED run=" << run_id << " mode=" << DiagnosticModeName(mode)
                      << " tolerance=" << kDiagnosticTolerance << " step=" << first_exceed_step;
        }

        return snapshots;
    }

    void CompareDiagnosticRuns(const std::vector<StepSnapshot> &lhs, const std::vector<StepSnapshot> &rhs) {
        ASSERT_EQ(lhs.size(), rhs.size());
        int first_different_step = -1;
        const char *first_different_stage = "none";

        for (size_t i = 0; i < lhs.size(); ++i) {
            const float loss_diff = std::abs(lhs[i].loss - rhs[i].loss);
            const float logit_diff = std::abs(lhs[i].logit - rhs[i].logit);
            const float reference_diff_delta = std::abs(lhs[i].reference_diff - rhs[i].reference_diff);
            LOG(INFO) << "DIAG_COMPARE step=" << lhs[i].step << " loss_a=" << lhs[i].loss
                      << " loss_b=" << rhs[i].loss << " loss_diff=" << loss_diff
                      << " logit_a=" << lhs[i].logit << " logit_b=" << rhs[i].logit
                      << " logit_diff=" << logit_diff << " reference_diff_delta=" << reference_diff_delta;

            if (first_different_step < 0 && (loss_diff != 0.0f || logit_diff != 0.0f)) {
                first_different_step = lhs[i].step;
                first_different_stage = loss_diff != 0.0f ? "loss" : "logit";
            }
        }

        LOG(INFO) << "DIAG_DETERMINISM same_process_equal=" << (first_different_step < 0)
                  << " first_different_step=" << first_different_step
                  << " first_different_stage=" << first_different_stage;
    }
    
    std::unique_ptr<GPT2> model;
    std::unique_ptr<DataLoader> train_loader;
    std::unique_ptr<optimizers::SGD> optimizer;
    std::unique_ptr<nn::CrossEntropyLoss> loss_fn;
    std::unique_ptr<Tokenizer> tokenizer;
    Device device;
    std::shared_ptr<Tensor> logits;
    int grad_accum_steps = 0;

    std::string llmc_filepath;
    std::string input_bin;
    std::string tokenizer_bin;
    std::string logits_reference;
    std::string device_flag;
    std::string model_name;
    int batch_size = 2;
    int sequence_length = 64;    
    int total_batch_size = 256;
    int num_iteration = 10;    // 迭代次数
    int text_length = 64;    // 生成文本长度
    int freq_generate_txt = 10;
    float learning_rate = 1e-4;
};

TEST_F(GPT2TrainingTest, SingleStepDiagnostics) {
    auto train_iter = train_loader->begin();

    optimizer->ZeroGrad();

    auto [x, y] = *train_iter;
    x = std::make_shared<Tensor>(x->To(device));
    y = std::make_shared<Tensor>(y->To(device));

    auto outputs = model->Forward({x, y});
    logits = outputs[0];

    ASSERT_NE(logits, nullptr) << "First output is null";
    ASSERT_GT(logits->NumElements(), 0) << "Empty logits tensor";
    ASSERT_EQ(logits->Dims().size(), 3) << "Logits should be 3D (batch, seq, vocab)";
    ASSERT_NE(loss_fn, nullptr) << "Loss function not initialized!";

    auto loss = loss_fn->Forward({logits, y})[0];
    ASSERT_NE(loss, nullptr) << "Loss output is null";

    auto loss_cpu = loss->To(Device());
    const float loss_value = static_cast<const float *>(loss_cpu.DataPtr())[0];

    loss->Backward();
    optimizer->Step();

    EXPECT_TRUE(std::isfinite(loss_value)) << "Loss is not finite: " << loss_value;

    auto logits_cpu = logits->To(Device(DeviceType::kCPU, 0));
    const float *logits_data = static_cast<const float *>(logits_cpu.DataPtr());
    const size_t num_elements = logits->NumElements();
    const size_t checked_samples = std::min(static_cast<size_t>(16), num_elements);

    for (size_t i = 0; i < checked_samples; ++i) {
        const size_t idx = i * num_elements / checked_samples;
        EXPECT_TRUE(std::isfinite(logits_data[idx])) << "Logit sample is not finite at position " << idx
                                                     << ": " << logits_data[idx];
    }
}


TEST_F(GPT2TrainingTest, TrainingStageDiagnostics) {
    const auto tokens_per_fwdbwd = batch_size * sequence_length;
    grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
    ASSERT_GT(grad_accum_steps, 0);

    Initialize();
    (void)RunTrainingDiagnosticsSequence(0, DiagnosticMode::kForwardOnly, 0, true);

    Initialize();
    (void)RunTrainingDiagnosticsSequence(0, DiagnosticMode::kForwardBackward, 0, true);

    Initialize();
    auto first_run = RunTrainingDiagnosticsSequence(1, DiagnosticMode::kFull, num_iteration, true);

    Initialize();
    auto second_run = RunTrainingDiagnosticsSequence(2, DiagnosticMode::kFull, num_iteration, false);
    CompareDiagnosticRuns(first_run, second_run);
}

TEST_F(GPT2TrainingTest, LogitsConsistency) {
    const auto tokens_per_fwdbwd = batch_size * sequence_length;    // 梯度累积步数
    grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
    
    for (int step = 0; step < num_iteration + 1; ++step) {
        LOG(INFO)<<"epoch: " << step;
        // 执行训练
        RunSingleStep();

        /* tokenizer */
        if ((step + 1) % freq_generate_txt == 0) {
            if (!tokenizer) {
                continue;
            }
            tokenizer->GenerateText(*model, batch_size, sequence_length, text_length, device);
        }
    }

    // 验证 logits
    if (!logits_reference.empty()) {
        bool validation_passed = LogitsValidator::Validate(*logits, logits_reference);
        EXPECT_TRUE(validation_passed) << "Logits validation failed!";
    } else {
        FAIL() << "No reference logits provided! Cannot validate.";        
    }
}
} // namespace infini_train
