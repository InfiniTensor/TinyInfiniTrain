#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/optimizer.h"

#include "example/common/tiny_shakespeare_dataset.h"
#include "example/common/tokenizer.h"
#include "example/gpt2/net.h"

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

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

struct WeightTyingSnapshot {
    std::shared_ptr<Tensor> wte;
    std::shared_ptr<Tensor> lm_head;
    std::shared_ptr<Tensor> wte_grad;
    std::shared_ptr<Tensor> lm_head_grad;
};

class InspectableSGD final : public optimizers::SGD {
public:
    using optimizers::SGD::SGD;

    const std::vector<std::shared_ptr<Tensor>> &params() const { return params_; }
};

WeightTyingSnapshot GetWeightTyingSnapshot(GPT2 &model) {
    const auto state_dict = model.StateDict();
    const auto wte_it = state_dict.find("transformer.wte.weight");
    const auto lm_head_it = state_dict.find("lm_head.weight");
    CHECK(wte_it != state_dict.end());
    CHECK(lm_head_it != state_dict.end());
    return {
        .wte = wte_it->second,
        .lm_head = lm_head_it->second,
        .wte_grad = wte_it->second->grad(),
        .lm_head_grad = lm_head_it->second->grad(),
    };
}

size_t CountPointerOccurrences(const std::vector<std::shared_ptr<Tensor>> &params, const std::shared_ptr<Tensor> &needle) {
    return std::count_if(params.begin(), params.end(), [&](const auto &param) { return param.get() == needle.get(); });
}

size_t CountDataPtrOccurrences(const std::vector<std::shared_ptr<Tensor>> &params, const void *data_ptr) {
    return std::count_if(params.begin(), params.end(), [&](const auto &param) { return param->DataPtr() == data_ptr; });
}

size_t CountUniqueTensorObjects(const std::vector<std::shared_ptr<Tensor>> &params) {
    std::unordered_set<const Tensor *> objects;
    for (const auto &param : params) {
        objects.insert(param.get());
    }
    return objects.size();
}

size_t CountUniqueDataPtrs(const std::vector<std::shared_ptr<Tensor>> &params) {
    std::unordered_set<const void *> data_ptrs;
    for (const auto &param : params) {
        data_ptrs.insert(param->DataPtr());
    }
    return data_ptrs.size();
}

void LogWeightTyingSnapshot(const char *stage, const WeightTyingSnapshot &snapshot) {
    LOG(INFO) << "WEIGHT_TYING stage=" << stage
              << " wte_tensor=" << snapshot.wte.get()
              << " lm_head_tensor=" << snapshot.lm_head.get()
              << " same_tensor=" << (snapshot.wte.get() == snapshot.lm_head.get())
              << " wte_data=" << snapshot.wte->DataPtr()
              << " lm_head_data=" << snapshot.lm_head->DataPtr()
              << " same_data=" << (snapshot.wte->DataPtr() == snapshot.lm_head->DataPtr())
              << " storage_data_ptr_proxy=" << snapshot.wte->DataPtr() << "/" << snapshot.lm_head->DataPtr()
              << " wte_grad=" << snapshot.wte_grad.get()
              << " lm_head_grad=" << snapshot.lm_head_grad.get()
              << " same_grad=" << (snapshot.wte_grad.get() == snapshot.lm_head_grad.get())
              << " wte_grad_data=" << (snapshot.wte_grad ? snapshot.wte_grad->DataPtr() : nullptr)
              << " lm_head_grad_data=" << (snapshot.lm_head_grad ? snapshot.lm_head_grad->DataPtr() : nullptr)
              << " same_grad_data=" << (snapshot.wte_grad && snapshot.lm_head_grad
                                              && snapshot.wte_grad->DataPtr() == snapshot.lm_head_grad->DataPtr());
}

void LogOptimizerAliases(const char *stage, const std::vector<std::shared_ptr<Tensor>> &params,
                         const WeightTyingSnapshot &snapshot) {
    LOG(INFO) << "WEIGHT_TYING_OPT stage=" << stage
              << " total_params=" << params.size()
              << " unique_tensor_objects=" << CountUniqueTensorObjects(params)
              << " unique_data_ptrs=" << CountUniqueDataPtrs(params)
              << " wte_object_occurrences=" << CountPointerOccurrences(params, snapshot.wte)
              << " lm_head_object_occurrences=" << CountPointerOccurrences(params, snapshot.lm_head)
              << " shared_data_occurrences=" << CountDataPtrOccurrences(params, snapshot.wte->DataPtr());
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i].get() == snapshot.wte.get() || params[i].get() == snapshot.lm_head.get()
            || params[i]->DataPtr() == snapshot.wte->DataPtr()) {
            LOG(INFO) << "WEIGHT_TYING_OPT_PARAM stage=" << stage << " index=" << i
                      << " tensor=" << params[i].get() << " data=" << params[i]->DataPtr()
                      << " grad=" << params[i]->grad().get()
                      << " grad_data=" << (params[i]->grad() ? params[i]->grad()->DataPtr() : nullptr);
        }
    }
}


struct LogitsBinary {
    std::vector<int64_t> dims;
    std::vector<float> data;
};

struct RowDiffMetrics {
    size_t row_index = 0;
    size_t batch = 0;
    size_t sequence = 0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rmse = 0.0;
    size_t count_gt_tolerance = 0;
};

struct LogitsComparisonMetrics {
    size_t elements = 0;
    bool has_mismatch = false;
    size_t first_mismatch = 0;
    size_t first_mismatch_batch = 0;
    size_t first_mismatch_sequence = 0;
    size_t first_mismatch_vocab = 0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rmse = 0.0;
    size_t count_gt_tolerance = 0;
    size_t rows_with_gt_tolerance = 0;
    size_t rows_with_full_vocab_gt_tolerance = 0;
    size_t reference_nan_count = 0;
    size_t candidate_nan_count = 0;
    size_t reference_inf_count = 0;
    size_t candidate_inf_count = 0;
    double cosine = 0.0;
    bool cosine_defined = false;
    float reference0 = 0.0f;
    float candidate0 = 0.0f;
    float reference_probe = 0.0f;
    float candidate_probe = 0.0f;
    std::vector<RowDiffMetrics> row_metrics;
};

struct GitMetadata {
    std::string branch = "unknown";
    std::string commit = "unknown";
    bool working_tree_dirty = true;
    std::string unstaged_diff_sha256 = "unknown";
    std::string staged_diff_sha256 = "unknown";
    std::string status_porcelain_sha256 = "unknown";
};

struct ReferenceTraceMetadata {
    GitMetadata git;
    std::string model_checkpoint_path;
    std::string model_checkpoint_sha256;
    std::string training_dataset_path;
    std::string training_dataset_sha256;
    std::string tokenizer_path;
    std::string tokenizer_sha256;
    std::string training_microbatches_sha256;
    int training_microbatch_count = 0;
    int training_microbatches_repeated_use_count = 0;
    std::string final_forward_input_sha256;
    std::string final_forward_target_sha256;
};

struct ReferenceHeader {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t dtype = 0;
    uint32_t ndim = 0;
    uint64_t num_elements = 0;
};

struct ReferenceWriteResult {
    std::string path;
    std::string sha256;
};

static_assert(sizeof(ReferenceHeader) == 24);

constexpr uint32_t kReferenceMagic = 0x46523247u; // "G2RF" in little-endian byte order.
constexpr uint32_t kReferenceVersion = 1;
constexpr uint32_t kReferenceDtypeFloat32 = 1;

[[noreturn]] void ThrowRuntimeError(const std::string &message) {
    throw std::runtime_error(message);
}

void RequireOrThrow(bool condition, const std::string &message) {
    if (!condition) {
        ThrowRuntimeError(message);
    }
}

bool EndsWith(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

std::string TrimWhitespace(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) {
        ++first;
    }
    return value.substr(first);
}

std::string ReadTextFileFirstLine(const std::filesystem::path &path) {
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    return line;
}

std::filesystem::path FindRepoRoot() {
    auto path = std::filesystem::current_path();
    while (!path.empty()) {
        if (std::filesystem::exists(path / ".git")) {
            return path;
        }
        path = path.parent_path();
    }
    return std::filesystem::current_path();
}

std::filesystem::path ResolveGitDir(const std::filesystem::path &repo_root) {
    const auto git_path = repo_root / ".git";
    if (std::filesystem::is_directory(git_path)) {
        return git_path;
    }
    if (!std::filesystem::is_regular_file(git_path)) {
        return {};
    }

    const auto line = ReadTextFileFirstLine(git_path);
    constexpr std::string_view prefix = "gitdir: ";
    if (line.rfind(prefix, 0) != 0) {
        return {};
    }
    std::filesystem::path git_dir(line.substr(prefix.size()));
    if (git_dir.is_relative()) {
        git_dir = repo_root / git_dir;
    }
    return git_dir.lexically_normal();
}

std::string ShellQuote(const std::string &value) {
    std::string quoted = "'";
    for (const char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

std::string RunCommandCapture(const std::string &command) {
    FILE *pipe = popen(command.c_str(), "r");
    RequireOrThrow(pipe != nullptr, "Failed to run command: " + command);
    std::array<char, 4096> buffer{};
    std::string output;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    const int status = pclose(pipe);
    RequireOrThrow(status == 0, "Command failed with status " + std::to_string(status) + ": " + command);
    return output;
}

std::string RunGitCommand(const std::filesystem::path &repo_root, const std::string &args) {
    return RunCommandCapture("git -C " + ShellQuote(repo_root.string()) + " " + args);
}

void AppendBytes(std::vector<uint8_t> &bytes, const void *data, size_t size) {
    if (size == 0) {
        return;
    }
    const auto *begin = static_cast<const uint8_t *>(data);
    bytes.insert(bytes.end(), begin, begin + size);
}

template <typename T> void AppendPod(std::vector<uint8_t> &bytes, const T &value) {
    AppendBytes(bytes, &value, sizeof(T));
}

void AppendStringRecord(std::vector<uint8_t> &bytes, std::string_view value) {
    const uint64_t size = static_cast<uint64_t>(value.size());
    AppendPod(bytes, size);
    AppendBytes(bytes, value.data(), value.size());
}

uint32_t Sha256RotateRight(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32 - bits));
}

std::string Sha256Hex(const uint8_t *data, size_t size) {
    static constexpr std::array<uint32_t, 64> kRoundConstants = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u,
    };

    std::array<uint32_t, 8> hash = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    std::vector<uint8_t> message;
    if (size > 0) {
        message.assign(data, data + size);
    }
    const uint64_t bit_size = static_cast<uint64_t>(size) * 8u;
    message.push_back(0x80u);
    while ((message.size() % 64) != 56) {
        message.push_back(0u);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        message.push_back(static_cast<uint8_t>((bit_size >> shift) & 0xffu));
    }

    for (size_t offset = 0; offset < message.size(); offset += 64) {
        std::array<uint32_t, 64> words{};
        for (size_t i = 0; i < 16; ++i) {
            const size_t j = offset + i * 4;
            words[i] = (static_cast<uint32_t>(message[j]) << 24) | (static_cast<uint32_t>(message[j + 1]) << 16)
                       | (static_cast<uint32_t>(message[j + 2]) << 8) | static_cast<uint32_t>(message[j + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = Sha256RotateRight(words[i - 15], 7) ^ Sha256RotateRight(words[i - 15], 18)
                                ^ (words[i - 15] >> 3);
            const uint32_t s1 = Sha256RotateRight(words[i - 2], 17) ^ Sha256RotateRight(words[i - 2], 19)
                                ^ (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }

        uint32_t a = hash[0];
        uint32_t b = hash[1];
        uint32_t c = hash[2];
        uint32_t d = hash[3];
        uint32_t e = hash[4];
        uint32_t f = hash[5];
        uint32_t g = hash[6];
        uint32_t h = hash[7];

        for (size_t i = 0; i < 64; ++i) {
            const uint32_t s1 = Sha256RotateRight(e, 6) ^ Sha256RotateRight(e, 11) ^ Sha256RotateRight(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + ch + kRoundConstants[i] + words[i];
            const uint32_t s0 = Sha256RotateRight(a, 2) ^ Sha256RotateRight(a, 13) ^ Sha256RotateRight(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        hash[0] += a;
        hash[1] += b;
        hash[2] += c;
        hash[3] += d;
        hash[4] += e;
        hash[5] += f;
        hash[6] += g;
        hash[7] += h;
    }

    std::ostringstream oss;
    for (const auto value : hash) {
        oss << std::hex << std::setfill('0') << std::setw(8) << value;
    }
    return oss.str();
}

std::string Sha256Bytes(const std::vector<uint8_t> &bytes) {
    return Sha256Hex(bytes.empty() ? nullptr : bytes.data(), bytes.size());
}

std::string Sha256TextOrNone(const std::string &text) {
    if (text.empty()) {
        return "none";
    }
    return Sha256Hex(reinterpret_cast<const uint8_t *>(text.data()), text.size());
}

std::vector<uint8_t> ReadFileBytesOrThrow(const std::string &filename) {
    std::ifstream input(filename, std::ios::binary);
    RequireOrThrow(input.is_open(), "Failed to open file for sha256: " + filename);
    std::vector<char> raw((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    RequireOrThrow(!input.bad(), "Failed to read file for sha256: " + filename);
    return std::vector<uint8_t>(raw.begin(), raw.end());
}

std::string Sha256File(const std::string &filename) {
    return Sha256Bytes(ReadFileBytesOrThrow(filename));
}

GitMetadata ReadGitMetadata() {
    GitMetadata metadata;
    const auto repo_root = FindRepoRoot();
    const auto git_dir = ResolveGitDir(repo_root);
    if (git_dir.empty()) {
        return metadata;
    }

    const auto head = ReadTextFileFirstLine(git_dir / "HEAD");
    constexpr std::string_view ref_prefix = "ref: ";
    if (head.rfind(ref_prefix, 0) == 0) {
        const std::string ref = head.substr(ref_prefix.size());
        constexpr std::string_view heads_prefix = "refs/heads/";
        metadata.branch = ref.rfind(heads_prefix, 0) == 0 ? ref.substr(heads_prefix.size()) : ref;
        const auto ref_path = git_dir / ref;
        if (std::filesystem::exists(ref_path)) {
            metadata.commit = ReadTextFileFirstLine(ref_path);
        }
    } else if (!head.empty()) {
        metadata.branch = "detached";
        metadata.commit = head;
    }

    const auto status = RunGitCommand(repo_root, "status --porcelain --untracked-files=all");
    const auto unstaged_diff = RunGitCommand(repo_root, "diff --binary");
    const auto staged_diff = RunGitCommand(repo_root, "diff --cached --binary");
    metadata.working_tree_dirty = !TrimWhitespace(status).empty();
    metadata.status_porcelain_sha256 = Sha256TextOrNone(status);
    metadata.unstaged_diff_sha256 = Sha256TextOrNone(unstaged_diff);
    metadata.staged_diff_sha256 = Sha256TextOrNone(staged_diff);
    return metadata;
}

std::string CurrentDateTimeString() {
    const auto now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S %z");
    return oss.str();
}

bool EnvFlagEnabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && std::string(value) == "1";
}

std::string EnvOrDefault(const char *name, const std::string &default_value) {
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? std::string(value) : default_value;
}

uint64_t FileSizeOrThrow(const std::string &filename) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(filename, ec);
    RequireOrThrow(!ec, "Failed to stat file: " + filename + ": " + ec.message());
    return static_cast<uint64_t>(size);
}

void ReadExact(std::ifstream &input, void *data, uint64_t size, const std::string &context) {
    RequireOrThrow(size <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()),
                   "Read too large for streamsize: " + context);
    if (size == 0) {
        return;
    }
    input.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
    RequireOrThrow(static_cast<uint64_t>(input.gcount()) == size, "Short read while reading " + context);
}

void WriteExact(std::ofstream &output, const void *data, uint64_t size, const std::string &context) {
    RequireOrThrow(size <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()),
                   "Write too large for streamsize: " + context);
    if (size == 0) {
        return;
    }
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    RequireOrThrow(output.good(), "Failed to write " + context);
}

template <typename T> T ReadPod(std::ifstream &input, const std::string &context) {
    T value{};
    ReadExact(input, &value, sizeof(T), context);
    return value;
}

uint64_t CheckedElementCount(const std::vector<int64_t> &dims) {
    RequireOrThrow(!dims.empty(), "Reference logits dims must not be empty");
    uint64_t count = 1;
    for (const auto dim : dims) {
        RequireOrThrow(dim > 0, "Reference logits dim must be positive: " + std::to_string(dim));
        const auto udim = static_cast<uint64_t>(dim);
        RequireOrThrow(count <= std::numeric_limits<uint64_t>::max() / udim,
                       "Reference logits element count overflows uint64_t");
        count *= udim;
    }
    RequireOrThrow(count <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
                   "Reference logits element count overflows size_t");
    return count;
}

void ValidateFinitePayloadOrThrow(const LogitsBinary &logits, const std::string &context) {
    size_t nan_count = 0;
    size_t inf_count = 0;
    for (const auto value : logits.data) {
        nan_count += std::isnan(value) ? 1 : 0;
        inf_count += std::isinf(value) ? 1 : 0;
    }
    RequireOrThrow(nan_count == 0 && inf_count == 0,
                   context + " contains non-finite logits: nan=" + std::to_string(nan_count)
                       + " inf=" + std::to_string(inf_count));
}

LogitsBinary ReadNewLogitsBinaryFile(const std::string &filename) {
    const auto file_size = FileSizeOrThrow(filename);
    std::ifstream infile(filename, std::ios::binary);
    RequireOrThrow(infile.is_open(), "Failed to open logits file: " + filename);

    const auto header = ReadPod<ReferenceHeader>(infile, filename + " header");
    RequireOrThrow(header.magic == kReferenceMagic,
                   "Invalid reference magic in " + filename + ": " + std::to_string(header.magic));
    RequireOrThrow(header.version == kReferenceVersion,
                   "Unsupported reference version in " + filename + ": " + std::to_string(header.version));
    RequireOrThrow(header.dtype == kReferenceDtypeFloat32,
                   "Unsupported reference dtype in " + filename + ": " + std::to_string(header.dtype));
    RequireOrThrow(header.ndim > 0, "Reference ndim must be positive: " + filename);
    RequireOrThrow(header.ndim <= 8, "Reference ndim is unexpectedly large: " + filename);

    std::vector<int64_t> dims(header.ndim);
    ReadExact(infile, dims.data(), static_cast<uint64_t>(dims.size() * sizeof(int64_t)), filename + " dims");
    const auto num_elements = CheckedElementCount(dims);
    RequireOrThrow(num_elements == header.num_elements,
                   "Header num_elements does not match dims product in " + filename);
    const uint64_t expected_size = sizeof(ReferenceHeader) + static_cast<uint64_t>(dims.size() * sizeof(int64_t))
                                   + num_elements * sizeof(float);
    RequireOrThrow(file_size == expected_size,
                   "Reference file size mismatch for " + filename + ": expected " + std::to_string(expected_size)
                       + " actual " + std::to_string(file_size));

    std::vector<float> data(static_cast<size_t>(num_elements));
    ReadExact(infile, data.data(), num_elements * sizeof(float), filename + " payload");
    RequireOrThrow(infile.peek() == std::char_traits<char>::eof(), "Reference file has trailing bytes: " + filename);
    LogitsBinary logits{.dims = std::move(dims), .data = std::move(data)};
    ValidateFinitePayloadOrThrow(logits, filename);
    return logits;
}

LogitsBinary ReadLegacyLogitsBinaryFile(const std::string &filename) {
    const auto file_size = FileSizeOrThrow(filename);
    std::ifstream infile(filename, std::ios::binary);
    RequireOrThrow(infile.is_open(), "Failed to open legacy logits file: " + filename);

    const auto num_dims = ReadPod<size_t>(infile, filename + " legacy num_dims");
    RequireOrThrow(num_dims > 0 && num_dims <= 8, "Invalid legacy logits ndim in " + filename);
    std::vector<int64_t> dims(num_dims);
    ReadExact(infile, dims.data(), static_cast<uint64_t>(dims.size() * sizeof(int64_t)), filename + " legacy dims");
    const auto num_elements = CheckedElementCount(dims);
    const uint64_t expected_size = sizeof(size_t) + static_cast<uint64_t>(dims.size() * sizeof(int64_t))
                                   + num_elements * sizeof(float);
    RequireOrThrow(file_size == expected_size,
                   "Legacy logits file size mismatch for " + filename + ": expected " + std::to_string(expected_size)
                       + " actual " + std::to_string(file_size));

    std::vector<float> data(static_cast<size_t>(num_elements));
    ReadExact(infile, data.data(), num_elements * sizeof(float), filename + " legacy payload");
    RequireOrThrow(infile.peek() == std::char_traits<char>::eof(), "Legacy logits file has trailing bytes: " + filename);
    LogitsBinary logits{.dims = std::move(dims), .data = std::move(data)};
    ValidateFinitePayloadOrThrow(logits, filename);
    return logits;
}

LogitsBinary ReadLogitsBinaryFile(const std::string &filename) {
    return ReadNewLogitsBinaryFile(filename);
}

LogitsBinary ReadLogitsBinaryFileAuto(const std::string &filename) {
    std::ifstream infile(filename, std::ios::binary);
    RequireOrThrow(infile.is_open(), "Failed to open logits file: " + filename);
    const auto maybe_magic = ReadPod<uint32_t>(infile, filename + " format probe");
    infile.close();
    if (maybe_magic == kReferenceMagic) {
        return ReadNewLogitsBinaryFile(filename);
    }
    return ReadLegacyLogitsBinaryFile(filename);
}

LogitsBinary CopyTensorToLogitsBinary(Tensor &tensor) {
    CHECK_EQ(static_cast<int>(tensor.Dtype()), static_cast<int>(DataType::kFLOAT32));
    auto cpu_tensor = tensor.To(Device(DeviceType::kCPU, 0));
    const auto *data = static_cast<const float *>(cpu_tensor.DataPtr());
    std::vector<float> copied(data, data + cpu_tensor.NumElements());
    return {.dims = cpu_tensor.Dims(), .data = std::move(copied)};
}

void WriteLogitsBinaryFileRaw(const std::string &filename, const LogitsBinary &logits) {
    ValidateFinitePayloadOrThrow(logits, filename);
    const auto num_elements = CheckedElementCount(logits.dims);
    RequireOrThrow(num_elements == logits.data.size(), "Logits data size does not match dims for " + filename);

    std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);
    RequireOrThrow(outfile.is_open(), "Failed to open logits output file: " + filename);
    const ReferenceHeader header{.magic = kReferenceMagic,
                                 .version = kReferenceVersion,
                                 .dtype = kReferenceDtypeFloat32,
                                 .ndim = static_cast<uint32_t>(logits.dims.size()),
                                 .num_elements = num_elements};
    WriteExact(outfile, &header, sizeof(header), filename + " header");
    WriteExact(outfile, logits.dims.data(), static_cast<uint64_t>(logits.dims.size() * sizeof(int64_t)),
               filename + " dims");
    WriteExact(outfile, logits.data.data(), num_elements * sizeof(float), filename + " payload");
    outfile.flush();
    RequireOrThrow(outfile.good(), "Failed to flush logits output file: " + filename);
    outfile.close();
    RequireOrThrow(outfile.good(), "Failed to close logits output file: " + filename);
}

ReferenceWriteResult WriteLogitsBinaryFileAtomic(const std::string &filename, const LogitsBinary &logits,
                                                 bool overwrite) {
    const std::string tmp_filename = filename + ".tmp";
    bool created_tmp = false;
    try {
        RequireOrThrow(overwrite || !std::filesystem::exists(filename),
                       "Refusing to overwrite existing generated logits file: " + filename);
        RequireOrThrow(!std::filesystem::exists(tmp_filename), "Temporary logits file already exists: " + tmp_filename);
        created_tmp = true;
        WriteLogitsBinaryFileRaw(tmp_filename, logits);
        const auto verified = ReadNewLogitsBinaryFile(tmp_filename);
        RequireOrThrow(verified.dims == logits.dims, "Verified logits dims mismatch for " + tmp_filename);
        RequireOrThrow(verified.data == logits.data, "Verified logits payload mismatch for " + tmp_filename);
        const auto sha256 = Sha256File(tmp_filename);
        std::filesystem::rename(tmp_filename, filename);
        created_tmp = false;
        return {.path = filename, .sha256 = sha256};
    } catch (...) {
        if (created_tmp) {
            std::error_code ec;
            std::filesystem::remove(tmp_filename, ec);
        }
        throw;
    }
}

void WriteTextFileAtomic(const std::string &filename, const std::string &contents, bool overwrite) {
    const std::string tmp_filename = filename + ".tmp";
    bool created_tmp = false;
    try {
        RequireOrThrow(overwrite || !std::filesystem::exists(filename),
                       "Refusing to overwrite existing generated metadata file: " + filename);
        RequireOrThrow(!std::filesystem::exists(tmp_filename), "Temporary metadata file already exists: " + tmp_filename);
        created_tmp = true;
        std::ofstream output(tmp_filename, std::ios::binary | std::ios::trunc);
        RequireOrThrow(output.is_open(), "Failed to open metadata output file: " + tmp_filename);
        WriteExact(output, contents.data(), contents.size(), tmp_filename + " contents");
        output.flush();
        RequireOrThrow(output.good(), "Failed to flush metadata output file: " + tmp_filename);
        output.close();
        RequireOrThrow(output.good(), "Failed to close metadata output file: " + tmp_filename);
        created_tmp = true;
        const auto written = ReadFileBytesOrThrow(tmp_filename);
        const std::string roundtrip(reinterpret_cast<const char *>(written.data()), written.size());
        RequireOrThrow(roundtrip == contents, "Verified metadata contents mismatch for " + tmp_filename);
        std::filesystem::rename(tmp_filename, filename);
        created_tmp = false;
    } catch (...) {
        if (created_tmp) {
            std::error_code ec;
            std::filesystem::remove(tmp_filename, ec);
        }
        throw;
    }
}

void AppendTensorRecord(std::vector<uint8_t> &bytes, std::string_view label, Tensor &tensor) {
    auto cpu_tensor = tensor.To(Device(DeviceType::kCPU, 0));
    AppendStringRecord(bytes, label);
    const int32_t dtype = static_cast<int32_t>(cpu_tensor.Dtype());
    const uint64_t ndim = static_cast<uint64_t>(cpu_tensor.Dims().size());
    const uint64_t num_elements = static_cast<uint64_t>(cpu_tensor.NumElements());
    const uint64_t size_in_bytes = static_cast<uint64_t>(cpu_tensor.SizeInBytes());
    AppendPod(bytes, dtype);
    AppendPod(bytes, ndim);
    for (const auto dim : cpu_tensor.Dims()) {
        AppendPod(bytes, dim);
    }
    AppendPod(bytes, num_elements);
    AppendPod(bytes, size_in_bytes);
    AppendBytes(bytes, cpu_tensor.DataPtr(), static_cast<size_t>(size_in_bytes));
}

std::string HashTensorRecord(std::string_view label, Tensor &tensor) {
    std::vector<uint8_t> bytes;
    AppendTensorRecord(bytes, label, tensor);
    return Sha256Bytes(bytes);
}

LogitsComparisonMetrics CompareLogitsFull(const LogitsBinary &reference, const LogitsBinary &candidate,
                                          float tolerance) {
    RequireOrThrow(reference.dims == candidate.dims,
                   "reference dims=" + DimsToString(reference.dims) + " candidate dims=" + DimsToString(candidate.dims));
    RequireOrThrow(reference.data.size() == candidate.data.size(), "Logits element count mismatch");
    RequireOrThrow(!reference.data.empty(), "Logits comparison input is empty");

    const bool has_rows = reference.dims.size() == 3;
    const size_t batch_size = has_rows ? static_cast<size_t>(reference.dims[0]) : 0;
    const size_t sequence_length = has_rows ? static_cast<size_t>(reference.dims[1]) : 0;
    const size_t vocab_size = has_rows ? static_cast<size_t>(reference.dims[2]) : 0;
    const size_t row_count = has_rows ? batch_size * sequence_length : 0;
    std::vector<double> row_sum_abs(row_count, 0.0);
    std::vector<double> row_sum_sq(row_count, 0.0);

    LogitsComparisonMetrics metrics;
    metrics.elements = reference.data.size();
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double dot = 0.0;
    double ref_sq = 0.0;
    double cand_sq = 0.0;
    for (size_t i = 0; i < reference.data.size(); ++i) {
        const double ref = reference.data[i];
        const double cand = candidate.data[i];
        metrics.reference_nan_count += std::isnan(ref) ? 1 : 0;
        metrics.candidate_nan_count += std::isnan(cand) ? 1 : 0;
        metrics.reference_inf_count += std::isinf(ref) ? 1 : 0;
        metrics.candidate_inf_count += std::isinf(cand) ? 1 : 0;
        const bool nonfinite = !std::isfinite(ref) || !std::isfinite(cand);
        const double diff = nonfinite ? std::numeric_limits<double>::infinity() : std::abs(ref - cand);
        const bool mismatch = nonfinite || diff > tolerance;
        if (!metrics.has_mismatch && mismatch) {
            metrics.has_mismatch = true;
            metrics.first_mismatch = i;
            if (has_rows) {
                metrics.first_mismatch_batch = i / (sequence_length * vocab_size);
                const size_t remainder = i % (sequence_length * vocab_size);
                metrics.first_mismatch_sequence = remainder / vocab_size;
                metrics.first_mismatch_vocab = remainder % vocab_size;
            }
        }
        metrics.max_abs = std::max(metrics.max_abs, diff);
        sum_abs += diff;
        sum_sq += diff * diff;
        if (!nonfinite) {
            dot += ref * cand;
            ref_sq += ref * ref;
            cand_sq += cand * cand;
        }
        if (mismatch) {
            ++metrics.count_gt_tolerance;
        }
        if (has_rows) {
            const size_t row_index = i / vocab_size;
            row_sum_abs[row_index] += diff;
            row_sum_sq[row_index] += diff * diff;
        }
    }
    metrics.mean_abs = sum_abs / static_cast<double>(reference.data.size());
    metrics.rmse = std::sqrt(sum_sq / static_cast<double>(reference.data.size()));
    const double denom = std::sqrt(ref_sq) * std::sqrt(cand_sq);
    metrics.cosine_defined = denom > 0.0 && std::isfinite(denom);
    metrics.cosine = metrics.cosine_defined ? dot / denom : 0.0;
    metrics.reference0 = reference.data[0];
    metrics.candidate0 = candidate.data[0];
    RequireOrThrow(kLogitProbeIndex < reference.data.size(), "Logit probe index is out of range");
    metrics.reference_probe = reference.data[kLogitProbeIndex];
    metrics.candidate_probe = candidate.data[kLogitProbeIndex];

    if (has_rows) {
        metrics.row_metrics.reserve(row_count);
        for (size_t row = 0; row < row_count; ++row) {
            RowDiffMetrics row_metrics;
            row_metrics.row_index = row;
            row_metrics.batch = row / sequence_length;
            row_metrics.sequence = row % sequence_length;
            row_metrics.mean_abs = row_sum_abs[row] / static_cast<double>(vocab_size);
            row_metrics.rmse = std::sqrt(row_sum_sq[row] / static_cast<double>(vocab_size));
            const size_t row_begin = row * vocab_size;
            for (size_t vocab = 0; vocab < vocab_size; ++vocab) {
                const size_t index = row_begin + vocab;
                const double ref = reference.data[index];
                const double cand = candidate.data[index];
                const bool nonfinite = !std::isfinite(ref) || !std::isfinite(cand);
                const double diff = nonfinite ? std::numeric_limits<double>::infinity() : std::abs(ref - cand);
                row_metrics.max_abs = std::max(row_metrics.max_abs, diff);
                if (nonfinite || diff > tolerance) {
                    ++row_metrics.count_gt_tolerance;
                }
            }
            if (row_metrics.count_gt_tolerance > 0) {
                ++metrics.rows_with_gt_tolerance;
            }
            if (row_metrics.count_gt_tolerance == vocab_size) {
                ++metrics.rows_with_full_vocab_gt_tolerance;
            }
            metrics.row_metrics.push_back(row_metrics);
        }
    }
    return metrics;
}

void LogTopRows(const std::string &label, std::vector<RowDiffMetrics> rows, std::string_view sort_key,
                size_t vocab_size) {
    if (sort_key == "count") {
        std::sort(rows.begin(), rows.end(), [](const RowDiffMetrics &a, const RowDiffMetrics &b) {
            if (a.count_gt_tolerance != b.count_gt_tolerance) {
                return a.count_gt_tolerance > b.count_gt_tolerance;
            }
            return a.max_abs > b.max_abs;
        });
    } else {
        std::sort(rows.begin(), rows.end(), [](const RowDiffMetrics &a, const RowDiffMetrics &b) {
            if (a.max_abs != b.max_abs) {
                return a.max_abs > b.max_abs;
            }
            return a.count_gt_tolerance > b.count_gt_tolerance;
        });
    }

    const size_t limit = std::min<size_t>(10, rows.size());
    for (size_t rank = 0; rank < limit; ++rank) {
        const auto &row = rows[rank];
        LOG(INFO) << "LOGITS_ROW_TOP label=" << label
                  << " sort=" << sort_key
                  << " rank=" << rank
                  << " batch=" << row.batch
                  << " sequence=" << row.sequence
                  << " max_abs=" << row.max_abs
                  << " mean_abs=" << row.mean_abs
                  << " rmse=" << row.rmse
                  << " count_gt_tolerance=" << row.count_gt_tolerance
                  << " full_vocab_gt=" << (row.count_gt_tolerance == vocab_size);
    }
}

void LogLogitsComparison(const std::string &label, const LogitsComparisonMetrics &metrics, float tolerance) {
    const size_t vocab_size = metrics.row_metrics.empty() ? 0 : metrics.elements / metrics.row_metrics.size();
    LOG(INFO) << "LOGITS_FULL_COMPARE label=" << label
              << " tolerance=" << tolerance
              << " elements=" << metrics.elements
              << " first_mismatch=" << (metrics.has_mismatch ? std::to_string(metrics.first_mismatch) : std::string("none"))
              << " first_mismatch_batch=" << metrics.first_mismatch_batch
              << " first_mismatch_sequence=" << metrics.first_mismatch_sequence
              << " first_mismatch_vocab=" << metrics.first_mismatch_vocab
              << " max_abs=" << metrics.max_abs
              << " mean_abs=" << metrics.mean_abs
              << " rmse=" << metrics.rmse
              << " count_gt_1e3=" << metrics.count_gt_tolerance
              << " rows_with_gt_1e3=" << metrics.rows_with_gt_tolerance
              << " rows_with_full_vocab_gt_1e3=" << metrics.rows_with_full_vocab_gt_tolerance
              << " reference_nan_count=" << metrics.reference_nan_count
              << " candidate_nan_count=" << metrics.candidate_nan_count
              << " reference_inf_count=" << metrics.reference_inf_count
              << " candidate_inf_count=" << metrics.candidate_inf_count
              << " cosine=" << (metrics.cosine_defined ? std::to_string(metrics.cosine) : std::string("undefined"))
              << " ref0=" << metrics.reference0
              << " candidate0=" << metrics.candidate0
              << " diff0=" << std::abs(metrics.reference0 - metrics.candidate0)
              << " ref385973=" << metrics.reference_probe
              << " candidate385973=" << metrics.candidate_probe
              << " diff385973=" << std::abs(metrics.reference_probe - metrics.candidate_probe);
    for (const auto &row : metrics.row_metrics) {
        LOG(INFO) << "LOGITS_ROW_COMPARE label=" << label
                  << " batch=" << row.batch
                  << " sequence=" << row.sequence
                  << " max_abs=" << row.max_abs
                  << " mean_abs=" << row.mean_abs
                  << " rmse=" << row.rmse
                  << " count_gt_tolerance=" << row.count_gt_tolerance
                  << " full_vocab_gt=" << (vocab_size != 0 && row.count_gt_tolerance == vocab_size);
    }
    LogTopRows(label, metrics.row_metrics, "count", vocab_size);
    LogTopRows(label, metrics.row_metrics, "max_abs", vocab_size);
}


struct TensorComparisonMetrics {
    std::vector<int64_t> dims;
    size_t elements = 0;
    bool has_bitwise_mismatch = false;
    size_t first_mismatch = 0;
    size_t first_mismatch_batch = 0;
    size_t first_mismatch_sequence = 0;
    size_t first_mismatch_feature = 0;
    float first_lhs = 0.0f;
    float first_rhs = 0.0f;
    double first_mismatch_abs = 0.0;
    size_t count_bitwise_mismatch = 0;
    bool has_gt_tolerance = false;
    size_t first_gt_tolerance = 0;
    size_t first_gt_tolerance_batch = 0;
    size_t first_gt_tolerance_sequence = 0;
    size_t first_gt_tolerance_feature = 0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rmse = 0.0;
    size_t count_gt_tolerance = 0;
    size_t lhs_nan_count = 0;
    size_t rhs_nan_count = 0;
    size_t lhs_inf_count = 0;
    size_t rhs_inf_count = 0;
    size_t lhs_nonzero_count = 0;
    size_t rhs_nonzero_count = 0;
    std::vector<RowDiffMetrics> row_metrics;
};

struct FirstDivergenceTracker {
    bool found = false;
    int update = -1;
    int micro = -1;
    std::string stage = "none";
    std::string tensor = "none";
    double max_abs = 0.0;
    size_t first_mismatch = 0;

    void ObserveTensor(int observed_update, int observed_micro, const std::string &observed_stage,
                       const std::string &observed_tensor, const TensorComparisonMetrics &metrics) {
        if (!found && metrics.count_bitwise_mismatch > 0) {
            found = true;
            update = observed_update;
            micro = observed_micro;
            stage = observed_stage;
            tensor = observed_tensor;
            max_abs = metrics.max_abs;
            first_mismatch = metrics.first_mismatch;
        }
    }

    void ObserveScalar(int observed_update, int observed_micro, const std::string &observed_stage,
                       const std::string &observed_tensor, bool bitwise_mismatch, double diff) {
        if (!found && bitwise_mismatch) {
            found = true;
            update = observed_update;
            micro = observed_micro;
            stage = observed_stage;
            tensor = observed_tensor;
            max_abs = diff;
            first_mismatch = 0;
        }
    }

    void Log() const {
        LOG(INFO) << "FIRST_DIVERGENCE update=" << update
                  << " micro=" << micro
                  << " stage=" << stage
                  << " tensor=" << tensor
                  << " max_abs=" << max_abs
                  << " first_mismatch=" << (found ? std::to_string(first_mismatch) : std::string("none"));
    }
};

std::string OptionalIndexString(bool has_index, size_t index) {
    return has_index ? std::to_string(index) : std::string("none");
}

void Decode3DIndex(const std::vector<int64_t> &dims, size_t index, size_t *batch, size_t *sequence, size_t *feature) {
    if (dims.size() != 3) {
        *batch = 0;
        *sequence = 0;
        *feature = index;
        return;
    }
    const size_t sequence_length = static_cast<size_t>(dims[1]);
    const size_t feature_size = static_cast<size_t>(dims[2]);
    *batch = index / (sequence_length * feature_size);
    const size_t remainder = index % (sequence_length * feature_size);
    *sequence = remainder / feature_size;
    *feature = remainder % feature_size;
}

TensorComparisonMetrics CompareFloatData(const std::vector<int64_t> &dims, const float *lhs, const float *rhs,
                                         size_t elements, float tolerance) {
    const bool has_rows = dims.size() == 3;
    const size_t row_count = has_rows ? static_cast<size_t>(dims[0] * dims[1]) : 0;
    const size_t feature_size = has_rows ? static_cast<size_t>(dims[2]) : 0;
    std::vector<double> row_sum_abs(row_count, 0.0);
    std::vector<double> row_sum_sq(row_count, 0.0);
    std::vector<double> row_max_abs(row_count, 0.0);
    std::vector<size_t> row_count_gt(row_count, 0);

    TensorComparisonMetrics metrics;
    metrics.dims = dims;
    metrics.elements = elements;

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < elements; ++i) {
        const float lhs_value = lhs[i];
        const float rhs_value = rhs[i];
        metrics.lhs_nan_count += std::isnan(lhs_value) ? 1 : 0;
        metrics.rhs_nan_count += std::isnan(rhs_value) ? 1 : 0;
        metrics.lhs_inf_count += std::isinf(lhs_value) ? 1 : 0;
        metrics.rhs_inf_count += std::isinf(rhs_value) ? 1 : 0;
        metrics.lhs_nonzero_count += lhs_value != 0.0f ? 1 : 0;
        metrics.rhs_nonzero_count += rhs_value != 0.0f ? 1 : 0;

        const bool nonfinite = !std::isfinite(lhs_value) || !std::isfinite(rhs_value);
        const double diff = nonfinite ? std::numeric_limits<double>::infinity()
                                      : std::abs(static_cast<double>(lhs_value) - static_cast<double>(rhs_value));
        const bool bitwise_mismatch = std::memcmp(&lhs_value, &rhs_value, sizeof(float)) != 0;
        if (bitwise_mismatch) {
            ++metrics.count_bitwise_mismatch;
            if (!metrics.has_bitwise_mismatch) {
                metrics.has_bitwise_mismatch = true;
                metrics.first_mismatch = i;
                metrics.first_lhs = lhs_value;
                metrics.first_rhs = rhs_value;
                metrics.first_mismatch_abs = diff;
                Decode3DIndex(dims, i, &metrics.first_mismatch_batch, &metrics.first_mismatch_sequence,
                              &metrics.first_mismatch_feature);
            }
        }

        const bool exceeds_tolerance = nonfinite || diff > tolerance;
        if (exceeds_tolerance) {
            ++metrics.count_gt_tolerance;
            if (!metrics.has_gt_tolerance) {
                metrics.has_gt_tolerance = true;
                metrics.first_gt_tolerance = i;
                Decode3DIndex(dims, i, &metrics.first_gt_tolerance_batch, &metrics.first_gt_tolerance_sequence,
                              &metrics.first_gt_tolerance_feature);
            }
        }

        metrics.max_abs = std::max(metrics.max_abs, diff);
        sum_abs += diff;
        sum_sq += diff * diff;
        if (has_rows) {
            const size_t row = i / feature_size;
            row_sum_abs[row] += diff;
            row_sum_sq[row] += diff * diff;
            row_max_abs[row] = std::max(row_max_abs[row], diff);
            if (exceeds_tolerance) {
                ++row_count_gt[row];
            }
        }
    }

    if (elements > 0) {
        metrics.mean_abs = sum_abs / static_cast<double>(elements);
        metrics.rmse = std::sqrt(sum_sq / static_cast<double>(elements));
    }

    if (has_rows) {
        metrics.row_metrics.reserve(row_count);
        for (size_t row = 0; row < row_count; ++row) {
            RowDiffMetrics row_metrics;
            row_metrics.row_index = row;
            row_metrics.batch = row / static_cast<size_t>(dims[1]);
            row_metrics.sequence = row % static_cast<size_t>(dims[1]);
            row_metrics.max_abs = row_max_abs[row];
            row_metrics.mean_abs = row_sum_abs[row] / static_cast<double>(feature_size);
            row_metrics.rmse = std::sqrt(row_sum_sq[row] / static_cast<double>(feature_size));
            row_metrics.count_gt_tolerance = row_count_gt[row];
            metrics.row_metrics.push_back(row_metrics);
        }
    }
    return metrics;
}

TensorComparisonMetrics CompareFloatTensors(Tensor &lhs, Tensor &rhs, float tolerance) {
    CHECK_EQ(static_cast<int>(lhs.Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(rhs.Dtype()), static_cast<int>(DataType::kFLOAT32));
    auto lhs_cpu = lhs.To(Device(DeviceType::kCPU, 0));
    auto rhs_cpu = rhs.To(Device(DeviceType::kCPU, 0));
    CHECK(lhs_cpu.Dims() == rhs_cpu.Dims()) << "lhs dims=" << DimsToString(lhs_cpu.Dims())
                                            << " rhs dims=" << DimsToString(rhs_cpu.Dims());
    CHECK_EQ(lhs_cpu.NumElements(), rhs_cpu.NumElements());
    return CompareFloatData(lhs_cpu.Dims(), static_cast<const float *>(lhs_cpu.DataPtr()),
                            static_cast<const float *>(rhs_cpu.DataPtr()), lhs_cpu.NumElements(), tolerance);
}

RowDiffMetrics BestRowByMaxAbs(const TensorComparisonMetrics &metrics) {
    CHECK(!metrics.row_metrics.empty());
    return *std::max_element(metrics.row_metrics.begin(), metrics.row_metrics.end(),
                             [](const RowDiffMetrics &lhs, const RowDiffMetrics &rhs) {
                                 if (lhs.max_abs != rhs.max_abs) {
                                     return lhs.max_abs < rhs.max_abs;
                                 }
                                 return lhs.count_gt_tolerance < rhs.count_gt_tolerance;
                             });
}

RowDiffMetrics BestRowByCount(const TensorComparisonMetrics &metrics) {
    CHECK(!metrics.row_metrics.empty());
    return *std::max_element(metrics.row_metrics.begin(), metrics.row_metrics.end(),
                             [](const RowDiffMetrics &lhs, const RowDiffMetrics &rhs) {
                                 if (lhs.count_gt_tolerance != rhs.count_gt_tolerance) {
                                     return lhs.count_gt_tolerance < rhs.count_gt_tolerance;
                                 }
                                 return lhs.max_abs < rhs.max_abs;
                             });
}

void LogTensorTopRows(const std::string &label, std::vector<RowDiffMetrics> rows, std::string_view sort_key) {
    if (rows.empty()) {
        return;
    }
    if (sort_key == "count") {
        std::sort(rows.begin(), rows.end(), [](const RowDiffMetrics &lhs, const RowDiffMetrics &rhs) {
            if (lhs.count_gt_tolerance != rhs.count_gt_tolerance) {
                return lhs.count_gt_tolerance > rhs.count_gt_tolerance;
            }
            return lhs.max_abs > rhs.max_abs;
        });
    } else {
        std::sort(rows.begin(), rows.end(), [](const RowDiffMetrics &lhs, const RowDiffMetrics &rhs) {
            if (lhs.max_abs != rhs.max_abs) {
                return lhs.max_abs > rhs.max_abs;
            }
            return lhs.count_gt_tolerance > rhs.count_gt_tolerance;
        });
    }

    const size_t limit = std::min<size_t>(10, rows.size());
    for (size_t rank = 0; rank < limit; ++rank) {
        const auto &row = rows[rank];
        LOG(INFO) << "TENSOR_ROW_TOP label=" << label
                  << " sort=" << sort_key
                  << " rank=" << rank
                  << " batch=" << row.batch
                  << " sequence=" << row.sequence
                  << " max_abs=" << row.max_abs
                  << " mean_abs=" << row.mean_abs
                  << " rmse=" << row.rmse
                  << " count_gt_tolerance=" << row.count_gt_tolerance;
    }
}

void LogHistoricalRowProbe(const std::string &label, const TensorComparisonMetrics &metrics, size_t batch, size_t sequence) {
    if (metrics.dims.size() != 3) {
        return;
    }
    const size_t sequence_length = static_cast<size_t>(metrics.dims[1]);
    const size_t row = batch * sequence_length + sequence;
    if (row >= metrics.row_metrics.size()) {
        return;
    }
    const auto &row_metrics = metrics.row_metrics[row];
    LOG(INFO) << "TENSOR_ROW_PROBE label=" << label
              << " batch=" << batch
              << " sequence=" << sequence
              << " max_abs=" << row_metrics.max_abs
              << " mean_abs=" << row_metrics.mean_abs
              << " rmse=" << row_metrics.rmse
              << " count_gt_tolerance=" << row_metrics.count_gt_tolerance;
}

void LogTensorDiffComparison(const std::string &label, const TensorComparisonMetrics &metrics, float tolerance) {
    LOG(INFO) << "TENSOR_COMPARE label=" << label
              << " tolerance=" << tolerance
              << " dims=" << DimsToString(metrics.dims)
              << " elements=" << metrics.elements
              << " bitwise_mismatch_count=" << metrics.count_bitwise_mismatch
              << " first_mismatch=" << OptionalIndexString(metrics.has_bitwise_mismatch, metrics.first_mismatch)
              << " first_mismatch_batch=" << metrics.first_mismatch_batch
              << " first_mismatch_sequence=" << metrics.first_mismatch_sequence
              << " first_mismatch_feature=" << metrics.first_mismatch_feature
              << " first_lhs=" << metrics.first_lhs
              << " first_rhs=" << metrics.first_rhs
              << " first_mismatch_abs=" << metrics.first_mismatch_abs
              << " max_abs=" << metrics.max_abs
              << " mean_abs=" << metrics.mean_abs
              << " rmse=" << metrics.rmse
              << " count_gt_tolerance=" << metrics.count_gt_tolerance
              << " first_gt_tolerance=" << OptionalIndexString(metrics.has_gt_tolerance, metrics.first_gt_tolerance)
              << " first_gt_batch=" << metrics.first_gt_tolerance_batch
              << " first_gt_sequence=" << metrics.first_gt_tolerance_sequence
              << " first_gt_feature=" << metrics.first_gt_tolerance_feature
              << " lhs_nan_count=" << metrics.lhs_nan_count
              << " rhs_nan_count=" << metrics.rhs_nan_count
              << " lhs_inf_count=" << metrics.lhs_inf_count
              << " rhs_inf_count=" << metrics.rhs_inf_count
              << " lhs_nonzero_count=" << metrics.lhs_nonzero_count
              << " rhs_nonzero_count=" << metrics.rhs_nonzero_count;
    if (!metrics.row_metrics.empty()) {
        const auto max_row = BestRowByMaxAbs(metrics);
        const auto count_row = BestRowByCount(metrics);
        LOG(INFO) << "TENSOR_ROW_AUTO label=" << label
                  << " kind=max_abs batch=" << max_row.batch
                  << " sequence=" << max_row.sequence
                  << " max_abs=" << max_row.max_abs
                  << " mean_abs=" << max_row.mean_abs
                  << " rmse=" << max_row.rmse
                  << " count_gt_tolerance=" << max_row.count_gt_tolerance;
        LOG(INFO) << "TENSOR_ROW_AUTO label=" << label
                  << " kind=count_gt_tolerance batch=" << count_row.batch
                  << " sequence=" << count_row.sequence
                  << " max_abs=" << count_row.max_abs
                  << " mean_abs=" << count_row.mean_abs
                  << " rmse=" << count_row.rmse
                  << " count_gt_tolerance=" << count_row.count_gt_tolerance;
        if (metrics.has_bitwise_mismatch) {
            LOG(INFO) << "TENSOR_ROW_AUTO label=" << label
                      << " kind=first_mismatch batch=" << metrics.first_mismatch_batch
                      << " sequence=" << metrics.first_mismatch_sequence
                      << " feature=" << metrics.first_mismatch_feature;
        }
        LogTensorTopRows(label, metrics.row_metrics, "count");
        LogTensorTopRows(label, metrics.row_metrics, "max_abs");
        LogHistoricalRowProbe(label, metrics, 0, 45);
        LogHistoricalRowProbe(label, metrics, 0, 60);
        LogHistoricalRowProbe(label, metrics, 1, 35);
    }
}

bool TensorHasNonFiniteDiffInput(const TensorComparisonMetrics &metrics) {
    return metrics.lhs_nan_count != 0 || metrics.rhs_nan_count != 0 || metrics.lhs_inf_count != 0
           || metrics.rhs_inf_count != 0;
}

std::string CudaRuntimeVersionString() {
#ifdef USE_CUDA
    int version = 0;
    const auto status = cudaRuntimeGetVersion(&version);
    if (status != cudaSuccess) {
        return std::string("unavailable: ") + cudaGetErrorString(status);
    }
    return std::to_string(version);
#else
    return "not_built_with_cuda";
#endif
}

std::string GpuNameString(const Device &device) {
#ifdef USE_CUDA
    if (!device.IsCUDA()) {
        return "not_cuda_device";
    }
    cudaDeviceProp prop{};
    const auto status = cudaGetDeviceProperties(&prop, device.Index());
    if (status != cudaSuccess) {
        return std::string("unavailable: ") + cudaGetErrorString(status);
    }
    return prop.name;
#else
    (void)device;
    return "not_built_with_cuda";
#endif
}

std::string CublasVersionString() {
#ifdef USE_CUDA
    cublasHandle_t handle = nullptr;
    const auto create_status = cublasCreate(&handle);
    if (create_status != CUBLAS_STATUS_SUCCESS) {
        return "unavailable: cublasCreate failed with code " + std::to_string(static_cast<int>(create_status));
    }
    int version = 0;
    const auto version_status = cublasGetVersion(handle, &version);
    cublasDestroy(handle);
    if (version_status != CUBLAS_STATUS_SUCCESS) {
        return "unavailable: cublasGetVersion failed with code " + std::to_string(static_cast<int>(version_status));
    }
    return std::to_string(version);
#else
    return "not_built_with_cuda";
#endif
}

std::string BuildTypeString() {
#ifdef NDEBUG
    return "Release_or_NDEBUG";
#else
    return "Debug_or_no_NDEBUG";
#endif
}

std::string UseCudaString() {
#ifdef USE_CUDA
    return "ON";
#else
    return "OFF";
#endif
}

} // namespace

class GPT2TrainingTest : public ::testing::Test {
protected:
    void SetUp() override {
        llmc_filepath = "../../Data/gpt2_124M.bin";
        input_bin = "../../Data/tinyshakespeare/tiny_shakespeare_train.bin";
        tokenizer_bin = "../../Data/gpt2_tokenizer.bin";
        logits_reference = "../../Data/gpt2_logits_reference.bin";
        tied_logits_reference = "../../Data/gpt2_logits_reference_tied_10_updates.bin";
        tied_logits_reference_generated = "../../Data/gpt2_logits_reference_tied_10_updates.bin.generated";
        tied_logits_reference_meta_generated = "../../Data/gpt2_logits_reference_tied_10_updates.meta.txt.generated";

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
    


    struct DiagnosticMicrobatch {
        std::shared_ptr<Tensor> x_cpu;
        std::shared_ptr<Tensor> y_cpu;
    };

    struct DiagnosticForwardResult {
        std::shared_ptr<Tensor> logits;
        std::shared_ptr<Tensor> loss;
        float loss_value = 0.0f;
    };

    struct PairedTrainingRun {
        int run_id = 0;
        std::unique_ptr<GPT2> model;
        std::unique_ptr<optimizers::SGD> optimizer;
        std::unique_ptr<nn::CrossEntropyLoss> loss_fn;
        GPT2ForwardDiagnostics diagnostics;
    };

    struct ForwardComparisonBundle {
        TensorComparisonMetrics final_block_output;
        TensorComparisonMetrics ln_f_output;
        TensorComparisonMetrics lm_head_input;
        TensorComparisonMetrics logits;

        bool BitwiseEqual() const {
            return final_block_output.count_bitwise_mismatch == 0 && ln_f_output.count_bitwise_mismatch == 0
                   && lm_head_input.count_bitwise_mismatch == 0 && logits.count_bitwise_mismatch == 0;
        }
    };

    struct ParameterDiffRecord {
        std::string name;
        std::string dims;
        bool grad_present_lhs = false;
        bool grad_present_rhs = false;
        bool grad_presence_mismatch = false;
        bool has_grad_metrics = false;
        TensorComparisonMetrics grad_metrics;
        TensorComparisonMetrics param_metrics;
    };

    std::vector<DiagnosticMicrobatch> LoadDiagnosticMicrobatches(int count) {
        std::vector<DiagnosticMicrobatch> microbatches;
        microbatches.reserve(static_cast<size_t>(count));
        auto train_iter = train_loader->begin();
        for (int i = 0; i < count; ++i) {
            auto [x, y] = *train_iter;
            ++train_iter;
            microbatches.push_back({.x_cpu = x, .y_cpu = y});
            LOG(INFO) << "PAIRED_MICROBATCH index=" << i
                      << " input_dims=" << DimsToString(x->Dims())
                      << " target_dims=" << DimsToString(y->Dims())
                      << " input_hash=" << HashTensorRecord("paired_input", *x)
                      << " target_hash=" << HashTensorRecord("paired_target", *y);
        }
        return microbatches;
    }

    void InitializePairedTrainingRun(PairedTrainingRun &run, int run_id) {
        run.run_id = run_id;
        run.model = GPT2::FromLLMC(llmc_filepath);
        run.model->To(device);
        run.optimizer = std::make_unique<optimizers::SGD>(run.model->Parameters(), learning_rate);
        run.loss_fn = std::make_unique<nn::CrossEntropyLoss>();
        run.loss_fn->To(device);
        run.diagnostics.capture_block_outputs = true;
        run.model->SetForwardDiagnostics(&run.diagnostics);
        LOG(INFO) << "PAIRED_RUN_INIT run=" << run_id
                  << " device=" << (device.IsCUDA() ? "cuda" : "cpu")
                  << " diagnostics_capture_blocks=" << run.diagnostics.capture_block_outputs;
    }

    std::shared_ptr<Tensor> TensorToRunDevice(const std::shared_ptr<Tensor> &tensor) {
        return std::make_shared<Tensor>(tensor->To(device));
    }

    float ReadScalarTensor(Tensor &tensor) {
        auto cpu = tensor.To(Device(DeviceType::kCPU, 0));
        CHECK_EQ(cpu.NumElements(), 1);
        return static_cast<const float *>(cpu.DataPtr())[0];
    }

    DiagnosticForwardResult RunDiagnosticForward(PairedTrainingRun &run, const DiagnosticMicrobatch &microbatch) {
        auto x = TensorToRunDevice(microbatch.x_cpu);
        auto y = TensorToRunDevice(microbatch.y_cpu);
        auto outputs = run.model->Forward({x, y});
        CHECK_EQ(outputs.size(), 1);
        auto run_logits = outputs[0];
        CHECK(run_logits != nullptr);
        auto loss = run.loss_fn->Forward({run_logits, y})[0];
        CHECK(loss != nullptr);
        const float loss_value = ReadScalarTensor(*loss);
        return {.logits = run_logits, .loss = loss, .loss_value = loss_value};
    }

    void LogLossComparison(const std::string &stage, int update, int micro, float lhs, float rhs,
                           FirstDivergenceTracker &tracker) {
        const bool bitwise_mismatch = std::memcmp(&lhs, &rhs, sizeof(float)) != 0;
        const double diff = std::abs(static_cast<double>(lhs) - static_cast<double>(rhs));
        LOG(INFO) << "LOSS_COMPARE stage=" << stage
                  << " update=" << update
                  << " micro=" << micro
                  << " lhs=" << lhs
                  << " rhs=" << rhs
                  << " bitwise_mismatch=" << bitwise_mismatch
                  << " max_abs=" << diff
                  << " lhs_finite=" << std::isfinite(lhs)
                  << " rhs_finite=" << std::isfinite(rhs);
        EXPECT_TRUE(std::isfinite(lhs));
        EXPECT_TRUE(std::isfinite(rhs));
        tracker.ObserveScalar(update, micro, stage, "loss", bitwise_mismatch, diff);
    }

    TensorComparisonMetrics CompareAndLogForwardTensor(const std::string &stage, int update, int micro,
                                                       const std::string &tensor_name,
                                                       const std::shared_ptr<Tensor> &lhs,
                                                       const std::shared_ptr<Tensor> &rhs,
                                                       FirstDivergenceTracker &tracker) {
        CHECK(lhs != nullptr) << tensor_name;
        CHECK(rhs != nullptr) << tensor_name;
        auto metrics = CompareFloatTensors(*lhs, *rhs, kDiagnosticTolerance);
        const std::string label = stage + "." + tensor_name;
        LogTensorDiffComparison(label, metrics, kDiagnosticTolerance);
        EXPECT_FALSE(TensorHasNonFiniteDiffInput(metrics)) << label;
        tracker.ObserveTensor(update, micro, stage, tensor_name, metrics);
        return metrics;
    }

    void CompareBlockOutputsIfNeeded(const std::string &stage, int update, int micro,
                                     const GPT2ForwardDiagnostics &lhs, const GPT2ForwardDiagnostics &rhs,
                                     FirstDivergenceTracker &tracker) {
        auto entering_metrics = CompareAndLogForwardTensor(stage, update, micro, "transformer_input", lhs.transformer_input,
                                                           rhs.transformer_input, tracker);
        bool entering_equal = entering_metrics.count_bitwise_mismatch == 0;
        const size_t block_count = std::min(lhs.block_outputs.size(), rhs.block_outputs.size());
        for (size_t block = 0; block < block_count; ++block) {
            auto metrics = CompareAndLogForwardTensor(stage, update, micro,
                                                      "block_" + std::to_string(block) + "_output",
                                                      lhs.block_outputs[block], rhs.block_outputs[block], tracker);
            const bool leaving_equal = metrics.count_bitwise_mismatch == 0;
            LOG(INFO) << "BLOCK_COMPARE stage=" << stage
                      << " update=" << update
                      << " micro=" << micro
                      << " block=" << block
                      << " entering_equal=" << entering_equal
                      << " leaving_equal=" << leaving_equal
                      << " leaving_max_abs=" << metrics.max_abs
                      << " leaving_mean_abs=" << metrics.mean_abs
                      << " leaving_rmse=" << metrics.rmse
                      << " leaving_count_gt_tolerance=" << metrics.count_gt_tolerance;
            if (!leaving_equal) {
                LOG(INFO) << "BLOCK_FIRST_DIFFERENCE stage=" << stage
                          << " update=" << update
                          << " micro=" << micro
                          << " block=" << block
                          << " entering_equal=" << entering_equal
                          << " leaving_max_abs=" << metrics.max_abs
                          << " leaving_mean_abs=" << metrics.mean_abs
                          << " leaving_rmse=" << metrics.rmse;
                break;
            }
            entering_equal = leaving_equal;
        }
        if (lhs.block_outputs.size() != rhs.block_outputs.size()) {
            LOG(INFO) << "BLOCK_CAPTURE_SIZE_MISMATCH stage=" << stage
                      << " lhs_blocks=" << lhs.block_outputs.size()
                      << " rhs_blocks=" << rhs.block_outputs.size();
        }
    }

    ForwardComparisonBundle CompareForwardDiagnostics(const std::string &stage, int update, int micro,
                                                      const GPT2ForwardDiagnostics &lhs,
                                                      const GPT2ForwardDiagnostics &rhs,
                                                      FirstDivergenceTracker &tracker) {
        ForwardComparisonBundle bundle;
        bundle.final_block_output = CompareAndLogForwardTensor(stage, update, micro, "final_block_output",
                                                               lhs.final_block_output, rhs.final_block_output, tracker);
        bundle.ln_f_output = CompareAndLogForwardTensor(stage, update, micro, "ln_f_output", lhs.ln_f_output,
                                                        rhs.ln_f_output, tracker);
        bundle.lm_head_input = CompareAndLogForwardTensor(stage, update, micro, "lm_head_input", lhs.lm_head_input,
                                                          rhs.lm_head_input, tracker);
        bundle.logits = CompareAndLogForwardTensor(stage, update, micro, "logits", lhs.logits, rhs.logits, tracker);

        if (bundle.lm_head_input.count_bitwise_mismatch > 0) {
            LOG(INFO) << "LM_HEAD_INPUT_DIFFERENT stage=" << stage
                      << " update=" << update
                      << " micro=" << micro
                      << " max_abs=" << bundle.lm_head_input.max_abs
                      << " count_bitwise_mismatch=" << bundle.lm_head_input.count_bitwise_mismatch;
            CompareBlockOutputsIfNeeded(stage, update, micro, lhs, rhs, tracker);
        } else if (bundle.logits.count_bitwise_mismatch > 0) {
            LOG(INFO) << "LM_HEAD_INTERNAL_OR_OUTPUT_DIFFERENCE stage=" << stage
                      << " update=" << update
                      << " micro=" << micro
                      << " logits_max_abs=" << bundle.logits.max_abs;
        }
        return bundle;
    }

    std::vector<std::string>
    SortedParameterNames(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {
        std::vector<std::string> names;
        names.reserve(state_dict.size());
        for (const auto &[name, _] : state_dict) {
            names.push_back(name);
        }
        std::sort(names.begin(), names.end());
        return names;
    }

    void LogParameterDiffSummary(const std::string &stage, const std::vector<ParameterDiffRecord> &records) {
        std::string first_grad_diff = "none";
        std::string first_param_diff = "none";
        for (const auto &record : records) {
            if (first_grad_diff == "none"
                && (record.grad_presence_mismatch
                    || (record.has_grad_metrics
                        && (record.grad_metrics.max_abs > 0.0 || record.grad_metrics.count_bitwise_mismatch > 0)))) {
                first_grad_diff = record.name;
            }
            if (first_param_diff == "none"
                && (record.param_metrics.max_abs > 0.0 || record.param_metrics.count_bitwise_mismatch > 0)) {
                first_param_diff = record.name;
            }
        }
        LOG(INFO) << "PARAM_STAGE_SUMMARY stage=" << stage
                  << " first_gradient_diff_param=" << first_grad_diff
                  << " first_parameter_diff_param=" << first_param_diff;

        std::vector<const ParameterDiffRecord *> grad_sorted;
        std::vector<const ParameterDiffRecord *> param_sorted;
        grad_sorted.reserve(records.size());
        param_sorted.reserve(records.size());
        for (const auto &record : records) {
            grad_sorted.push_back(&record);
            param_sorted.push_back(&record);
        }
        std::sort(grad_sorted.begin(), grad_sorted.end(), [](const auto *lhs, const auto *rhs) {
            const double lhs_value = lhs->grad_presence_mismatch
                                         ? std::numeric_limits<double>::infinity()
                                         : (lhs->has_grad_metrics ? lhs->grad_metrics.max_abs : 0.0);
            const double rhs_value = rhs->grad_presence_mismatch
                                         ? std::numeric_limits<double>::infinity()
                                         : (rhs->has_grad_metrics ? rhs->grad_metrics.max_abs : 0.0);
            if (lhs_value != rhs_value) {
                return lhs_value > rhs_value;
            }
            return lhs->name < rhs->name;
        });
        std::sort(param_sorted.begin(), param_sorted.end(), [](const auto *lhs, const auto *rhs) {
            if (lhs->param_metrics.max_abs != rhs->param_metrics.max_abs) {
                return lhs->param_metrics.max_abs > rhs->param_metrics.max_abs;
            }
            return lhs->name < rhs->name;
        });

        const size_t limit = std::min<size_t>(10, records.size());
        for (size_t rank = 0; rank < limit; ++rank) {
            const auto *record = grad_sorted[rank];
            LOG(INFO) << "PARAM_GRAD_TOP stage=" << stage
                      << " rank=" << rank
                      << " name=" << record->name
                      << " grad_present_lhs=" << record->grad_present_lhs
                      << " grad_present_rhs=" << record->grad_present_rhs
                      << " grad_max_abs=" << (record->has_grad_metrics ? record->grad_metrics.max_abs : 0.0)
                      << " grad_mean_abs=" << (record->has_grad_metrics ? record->grad_metrics.mean_abs : 0.0)
                      << " grad_rmse=" << (record->has_grad_metrics ? record->grad_metrics.rmse : 0.0)
                      << " grad_first_mismatch="
                      << (record->has_grad_metrics
                              ? OptionalIndexString(record->grad_metrics.has_bitwise_mismatch,
                                                    record->grad_metrics.first_mismatch)
                              : std::string("none"));
        }
        for (size_t rank = 0; rank < limit; ++rank) {
            const auto *record = param_sorted[rank];
            LOG(INFO) << "PARAM_VALUE_TOP stage=" << stage
                      << " rank=" << rank
                      << " name=" << record->name
                      << " param_max_abs=" << record->param_metrics.max_abs
                      << " param_mean_abs=" << record->param_metrics.mean_abs
                      << " param_rmse=" << record->param_metrics.rmse
                      << " param_first_mismatch="
                      << OptionalIndexString(record->param_metrics.has_bitwise_mismatch,
                                             record->param_metrics.first_mismatch);
        }
    }

    std::vector<ParameterDiffRecord> CompareAndLogAllParameters(const std::string &stage, int update, int micro,
                                                                GPT2 &lhs, GPT2 &rhs,
                                                                FirstDivergenceTracker &tracker) {
        const auto lhs_state = lhs.StateDict();
        const auto rhs_state = rhs.StateDict();
        const auto names = SortedParameterNames(lhs_state);
        CHECK_EQ(lhs_state.size(), rhs_state.size());

        std::vector<ParameterDiffRecord> records;
        records.reserve(names.size());
        for (const auto &name : names) {
            const auto lhs_it = lhs_state.find(name);
            const auto rhs_it = rhs_state.find(name);
            CHECK(lhs_it != lhs_state.end());
            CHECK(rhs_it != rhs_state.end()) << name;

            ParameterDiffRecord record;
            record.name = name;
            record.dims = DimsToString(lhs_it->second->Dims());
            record.param_metrics = CompareFloatTensors(*lhs_it->second, *rhs_it->second, kDiagnosticTolerance);
            EXPECT_FALSE(TensorHasNonFiniteDiffInput(record.param_metrics)) << stage << " " << name;
            tracker.ObserveTensor(update, micro, stage, name, record.param_metrics);

            const auto lhs_grad = lhs_it->second->grad();
            const auto rhs_grad = rhs_it->second->grad();
            record.grad_present_lhs = lhs_grad != nullptr;
            record.grad_present_rhs = rhs_grad != nullptr;
            record.grad_presence_mismatch = record.grad_present_lhs != record.grad_present_rhs;
            if (lhs_grad && rhs_grad) {
                record.has_grad_metrics = true;
                record.grad_metrics = CompareFloatTensors(*lhs_grad, *rhs_grad, kDiagnosticTolerance);
                EXPECT_FALSE(TensorHasNonFiniteDiffInput(record.grad_metrics)) << stage << " " << name << ".grad";
                tracker.ObserveTensor(update, micro, stage, name + ".grad", record.grad_metrics);
            } else if (record.grad_presence_mismatch) {
                tracker.ObserveScalar(update, micro, stage, name + ".grad_present", true, 0.0);
            }

            LOG(INFO) << "PARAM_COMPARE stage=" << stage
                      << " update=" << update
                      << " micro=" << micro
                      << " name=" << name
                      << " shape=" << record.dims
                      << " grad_present_lhs=" << record.grad_present_lhs
                      << " grad_present_rhs=" << record.grad_present_rhs
                      << " grad_max_abs=" << (record.has_grad_metrics ? record.grad_metrics.max_abs : 0.0)
                      << " grad_mean_abs=" << (record.has_grad_metrics ? record.grad_metrics.mean_abs : 0.0)
                      << " grad_rmse=" << (record.has_grad_metrics ? record.grad_metrics.rmse : 0.0)
                      << " grad_first_mismatch="
                      << (record.has_grad_metrics
                              ? OptionalIndexString(record.grad_metrics.has_bitwise_mismatch,
                                                    record.grad_metrics.first_mismatch)
                              : std::string("none"))
                      << " grad_count_bitwise_mismatch="
                      << (record.has_grad_metrics ? record.grad_metrics.count_bitwise_mismatch : 0)
                      << " grad_count_gt_tolerance="
                      << (record.has_grad_metrics ? record.grad_metrics.count_gt_tolerance : 0)
                      << " param_max_abs=" << record.param_metrics.max_abs
                      << " param_mean_abs=" << record.param_metrics.mean_abs
                      << " param_rmse=" << record.param_metrics.rmse
                      << " param_first_mismatch="
                      << OptionalIndexString(record.param_metrics.has_bitwise_mismatch, record.param_metrics.first_mismatch)
                      << " param_count_bitwise_mismatch=" << record.param_metrics.count_bitwise_mismatch
                      << " param_count_gt_tolerance=" << record.param_metrics.count_gt_tolerance;
            records.push_back(std::move(record));
        }
        LogParameterDiffSummary(stage, records);
        return records;
    }

    void LogWorstLogitsRow(const std::string &stage, const TensorComparisonMetrics &metrics) {
        if (metrics.row_metrics.empty()) {
            return;
        }
        const auto max_row = BestRowByMaxAbs(metrics);
        LOG(INFO) << "WORST_LOGITS_ROW stage=" << stage
                  << " batch=" << max_row.batch
                  << " sequence=" << max_row.sequence
                  << " logits_max_abs=" << max_row.max_abs
                  << " logits_mean_abs=" << max_row.mean_abs
                  << " logits_rmse=" << max_row.rmse
                  << " logits_count_gt_tolerance=" << max_row.count_gt_tolerance;
    }


    struct BackwardContributionSnapshot {
        std::string event;
        std::string source;
        LogitsBinary tensor;
    };

    class ScopedBackwardDiagnosticsObserver final {
    public:
        explicit ScopedBackwardDiagnosticsObserver(autograd::BackwardDiagnosticsObserver *observer) {
            previous_ = autograd::GetBackwardDiagnosticsObserver();
            autograd::SetBackwardDiagnosticsObserver(observer);
        }

        ~ScopedBackwardDiagnosticsObserver() {
            autograd::SetBackwardDiagnosticsObserver(previous_);
        }

    private:
        autograd::BackwardDiagnosticsObserver *previous_ = nullptr;
    };

    class SharedWeightBackwardObserver final : public autograd::BackwardDiagnosticsObserver {
    public:
        SharedWeightBackwardObserver(int run_id, const Tensor &shared_weight, const Tensor &shared_grad)
            : run_id_(run_id), shared_weight_data_(shared_weight.DataPtr()), shared_grad_data_(shared_grad.DataPtr()) {}

        void OnBackwardStart(const Tensor &root) override {
            (void)root;
            CHECK(shared_grad_tensor_ != nullptr);
            Capture("backward_start_shared_grad", "shared_grad", *shared_grad_tensor_);
        }

        void OnContributionProduced(const std::string &source, const Tensor *owner, const Tensor &contribution) override {
            if (owner == nullptr || owner->DataPtr() != shared_weight_data_) {
                return;
            }
            Capture(source + "_contribution_produced", source, contribution);
        }

        void OnAccumulateBefore(const autograd::BackwardContributionInfo &info, const Tensor &grad_buffer,
                                const Tensor &contribution) override {
            (void)contribution;
            if (grad_buffer.DataPtr() != shared_grad_data_) {
                return;
            }
            Capture("accumulate_before_" + info.source, info.source, grad_buffer);
        }

        void OnAccumulateAfter(const autograd::BackwardContributionInfo &info, const Tensor &grad_buffer,
                               const Tensor &contribution) override {
            (void)contribution;
            if (grad_buffer.DataPtr() != shared_grad_data_) {
                return;
            }
            Capture("accumulate_after_" + info.source, info.source, grad_buffer);
        }

        void OnBackwardEnd(const Tensor &root) override {
            (void)root;
            CHECK(shared_grad_tensor_ != nullptr);
            Capture("backward_end_shared_grad", "shared_grad", *shared_grad_tensor_);
        }

        void SetSharedGradTensor(const Tensor &shared_grad) {
            shared_grad_tensor_ = &shared_grad;
        }

        const std::vector<BackwardContributionSnapshot> &snapshots() const { return snapshots_; }

    private:
        void Capture(const std::string &event, const std::string &source, const Tensor &tensor) {
            auto payload = CopyTensorToLogitsBinary(const_cast<Tensor &>(tensor));
            const size_t nonzero_count = std::count_if(payload.data.begin(), payload.data.end(),
                                                       [](float value) { return value != 0.0f; });
            LOG(INFO) << "BACKWARD_CONTRIB_SNAPSHOT run=" << run_id_
                      << " event=" << event
                      << " source=" << source
                      << " dims=" << DimsToString(payload.dims)
                      << " elements=" << payload.data.size()
                      << " nonzero_count=" << nonzero_count;
            snapshots_.push_back({.event = event, .source = source, .tensor = std::move(payload)});
        }

        int run_id_ = 0;
        const void *shared_weight_data_ = nullptr;
        const void *shared_grad_data_ = nullptr;
        const Tensor *shared_grad_tensor_ = nullptr;
        std::vector<BackwardContributionSnapshot> snapshots_;
    };

    std::shared_ptr<Tensor> SharedLmHeadWeight(PairedTrainingRun &run) {
        auto state_dict = run.model->StateDict();
        const auto it = state_dict.find("lm_head.weight");
        CHECK(it != state_dict.end());
        return it->second;
    }

    std::vector<BackwardContributionSnapshot> RunBackwardWithSharedWeightObserver(PairedTrainingRun &run,
                                                                                  Tensor &loss) {
        auto shared_weight = SharedLmHeadWeight(run);
        auto shared_grad = shared_weight->grad();
        CHECK(shared_grad != nullptr) << "shared lm_head.weight grad must exist after optimizer->ZeroGrad()";
        SharedWeightBackwardObserver observer(run.run_id, *shared_weight, *shared_grad);
        observer.SetSharedGradTensor(*shared_grad);
        {
            ScopedBackwardDiagnosticsObserver scoped(&observer);
            loss.Backward();
        }
        return observer.snapshots();
    }

    TensorComparisonMetrics CompareLogitsPayloads(const LogitsBinary &lhs, const LogitsBinary &rhs,
                                                  float tolerance) {
        CHECK(lhs.dims == rhs.dims) << "lhs dims=" << DimsToString(lhs.dims)
                                    << " rhs dims=" << DimsToString(rhs.dims);
        CHECK_EQ(lhs.data.size(), rhs.data.size());
        return CompareFloatData(lhs.dims, lhs.data.data(), rhs.data.data(), lhs.data.size(), tolerance);
    }

    void CompareBackwardContributionSnapshots(const std::string &stage,
                                              const std::vector<BackwardContributionSnapshot> &lhs,
                                              const std::vector<BackwardContributionSnapshot> &rhs,
                                              FirstDivergenceTracker &tracker, int update, int micro) {
        LOG(INFO) << "BACKWARD_CONTRIB_COMPARE stage=" << stage
                  << " lhs_events=" << lhs.size()
                  << " rhs_events=" << rhs.size();
        ASSERT_EQ(lhs.size(), rhs.size()) << stage;
        for (size_t i = 0; i < lhs.size(); ++i) {
            EXPECT_EQ(lhs[i].event, rhs[i].event) << stage << " event index " << i;
            EXPECT_EQ(lhs[i].source, rhs[i].source) << stage << " event index " << i;
            const auto metrics = CompareLogitsPayloads(lhs[i].tensor, rhs[i].tensor, kDiagnosticTolerance);
            EXPECT_FALSE(TensorHasNonFiniteDiffInput(metrics)) << stage << " " << lhs[i].event;
            const std::string label = stage + "." + std::to_string(i) + "." + lhs[i].event;
            LogTensorDiffComparison(label, metrics, kDiagnosticTolerance);
            tracker.ObserveTensor(update, micro, stage, lhs[i].event, metrics);
            LOG(INFO) << "BACKWARD_CONTRIB_EVENT_COMPARE stage=" << stage
                      << " index=" << i
                      << " event=" << lhs[i].event
                      << " source=" << lhs[i].source
                      << " bitwise_equal=" << (metrics.count_bitwise_mismatch == 0)
                      << " max_abs=" << metrics.max_abs
                      << " mean_abs=" << metrics.mean_abs
                      << " rmse=" << metrics.rmse
                      << " first_mismatch=" << OptionalIndexString(metrics.has_bitwise_mismatch, metrics.first_mismatch)
                      << " lhs_nonzero_count=" << metrics.lhs_nonzero_count
                      << " rhs_nonzero_count=" << metrics.rhs_nonzero_count;
        }
    }

    std::shared_ptr<Tensor> MakeDeterministicFloatTensor(const std::vector<int64_t> &dims, float scale,
                                                         int offset) {
        auto tensor = std::make_shared<Tensor>(dims, DataType::kFLOAT32, Device(DeviceType::kCPU, 0));
        auto *data = static_cast<float *>(tensor->DataPtr());
        for (size_t i = 0; i < tensor->NumElements(); ++i) {
            const int value = static_cast<int>((i * 1103515245ULL + static_cast<size_t>(offset) * 12345ULL) % 4096ULL);
            data[i] = (static_cast<float>(value) - 2048.0f) * scale;
        }
        return tensor;
    }

    std::shared_ptr<Tensor> MakeDeterministicTokenTensor(const std::vector<int64_t> &dims, int64_t vocab_size) {
        auto tensor = std::make_shared<Tensor>(dims, DataType::kINT64, Device(DeviceType::kCPU, 0));
        auto *data = static_cast<int64_t *>(tensor->DataPtr());
        for (size_t i = 0; i < tensor->NumElements(); ++i) {
            data[i] = static_cast<int64_t>((i * 17 + (i % 5) * 3) % 32);
            CHECK_LT(data[i], vocab_size);
        }
        return tensor;
    }

    std::shared_ptr<Tensor> ToDiagnosticCudaTensor(const std::shared_ptr<Tensor> &cpu_tensor) {
        return std::make_shared<Tensor>(cpu_tensor->To(device));
    }

    void LogRepeatedTensorResults(const std::string &label, const std::vector<LogitsBinary> &results) {
        ASSERT_GE(results.size(), 2);
        int first_diff_iter = -1;
        double max_abs = 0.0;
        for (size_t i = 1; i < results.size(); ++i) {
            const auto metrics = CompareLogitsPayloads(results[0], results[i], 0.0f);
            EXPECT_FALSE(TensorHasNonFiniteDiffInput(metrics)) << label << " iteration " << i;
            if (metrics.count_bitwise_mismatch != 0 && first_diff_iter < 0) {
                first_diff_iter = static_cast<int>(i);
            }
            max_abs = std::max(max_abs, metrics.max_abs);
            LogTensorDiffComparison(label + ".iter" + std::to_string(i), metrics, 0.0f);
        }
        LOG(INFO) << "DETERMINISM_RESULT label=" << label
                  << " runs=" << results.size()
                  << " bitwise_stable=" << (first_diff_iter < 0)
                  << " first_diff_iter=" << first_diff_iter
                  << " max_abs=" << max_abs;
    }

    void LogMatrixRowDiffSummary(const std::string &label, const LogitsBinary &lhs, const LogitsBinary &rhs,
                                 float tolerance) {
        if (lhs.dims.size() != 2 || lhs.dims != rhs.dims) {
            return;
        }
        const size_t rows = static_cast<size_t>(lhs.dims[0]);
        const size_t cols = static_cast<size_t>(lhs.dims[1]);
        size_t best_row = 0;
        double best_max_abs = -1.0;
        double best_mean_abs = 0.0;
        double best_rmse = 0.0;
        size_t best_count_gt = 0;
        bool has_first_mismatch = false;
        size_t first_mismatch_row = 0;
        size_t first_mismatch_col = 0;
        for (size_t row = 0; row < rows; ++row) {
            double sum_abs = 0.0;
            double sum_sq = 0.0;
            double row_max_abs = 0.0;
            size_t row_count_gt = 0;
            for (size_t col = 0; col < cols; ++col) {
                const size_t index = row * cols + col;
                const float lhs_value = lhs.data[index];
                const float rhs_value = rhs.data[index];
                const bool nonfinite = !std::isfinite(lhs_value) || !std::isfinite(rhs_value);
                const double diff = nonfinite ? std::numeric_limits<double>::infinity()
                                              : std::abs(static_cast<double>(lhs_value) - static_cast<double>(rhs_value));
                if (!has_first_mismatch && std::memcmp(&lhs_value, &rhs_value, sizeof(float)) != 0) {
                    has_first_mismatch = true;
                    first_mismatch_row = row;
                    first_mismatch_col = col;
                }
                if (nonfinite || diff > tolerance) {
                    ++row_count_gt;
                }
                sum_abs += diff;
                sum_sq += diff * diff;
                row_max_abs = std::max(row_max_abs, diff);
            }
            if (row_max_abs > best_max_abs || (row_max_abs == best_max_abs && row_count_gt > best_count_gt)) {
                best_row = row;
                best_max_abs = row_max_abs;
                best_mean_abs = sum_abs / static_cast<double>(cols);
                best_rmse = std::sqrt(sum_sq / static_cast<double>(cols));
                best_count_gt = row_count_gt;
            }
        }
        LOG(INFO) << "MATRIX_ROW_AUTO label=" << label
                  << " kind=max_abs row=" << best_row
                  << " max_abs=" << best_max_abs
                  << " mean_abs=" << best_mean_abs
                  << " rmse=" << best_rmse
                  << " count_gt_tolerance=" << best_count_gt;
        LOG(INFO) << "MATRIX_ROW_AUTO label=" << label
                  << " kind=first_mismatch row="
                  << (has_first_mismatch ? std::to_string(first_mismatch_row) : std::string("none"))
                  << " col=" << (has_first_mismatch ? std::to_string(first_mismatch_col) : std::string("none"));
    }

    struct RepeatedTensorRunSummary {
        int first_diff_iter = -1;
        double max_abs = 0.0;
        size_t max_bitwise_mismatch_count = 0;
    };
    template <typename RunOnceT>
    RepeatedTensorRunSummary LogRepeatedTensorResultsIncremental(const std::string &label, int runs,
                                                                 RunOnceT run_once) {
        CHECK_GE(runs, 2);
        auto reference = run_once();
        RepeatedTensorRunSummary summary;
        for (int i = 1; i < runs; ++i) {
            auto current = run_once();
            const auto metrics = CompareLogitsPayloads(reference, current, 0.0f);
            EXPECT_FALSE(TensorHasNonFiniteDiffInput(metrics)) << label << " iteration " << i;
            if (metrics.count_bitwise_mismatch != 0 && summary.first_diff_iter < 0) {
                summary.first_diff_iter = i;
            }
            summary.max_abs = std::max(summary.max_abs, metrics.max_abs);
            summary.max_bitwise_mismatch_count = std::max(summary.max_bitwise_mismatch_count,
                                                          metrics.count_bitwise_mismatch);
            const std::string iter_label = label + ".iter" + std::to_string(i);
            LogTensorDiffComparison(iter_label, metrics, 0.0f);
            LogMatrixRowDiffSummary(iter_label, reference, current, 0.0f);
        }
        LOG(INFO) << "DETERMINISM_RESULT label=" << label
                  << " runs=" << runs
                  << " bitwise_stable=" << (summary.first_diff_iter < 0)
                  << " first_diff_iter=" << summary.first_diff_iter
                  << " max_abs=" << summary.max_abs
                  << " max_bitwise_mismatch_count=" << summary.max_bitwise_mismatch_count;
        return summary;
    }

    void ReleaseFixtureTrainingStateForStandaloneKernelTest() {
        logits.reset();
        optimizer.reset();
        loss_fn.reset();
        model.reset();
        tokenizer.reset();
        train_loader.reset();
    }
    LogitsBinary RunLinearWeightGradientOnce(const LogitsBinary &input_cpu_payload,
                                             const LogitsBinary &weight_cpu_payload,
                                             const LogitsBinary &grad_output_cpu_payload,
                                             const LogitsBinary &initial_grad_cpu_payload) {
        auto input_cpu = std::make_shared<Tensor>(input_cpu_payload.dims, DataType::kFLOAT32, Device(DeviceType::kCPU, 0));
        std::memcpy(input_cpu->DataPtr(), input_cpu_payload.data.data(), input_cpu_payload.data.size() * sizeof(float));
        auto weight_cpu = std::make_shared<Tensor>(weight_cpu_payload.dims, DataType::kFLOAT32, Device(DeviceType::kCPU, 0));
        std::memcpy(weight_cpu->DataPtr(), weight_cpu_payload.data.data(), weight_cpu_payload.data.size() * sizeof(float));
        auto grad_output_cpu = std::make_shared<Tensor>(grad_output_cpu_payload.dims, DataType::kFLOAT32,
                                                        Device(DeviceType::kCPU, 0));
        std::memcpy(grad_output_cpu->DataPtr(), grad_output_cpu_payload.data.data(),
                    grad_output_cpu_payload.data.size() * sizeof(float));
        auto initial_grad_cpu = std::make_shared<Tensor>(initial_grad_cpu_payload.dims, DataType::kFLOAT32,
                                                         Device(DeviceType::kCPU, 0));
        std::memcpy(initial_grad_cpu->DataPtr(), initial_grad_cpu_payload.data.data(),
                    initial_grad_cpu_payload.data.size() * sizeof(float));

        auto input = ToDiagnosticCudaTensor(input_cpu);
        auto weight = ToDiagnosticCudaTensor(weight_cpu);
        auto grad_output = ToDiagnosticCudaTensor(grad_output_cpu);
        auto grad_buffer = ToDiagnosticCudaTensor(initial_grad_cpu);
        const auto &linear_backward = Dispatcher::Instance().GetKernel({DeviceType::kCUDA, "LinearBackward"});
        auto gradients = linear_backward.Call<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>>(
            input, weight, true, 50257, grad_output, false);
        auto grad_weight = std::get<1>(gradients);
        const auto &accumulate_grad = Dispatcher::Instance().GetKernel({DeviceType::kCUDA, "AccumulateGrad"});
        accumulate_grad.Call<void>(grad_weight, 1.0f, grad_buffer);
        return CopyTensorToLogitsBinary(*grad_buffer);
    }

    std::unordered_map<int64_t, size_t> CountTokens(const std::vector<int64_t> &input_values) {
        std::unordered_map<int64_t, size_t> counts;
        for (const int64_t token : input_values) {
            ++counts[token];
        }
        return counts;
    }

    std::vector<std::pair<int64_t, size_t>> SortedTokenCounts(const std::vector<int64_t> &input_values) {
        auto counts = CountTokens(input_values);
        std::vector<std::pair<int64_t, size_t>> sorted(counts.begin(), counts.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.first < rhs.first;
        });
        return sorted;
    }

    void LogTokenCounts(const std::string &label, const std::vector<int64_t> &input_values) {
        const auto sorted = SortedTokenCounts(input_values);
        size_t repeated_token_rows = 0;
        size_t repeated_index_instances = 0;
        size_t max_repeat = 0;
        int64_t max_repeat_token = -1;
        for (const auto &[token, count] : sorted) {
            if (count > 1) {
                ++repeated_token_rows;
                repeated_index_instances += count;
            }
            if (count > max_repeat) {
                max_repeat = count;
                max_repeat_token = token;
            }
        }
        LOG(INFO) << "EMBEDDING_TOKEN_COUNTS_SUMMARY label=" << label
                  << " total_tokens=" << input_values.size()
                  << " unique_token_rows=" << sorted.size()
                  << " repeated_token_rows=" << repeated_token_rows
                  << " repeated_index_instances=" << repeated_index_instances
                  << " max_repeat_token=" << max_repeat_token
                  << " max_repeat=" << max_repeat;
        for (const auto &[token, count] : sorted) {
            LOG(INFO) << "EMBEDDING_TOKEN_COUNT label=" << label
                      << " token=" << token
                      << " count=" << count;
        }
    }

    std::vector<int64_t> MakeUniqueTokenValues(size_t num_tokens, int64_t vocab_size) {
        CHECK_LE(num_tokens + 1024, static_cast<size_t>(vocab_size));
        std::vector<int64_t> values(num_tokens);
        for (size_t i = 0; i < num_tokens; ++i) {
            values[i] = static_cast<int64_t>(1024 + i);
        }
        return values;
    }

    std::vector<int64_t> MakeRepeatedTokenValues(size_t num_tokens, int64_t vocab_size) {
        CHECK_GE(vocab_size, 32);
        std::vector<int64_t> values(num_tokens);
        for (size_t i = 0; i < num_tokens; ++i) {
            values[i] = static_cast<int64_t>((i * 17 + (i % 5) * 3) % 32);
        }
        return values;
    }

    std::vector<int64_t> MakeIdenticalTokenValues(size_t num_tokens, int64_t token_id) {
        return std::vector<int64_t>(num_tokens, token_id);
    }

    LogitsBinary ComputeEmbeddingBackwardCpuReference(const std::vector<int64_t> &input_values,
                                                      const LogitsBinary &grad_output_cpu_payload,
                                                      const std::vector<int64_t> &weight_dims) {
        CHECK_EQ(weight_dims.size(), 2);
        const int64_t num_embeddings = weight_dims[0];
        const int64_t embedding_dim = weight_dims[1];
        CHECK_EQ(grad_output_cpu_payload.data.size(), input_values.size() * static_cast<size_t>(embedding_dim));
        LogitsBinary result{.dims = weight_dims,
                            .data = std::vector<float>(static_cast<size_t>(num_embeddings * embedding_dim), 0.0f)};
        for (size_t i = 0; i < input_values.size(); ++i) {
            const int64_t token = input_values[i];
            if (token < 0 || token >= num_embeddings) {
                continue;
            }
            for (int64_t dim = 0; dim < embedding_dim; ++dim) {
                result.data[static_cast<size_t>(token * embedding_dim + dim)]
                    += grad_output_cpu_payload.data[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(dim)];
            }
        }
        return result;
    }

    void LogTokenGroupErrorSummary(const std::string &label, const LogitsBinary &lhs, const LogitsBinary &rhs,
                                   const std::vector<std::pair<int64_t, size_t>> &token_counts) {
        CHECK(lhs.dims == rhs.dims);
        CHECK_EQ(lhs.dims.size(), 2);
        const size_t embedding_dim = static_cast<size_t>(lhs.dims[1]);
        struct GroupStats {
            size_t rows = 0;
            size_t elements = 0;
            size_t count_bitwise_mismatch = 0;
            double max_abs = 0.0;
            double sum_abs = 0.0;
            double sum_sq = 0.0;
        };
        GroupStats repeated;
        GroupStats unique;
        auto accumulate_row = [&](GroupStats &stats, int64_t token) {
            CHECK_GE(token, 0);
            CHECK_LT(token, lhs.dims[0]);
            ++stats.rows;
            for (size_t dim = 0; dim < embedding_dim; ++dim) {
                const size_t index = static_cast<size_t>(token) * embedding_dim + dim;
                const float lhs_value = lhs.data[index];
                const float rhs_value = rhs.data[index];
                const double diff = std::abs(static_cast<double>(lhs_value) - static_cast<double>(rhs_value));
                stats.count_bitwise_mismatch += std::memcmp(&lhs_value, &rhs_value, sizeof(float)) != 0 ? 1 : 0;
                stats.max_abs = std::max(stats.max_abs, diff);
                stats.sum_abs += diff;
                stats.sum_sq += diff * diff;
                ++stats.elements;
            }
        };
        for (const auto &[token, count] : token_counts) {
            if (token < 0 || token >= lhs.dims[0]) {
                continue;
            }
            if (count > 1) {
                accumulate_row(repeated, token);
            } else {
                accumulate_row(unique, token);
            }
        }
        auto log_group = [&](const char *kind, const GroupStats &stats) {
            LOG(INFO) << "EMBEDDING_TOKEN_GROUP_COMPARE label=" << label
                      << " kind=" << kind
                      << " rows=" << stats.rows
                      << " elements=" << stats.elements
                      << " bitwise_mismatch_count=" << stats.count_bitwise_mismatch
                      << " max_abs=" << stats.max_abs
                      << " mean_abs=" << (stats.elements == 0 ? 0.0 : stats.sum_abs / static_cast<double>(stats.elements))
                      << " rmse=" << (stats.elements == 0 ? 0.0 : std::sqrt(stats.sum_sq / static_cast<double>(stats.elements)));
        };
        log_group("repeated", repeated);
        log_group("unique", unique);
    }

    double MeasureEmbeddingBackwardAverageMs(const std::vector<int64_t> &input_dims,
                                             const std::vector<int64_t> &input_values,
                                             const LogitsBinary &grad_output_cpu_payload, int warmup_runs,
                                             int timed_runs) {
#ifdef USE_CUDA
        auto input_cpu = std::make_shared<Tensor>(input_dims, DataType::kINT64, Device(DeviceType::kCPU, 0));
        std::memcpy(input_cpu->DataPtr(), input_values.data(), input_values.size() * sizeof(int64_t));
        auto grad_output_cpu = std::make_shared<Tensor>(grad_output_cpu_payload.dims, DataType::kFLOAT32,
                                                        Device(DeviceType::kCPU, 0));
        std::memcpy(grad_output_cpu->DataPtr(), grad_output_cpu_payload.data.data(),
                    grad_output_cpu_payload.data.size() * sizeof(float));
        auto input = ToDiagnosticCudaTensor(input_cpu);
        auto grad_output = ToDiagnosticCudaTensor(grad_output_cpu);
        const auto &embedding_backward = Dispatcher::Instance().GetKernel({DeviceType::kCUDA, "EmbeddingBackward"});
        for (int i = 0; i < warmup_runs; ++i) {
            auto grad_weight = embedding_backward.Call<std::shared_ptr<Tensor>>(input, std::vector<int64_t>{50257, 768},
                                                                                grad_output);
            (void)grad_weight;
        }
        cudaDeviceSynchronize();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < timed_runs; ++i) {
            auto grad_weight = embedding_backward.Call<std::shared_ptr<Tensor>>(input, std::vector<int64_t>{50257, 768},
                                                                                grad_output);
            (void)grad_weight;
        }
        cudaDeviceSynchronize();
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return elapsed_ms / static_cast<double>(timed_runs);
#else
        (void)input_dims;
        (void)input_values;
        (void)grad_output_cpu_payload;
        (void)warmup_runs;
        (void)timed_runs;
        return -1.0;
#endif
    }

    RepeatedTensorRunSummary LogEmbeddingCaseDiagnostics(const std::string &label,
                                                         const std::vector<int64_t> &input_dims,
                                                         const std::vector<int64_t> &input_values,
                                                         const LogitsBinary &grad_output_payload, int runs) {
        constexpr int64_t kVocabSize = 50257;
        constexpr int64_t kHiddenSize = 768;
        const std::vector<int64_t> weight_dims{kVocabSize, kHiddenSize};
        const auto token_counts = SortedTokenCounts(input_values);
        LogTokenCounts(label, input_values);
        const auto cpu_reference = ComputeEmbeddingBackwardCpuReference(input_values, grad_output_payload, weight_dims);
        const double average_ms = MeasureEmbeddingBackwardAverageMs(input_dims, input_values, grad_output_payload, 5, 30);
        LOG(INFO) << "EMBEDDING_BACKWARD_PERF label=" << label
                  << " average_ms=" << average_ms
                  << " warmup_runs=5 timed_runs=30"
                  << " shape_tokens=" << input_values.size()
                  << " embedding_dim=" << kHiddenSize
                  << " vocab_size=" << kVocabSize;

        CHECK_GE(runs, 2);
        auto first_cuda = RunEmbeddingWeightGradientOnce(input_dims, input_values, grad_output_payload);
        auto first_cpu_metrics = CompareLogitsPayloads(cpu_reference, first_cuda, 0.0f);
        LogTensorDiffComparison(label + ".cuda_vs_cpu.iter0", first_cpu_metrics, 0.0f);
        LogMatrixRowDiffSummary(label + ".cuda_vs_cpu.iter0", cpu_reference, first_cuda, 0.0f);
        LogTokenGroupErrorSummary(label + ".cuda_vs_cpu.iter0", cpu_reference, first_cuda, token_counts);

        RepeatedTensorRunSummary summary;
        for (int i = 1; i < runs; ++i) {
            auto current_cuda = RunEmbeddingWeightGradientOnce(input_dims, input_values, grad_output_payload);
            const auto cuda_vs_cuda = CompareLogitsPayloads(first_cuda, current_cuda, 0.0f);
            EXPECT_FALSE(TensorHasNonFiniteDiffInput(cuda_vs_cuda)) << label << " cuda_vs_cuda iteration " << i;
            if (cuda_vs_cuda.count_bitwise_mismatch != 0 && summary.first_diff_iter < 0) {
                summary.first_diff_iter = i;
            }
            summary.max_abs = std::max(summary.max_abs, cuda_vs_cuda.max_abs);
            summary.max_bitwise_mismatch_count = std::max(summary.max_bitwise_mismatch_count,
                                                          cuda_vs_cuda.count_bitwise_mismatch);
            const std::string cuda_label = label + ".cuda_vs_cuda.iter" + std::to_string(i);
            LogTensorDiffComparison(cuda_label, cuda_vs_cuda, 0.0f);
            LogMatrixRowDiffSummary(cuda_label, first_cuda, current_cuda, 0.0f);
            LogTokenGroupErrorSummary(cuda_label, first_cuda, current_cuda, token_counts);

            const auto cuda_vs_cpu = CompareLogitsPayloads(cpu_reference, current_cuda, 0.0f);
            EXPECT_FALSE(TensorHasNonFiniteDiffInput(cuda_vs_cpu)) << label << " cuda_vs_cpu iteration " << i;
            const std::string cpu_label = label + ".cuda_vs_cpu.iter" + std::to_string(i);
            LogTensorDiffComparison(cpu_label, cuda_vs_cpu, 0.0f);
            LogMatrixRowDiffSummary(cpu_label, cpu_reference, current_cuda, 0.0f);
            LogTokenGroupErrorSummary(cpu_label, cpu_reference, current_cuda, token_counts);
        }
        LOG(INFO) << "DETERMINISM_RESULT label=" << label
                  << " runs=" << runs
                  << " bitwise_stable=" << (summary.first_diff_iter < 0)
                  << " first_diff_iter=" << summary.first_diff_iter
                  << " max_abs=" << summary.max_abs
                  << " max_bitwise_mismatch_count=" << summary.max_bitwise_mismatch_count;
        return summary;
    }
    LogitsBinary RunEmbeddingWeightGradientOnce(const std::vector<int64_t> &input_dims,
                                                const std::vector<int64_t> &input_values,
                                                const LogitsBinary &grad_output_cpu_payload) {
        auto input_cpu = std::make_shared<Tensor>(input_dims, DataType::kINT64, Device(DeviceType::kCPU, 0));
        std::memcpy(input_cpu->DataPtr(), input_values.data(), input_values.size() * sizeof(int64_t));
        auto grad_output_cpu = std::make_shared<Tensor>(grad_output_cpu_payload.dims, DataType::kFLOAT32,
                                                        Device(DeviceType::kCPU, 0));
        std::memcpy(grad_output_cpu->DataPtr(), grad_output_cpu_payload.data.data(),
                    grad_output_cpu_payload.data.size() * sizeof(float));
        auto input = ToDiagnosticCudaTensor(input_cpu);
        auto grad_output = ToDiagnosticCudaTensor(grad_output_cpu);
        const auto &embedding_backward = Dispatcher::Instance().GetKernel({DeviceType::kCUDA, "EmbeddingBackward"});
        auto grad_weight = embedding_backward.Call<std::shared_ptr<Tensor>>(input, std::vector<int64_t>{50257, 768},
                                                                            grad_output);
        return CopyTensorToLogitsBinary(*grad_weight);
    }

    LogitsBinary RunSharedAccumulationOnce(const LogitsBinary &initial_grad_cpu_payload,
                                           const LogitsBinary &contribution_a_cpu_payload,
                                           const LogitsBinary &contribution_b_cpu_payload,
                                           const std::string &order) {
        auto initial_grad_cpu = std::make_shared<Tensor>(initial_grad_cpu_payload.dims, DataType::kFLOAT32,
                                                         Device(DeviceType::kCPU, 0));
        std::memcpy(initial_grad_cpu->DataPtr(), initial_grad_cpu_payload.data.data(),
                    initial_grad_cpu_payload.data.size() * sizeof(float));
        auto contribution_a_cpu = std::make_shared<Tensor>(contribution_a_cpu_payload.dims, DataType::kFLOAT32,
                                                           Device(DeviceType::kCPU, 0));
        std::memcpy(contribution_a_cpu->DataPtr(), contribution_a_cpu_payload.data.data(),
                    contribution_a_cpu_payload.data.size() * sizeof(float));
        auto contribution_b_cpu = std::make_shared<Tensor>(contribution_b_cpu_payload.dims, DataType::kFLOAT32,
                                                           Device(DeviceType::kCPU, 0));
        std::memcpy(contribution_b_cpu->DataPtr(), contribution_b_cpu_payload.data.data(),
                    contribution_b_cpu_payload.data.size() * sizeof(float));
        auto grad_buffer = ToDiagnosticCudaTensor(initial_grad_cpu);
        auto contribution_a = ToDiagnosticCudaTensor(contribution_a_cpu);
        auto contribution_b = ToDiagnosticCudaTensor(contribution_b_cpu);
        const auto &accumulate_grad = Dispatcher::Instance().GetKernel({DeviceType::kCUDA, "AccumulateGrad"});
        auto add_a = [&]() { accumulate_grad.Call<void>(contribution_a, 1.0f, grad_buffer); };
        auto add_b = [&]() { accumulate_grad.Call<void>(contribution_b, 1.0f, grad_buffer); };
        if (order == "A_then_B") {
            add_a();
            add_b();
        } else if (order == "B_then_A" || order == "real_autograd_order") {
            add_b();
            add_a();
        } else {
            CHECK(false) << "Unknown accumulation order: " << order;
        }
        return CopyTensorToLogitsBinary(*grad_buffer);
    }
    void ConfigureGradAccumulation() {
        const auto tokens_per_fwdbwd = batch_size * sequence_length;
        grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
        CHECK_GT(grad_accum_steps, 0);
        CHECK_EQ(grad_accum_steps, 2) << "GPT-2 tied reference semantics currently expect two microbatches";
    }

    float RunTrainingUpdate(int update_index, bool emit_logs) {
        auto train_iter = train_loader->begin();
        optimizer->ZeroGrad();
        float lossf = 0.0f;

        for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
            auto [x, y] = *train_iter;
            ++train_iter;
            x = std::make_shared<Tensor>(x->To(device));
            y = std::make_shared<Tensor>(y->To(device));

            auto outputs = model->Forward({x, y});
            auto update_logits = outputs[0];
            CHECK(update_logits != nullptr) << "Training update logits are null";

            auto loss = loss_fn->Forward({update_logits, y})[0];
            auto loss_cpu = loss->To(Device());
            lossf += static_cast<const float *>(loss_cpu.DataPtr())[0] / grad_accum_steps;

            // Current accumulation semantics intentionally match the legacy test: the logged loss is averaged,
            // but backward uses the unscaled microbatch loss. Gradient scaling is a separate follow-up decision.
            loss->Backward();
        }

        optimizer->Step();
        if (emit_logs) {
            LOG(INFO) << "TIED_REF_UPDATE update=" << update_index << " loss=" << lossf
                      << " grad_accum_steps=" << grad_accum_steps;
        }
        return lossf;
    }

    std::shared_ptr<Tensor> RunForwardOnly(int batch_offset, bool emit_logs) {
        auto train_iter = train_loader->begin();
        for (int i = 0; i < batch_offset; ++i) {
            ++train_iter;
        }
        auto [x, y] = *train_iter;
        x = std::make_shared<Tensor>(x->To(device));
        y = std::make_shared<Tensor>(y->To(device));

        auto outputs = model->Forward({x, y});
        logits = outputs[0];
        CHECK(logits != nullptr) << "Final forward logits are null";
        if (emit_logs) {
            const auto summary = SummarizeFloatTensor(*logits, kLogitProbeIndex);
            LOG(INFO) << "TIED_REF_FINAL_FORWARD batch_offset=" << batch_offset
                      << " dims=" << summary.dims << " elements=" << summary.num_elements
                      << " logits0=" << CopyTensorToLogitsBinary(*logits).data[0]
                      << " logits385973=" << summary.sample;
        }
        return logits;
    }

    std::shared_ptr<Tensor> RunTiedReferenceTrainingAndForward(bool emit_logs) {
        ConfigureGradAccumulation();
        for (int update = 0; update < num_reference_optimizer_updates; ++update) {
            (void)RunTrainingUpdate(update, emit_logs);
        }
        return RunForwardOnly(reference_forward_batch_offset, emit_logs);
    }

    void AssertTiedWeightsAndOptimizerDedup() {
        const auto snapshot = GetWeightTyingSnapshot(*model);
        EXPECT_EQ(snapshot.wte.get(), snapshot.lm_head.get()) << "Expected tied weights to share Tensor object";
        EXPECT_EQ(snapshot.wte->DataPtr(), snapshot.lm_head->DataPtr()) << "Expected tied weights to share data";

        const auto params = model->Parameters();
        EXPECT_EQ(CountPointerOccurrences(params, snapshot.wte), 1)
            << "Shared weight should appear once in optimizer parameter list";
        EXPECT_EQ(CountDataPtrOccurrences(params, snapshot.wte->DataPtr()), 1)
            << "Shared weight data should appear once in optimizer parameter list";
    }

    std::string HashReferenceTrainingMicrobatches() {
        ConfigureGradAccumulation();
        std::vector<uint8_t> bytes;
        AppendPod(bytes, num_reference_optimizer_updates);
        AppendPod(bytes, grad_accum_steps);
        auto train_iter = train_loader->begin();
        for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
            auto [x, y] = *train_iter;
            ++train_iter;
            AppendStringRecord(bytes, "microbatch");
            AppendPod(bytes, micro_step);
            AppendTensorRecord(bytes, "input", *x);
            AppendTensorRecord(bytes, "target", *y);
        }
        return Sha256Bytes(bytes);
    }

    std::pair<std::string, std::string> HashReferenceForwardBatch() {
        auto train_iter = train_loader->begin();
        for (int i = 0; i < reference_forward_batch_offset; ++i) {
            ++train_iter;
        }
        auto [x, y] = *train_iter;
        return {HashTensorRecord("final_forward_input", *x), HashTensorRecord("final_forward_target", *y)};
    }

    ReferenceTraceMetadata CollectReferenceTraceMetadata() {
        ReferenceTraceMetadata trace;
        trace.git = ReadGitMetadata();
        trace.model_checkpoint_path = std::filesystem::weakly_canonical(llmc_filepath).string();
        trace.model_checkpoint_sha256 = Sha256File(llmc_filepath);
        trace.training_dataset_path = std::filesystem::weakly_canonical(input_bin).string();
        trace.training_dataset_sha256 = Sha256File(input_bin);
        trace.tokenizer_path = std::filesystem::weakly_canonical(tokenizer_bin).string();
        trace.tokenizer_sha256 = Sha256File(tokenizer_bin);
        trace.training_microbatches_sha256 = HashReferenceTrainingMicrobatches();
        trace.training_microbatch_count = grad_accum_steps;
        trace.training_microbatches_repeated_use_count = num_reference_optimizer_updates;
        const auto [final_input_sha256, final_target_sha256] = HashReferenceForwardBatch();
        trace.final_forward_input_sha256 = final_input_sha256;
        trace.final_forward_target_sha256 = final_target_sha256;
        return trace;
    }

    std::string BuildReferenceMetadata(const LogitsBinary &generated_logits, const std::string &output_path,
                                       const std::string &reference_sha256,
                                       const ReferenceTraceMetadata &trace) const {
        const auto snapshot = GetWeightTyingSnapshot(*model);
        const auto params = model->Parameters();
        std::ostringstream metadata;
        metadata << "git_commit=" << trace.git.commit << "\n";
        metadata << "git_branch=" << trace.git.branch << "\n";
        metadata << "working_tree_dirty=" << (trace.git.working_tree_dirty ? "true" : "false") << "\n";
        metadata << "unstaged_diff_sha256=" << trace.git.unstaged_diff_sha256 << "\n";
        metadata << "staged_diff_sha256=" << trace.git.staged_diff_sha256 << "\n";
        metadata << "git_status_porcelain_sha256=" << trace.git.status_porcelain_sha256 << "\n";
        metadata << "generated_at=" << CurrentDateTimeString() << "\n";
        metadata << "cuda_runtime_version=" << CudaRuntimeVersionString() << "\n";
        metadata << "gpu_name=" << GpuNameString(device) << "\n";
        metadata << "cublas_version=" << CublasVersionString() << "\n";
        metadata << "build_type=" << BuildTypeString() << "\n";
        metadata << "use_cuda=" << UseCudaString() << "\n";
        metadata << "model_checkpoint_path=" << trace.model_checkpoint_path << "\n";
        metadata << "model_checkpoint_sha256=" << trace.model_checkpoint_sha256 << "\n";
        metadata << "training_dataset_path=" << trace.training_dataset_path << "\n";
        metadata << "training_dataset_sha256=" << trace.training_dataset_sha256 << "\n";
        metadata << "tokenizer_path=" << trace.tokenizer_path << "\n";
        metadata << "tokenizer_sha256=" << trace.tokenizer_sha256 << "\n";
        metadata << "training_microbatches_sha256=" << trace.training_microbatches_sha256 << "\n";
        metadata << "training_microbatch_count=" << trace.training_microbatch_count << "\n";
        metadata << "training_microbatches_repeated_use_count=" << trace.training_microbatches_repeated_use_count << "\n";
        metadata << "final_forward_batch_offset=" << reference_forward_batch_offset << "\n";
        metadata << "final_forward_input_sha256=" << trace.final_forward_input_sha256 << "\n";
        metadata << "final_forward_target_sha256=" << trace.final_forward_target_sha256 << "\n";
        metadata << "batch_size=" << batch_size << "\n";
        metadata << "sequence_length=" << sequence_length << "\n";
        metadata << "vocab_size=" << generated_logits.dims.at(2) << "\n";
        metadata << "total_batch_size=" << total_batch_size << "\n";
        metadata << "grad_accum_steps=" << grad_accum_steps << "\n";
        metadata << "num_optimizer_updates=" << num_reference_optimizer_updates << "\n";
        metadata << "learning_rate=" << learning_rate << "\n";
        metadata << "optimizer=SGD\n";
        metadata << "weight_tying_same_tensor=" << (snapshot.wte.get() == snapshot.lm_head.get()) << "\n";
        metadata << "weight_tying_same_data=" << (snapshot.wte->DataPtr() == snapshot.lm_head->DataPtr()) << "\n";
        metadata << "optimizer_param_total=" << params.size() << "\n";
        metadata << "optimizer_unique_tensor_objects=" << CountUniqueTensorObjects(params) << "\n";
        metadata << "optimizer_unique_data_ptrs=" << CountUniqueDataPtrs(params) << "\n";
        metadata << "reference_format_magic=" << kReferenceMagic << "\n";
        metadata << "reference_format_version=" << kReferenceVersion << "\n";
        metadata << "reference_format_dtype=float32\n";
        metadata << "reference_shape=" << DimsToString(generated_logits.dims) << "\n";
        metadata << "reference_dtype=float32\n";
        metadata << "reference_file=" << output_path << "\n";
        metadata << "reference_file_sha256=" << reference_sha256 << "\n";
        metadata << "loss_logging=loss_sum_divided_by_grad_accum_steps\n";
        metadata << "backward_loss_scaling=unscaled_microbatch_loss\n";
        metadata << "generation_mode=candidate_generated_only\n";
        return metadata.str();
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
    std::string tied_logits_reference;
    std::string tied_logits_reference_generated;
    std::string tied_logits_reference_meta_generated;
    std::string device_flag;
    std::string model_name;
    int batch_size = 2;
    int sequence_length = 64;    
    int total_batch_size = 256;
    int num_iteration = 10;    // 迭代次数
    int num_reference_optimizer_updates = 10;
    int reference_forward_batch_offset = 0;
    int text_length = 64;    // 生成文本长度
    int freq_generate_txt = 10;
    float learning_rate = 1e-4;
};


TEST_F(GPT2TrainingTest, WeightTyingPreservedAcrossDeviceMove) {
    auto constructed_model = GPT2::FromLLMC(llmc_filepath);
    const auto constructed = GetWeightTyingSnapshot(*constructed_model);
    LogWeightTyingSnapshot("constructed_cpu", constructed);
    EXPECT_EQ(constructed.wte.get(), constructed.lm_head.get());
    EXPECT_EQ(constructed.wte->DataPtr(), constructed.lm_head->DataPtr());
    constructed_model.reset();

    const auto moved = GetWeightTyingSnapshot(*model);
    LogWeightTyingSnapshot("after_to", moved);
    EXPECT_EQ(moved.wte.get(), moved.lm_head.get()) << "weight tying object alias was broken by Module::To";
    EXPECT_EQ(moved.wte->DataPtr(), moved.lm_head->DataPtr()) << "weight tying storage/data alias was broken by Module::To";

    const auto params = model->Parameters();
    LogOptimizerAliases("after_to", params, moved);
    const auto shared_object_occurrences = CountPointerOccurrences(params, moved.wte);
    const auto shared_data_occurrences = CountDataPtrOccurrences(params, moved.wte->DataPtr());
    EXPECT_EQ(shared_object_occurrences, 1) << "shared weight appears multiple times by Tensor object in Parameters()";
    EXPECT_EQ(shared_data_occurrences, 1) << "shared weight appears multiple times by data/storage in Parameters()";

    InspectableSGD local_optimizer(params, learning_rate);
    LogOptimizerAliases("optimizer_collected", local_optimizer.params(), moved);

    auto train_iter = train_loader->begin();
    local_optimizer.ZeroGrad();
    auto [x, y] = *train_iter;
    x = std::make_shared<Tensor>(x->To(device));
    y = std::make_shared<Tensor>(y->To(device));

    auto outputs = model->Forward({x, y});
    auto local_logits = outputs[0];
    ASSERT_NE(local_logits, nullptr);
    auto loss = loss_fn->Forward({local_logits, y})[0];
    ASSERT_NE(loss, nullptr);
    loss->Backward();

    const auto after_backward = GetWeightTyingSnapshot(*model);
    LogWeightTyingSnapshot("after_backward", after_backward);
    EXPECT_EQ(after_backward.wte_grad.get(), after_backward.lm_head_grad.get())
        << "shared weight should expose the same grad object through both paths";
    EXPECT_EQ(after_backward.wte_grad->DataPtr(), after_backward.lm_head_grad->DataPtr())
        << "shared weight should expose the same grad storage through both paths";
    const auto wte_grad_summary = SummarizeFloatTensor(*after_backward.wte_grad, 0);
    const auto lm_head_grad_summary = SummarizeFloatTensor(*after_backward.lm_head_grad, 0);
    LogTensorSummary(0, DiagnosticMode::kFull, "weight_tying_backward", 0, -1, "transformer.wte.weight.grad",
                     wte_grad_summary);
    LogTensorSummary(0, DiagnosticMode::kFull, "weight_tying_backward", 0, -1, "lm_head.weight.grad",
                     lm_head_grad_summary);
    EXPECT_EQ(wte_grad_summary.sample, lm_head_grad_summary.sample);
    EXPECT_EQ(wte_grad_summary.l2, lm_head_grad_summary.l2);

    local_optimizer.Step();
    const auto after_step = GetWeightTyingSnapshot(*model);
    LogWeightTyingSnapshot("after_optimizer_step", after_step);
    EXPECT_EQ(after_step.wte.get(), after_step.lm_head.get());
    EXPECT_EQ(after_step.wte->DataPtr(), after_step.lm_head->DataPtr());
    const auto wte_summary = SummarizeFloatTensor(*after_step.wte, 0);
    const auto lm_head_summary = SummarizeFloatTensor(*after_step.lm_head, 0);
    EXPECT_EQ(wte_summary.sample, lm_head_summary.sample);
    EXPECT_EQ(wte_summary.l2, lm_head_summary.l2);
}

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


TEST_F(GPT2TrainingTest, LinearWeightGradientDeterminism) {
    ASSERT_TRUE(device.IsCUDA()) << "Linear weight-gradient determinism diagnostic requires CUDA";
    ReleaseFixtureTrainingStateForStandaloneKernelTest();

    constexpr int kRuns = 10;
    constexpr int64_t kHiddenSize = 768;
    constexpr int64_t kVocabSize = 50257;
    const int64_t flattened_tokens = static_cast<int64_t>(batch_size * sequence_length);
    LOG(INFO) << "LINEAR_WEIGHT_GRAD_DETERMINISM_BEGIN runs=" << kRuns
              << " flattened_tokens=" << flattened_tokens
              << " hidden_size=" << kHiddenSize
              << " vocab_size=" << kVocabSize
              << " input_shape=[" << batch_size << "x" << sequence_length << "x" << kHiddenSize << "]"
              << " grad_output_shape=[" << batch_size << "x" << sequence_length << "x" << kVocabSize << "]"
              << " weight_shape=[" << kVocabSize << "x" << kHiddenSize << "]"
              << " grad_buffer_shape=[" << kVocabSize << "x" << kHiddenSize << "]"
              << " transpose=true"
              << " accumulation_kernel=AccumulateGrad";

    auto input_cpu = MakeDeterministicFloatTensor({batch_size, sequence_length, kHiddenSize}, 1.0e-4f, 1);
    auto weight_cpu = MakeDeterministicFloatTensor({kVocabSize, kHiddenSize}, 1.0e-5f, 2);
    auto grad_output_cpu = MakeDeterministicFloatTensor({batch_size, sequence_length, kVocabSize}, 1.0e-6f, 3);
    auto initial_grad_cpu = MakeDeterministicFloatTensor({kVocabSize, kHiddenSize}, 1.0e-7f, 4);
    const auto input_payload = CopyTensorToLogitsBinary(*input_cpu);
    const auto weight_payload = CopyTensorToLogitsBinary(*weight_cpu);
    const auto grad_output_payload = CopyTensorToLogitsBinary(*grad_output_cpu);
    const auto initial_grad_payload = CopyTensorToLogitsBinary(*initial_grad_cpu);
    input_cpu.reset();
    weight_cpu.reset();
    grad_output_cpu.reset();
    initial_grad_cpu.reset();

    LogRepeatedTensorResultsIncremental("LinearWeightGradientDeterminism.full_shape", kRuns, [&]() {
        return RunLinearWeightGradientOnce(input_payload, weight_payload, grad_output_payload, initial_grad_payload);
    });
}

TEST_F(GPT2TrainingTest, EmbeddingWeightGradientDeterminism) {
    ASSERT_TRUE(device.IsCUDA()) << "Embedding weight-gradient determinism diagnostic requires CUDA";
    ReleaseFixtureTrainingStateForStandaloneKernelTest();

    constexpr int kRuns = 20;
    constexpr int64_t kHiddenSize = 768;
    constexpr int64_t kVocabSize = 50257;
    const std::vector<int64_t> input_dims{batch_size, sequence_length};
    const size_t num_tokens = static_cast<size_t>(batch_size * sequence_length);
    auto grad_output_cpu = MakeDeterministicFloatTensor({batch_size, sequence_length, kHiddenSize}, 1.0e-4f, 5);
    const auto grad_output_payload = CopyTensorToLogitsBinary(*grad_output_cpu);
    grad_output_cpu.reset();

    struct EmbeddingCase {
        std::string label;
        std::vector<int64_t> input_values;
        bool require_stable;
    };
    const std::vector<EmbeddingCase> cases = {
        {.label = "EmbeddingWeightGradientDeterminism.unique_tokens",
         .input_values = MakeUniqueTokenValues(num_tokens, kVocabSize),
         .require_stable = true},
        {.label = "EmbeddingWeightGradientDeterminism.repeated_tokens",
         .input_values = MakeRepeatedTokenValues(num_tokens, kVocabSize),
         .require_stable = false},
        {.label = "EmbeddingWeightGradientDeterminism.all_identical_tokens",
         .input_values = MakeIdenticalTokenValues(num_tokens, 5),
         .require_stable = false},
    };

    LOG(INFO) << "EMBEDDING_WEIGHT_GRAD_DETERMINISM_BEGIN runs=" << kRuns
              << " input_shape=" << DimsToString(input_dims)
              << " weight_shape=[" << kVocabSize << "x" << kHiddenSize << "]"
              << " grad_output_shape=[" << batch_size << "x" << sequence_length << "x" << kHiddenSize << "]"
              << " cases=unique_tokens,repeated_tokens,all_identical_tokens";

    for (const auto &embedding_case : cases) {
        const auto summary = LogEmbeddingCaseDiagnostics(embedding_case.label, input_dims, embedding_case.input_values,
                                                         grad_output_payload, kRuns);
        if (embedding_case.require_stable) {
            EXPECT_EQ(summary.first_diff_iter, -1)
                << "Embedding backward without repeated token indices must be bitwise stable before attributing the "
                << "nondeterminism to duplicate-index accumulation";
        }
    }
}
TEST_F(GPT2TrainingTest, SharedGradientAccumulationDeterminism) {
    ASSERT_TRUE(device.IsCUDA()) << "Shared gradient accumulation determinism diagnostic requires CUDA";
    ReleaseFixtureTrainingStateForStandaloneKernelTest();

    constexpr int kRuns = 10;
    constexpr int64_t kHiddenSize = 768;
    constexpr int64_t kVocabSize = 50257;
    auto initial_grad_cpu = MakeDeterministicFloatTensor({kVocabSize, kHiddenSize}, 1.0e-7f, 6);
    auto contribution_a_cpu = MakeDeterministicFloatTensor({kVocabSize, kHiddenSize}, 1.0e-6f, 7);
    auto contribution_b_cpu = MakeDeterministicFloatTensor({kVocabSize, kHiddenSize}, 1.0e-6f, 8);
    const auto initial_grad_payload = CopyTensorToLogitsBinary(*initial_grad_cpu);
    const auto contribution_a_payload = CopyTensorToLogitsBinary(*contribution_a_cpu);
    const auto contribution_b_payload = CopyTensorToLogitsBinary(*contribution_b_cpu);
    initial_grad_cpu.reset();
    contribution_a_cpu.reset();
    contribution_b_cpu.reset();

    const std::vector<std::string> orders{"A_then_B", "B_then_A", "real_autograd_order"};
    for (const auto &order : orders) {
        LOG(INFO) << "SHARED_ACCUMULATION_DETERMINISM_BEGIN order=" << order
                  << " runs=" << kRuns
                  << " shape=[" << kVocabSize << "x" << kHiddenSize << "]"
                  << " accumulation_kernel=AccumulateGrad"
                  << " real_autograd_order=B_then_A";
        LogRepeatedTensorResultsIncremental("SharedGradientAccumulationDeterminism." + order, kRuns, [&]() {
            return RunSharedAccumulationOnce(initial_grad_payload, contribution_a_payload, contribution_b_payload,
                                             order);
        });
    }
}

TEST_F(GPT2TrainingTest, TiedTrainingFirstDivergenceDiagnostics) {
    ConfigureGradAccumulation();
    ASSERT_EQ(grad_accum_steps, 2);
    LOG(INFO) << "PAIRED_DIAGNOSTIC_BEGIN batch_size=" << batch_size
              << " sequence_length=" << sequence_length
              << " grad_accum_steps=" << grad_accum_steps
              << " learning_rate=" << learning_rate
              << " step0_full_parameter_scan=true";

    auto microbatches = LoadDiagnosticMicrobatches(grad_accum_steps);

    model.reset();
    optimizer.reset();
    loss_fn.reset();
    tokenizer.reset();

    PairedTrainingRun lhs;
    PairedTrainingRun rhs;
    InitializePairedTrainingRun(lhs, 1);
    InitializePairedTrainingRun(rhs, 2);

    FirstDivergenceTracker first_divergence;
    (void)CompareAndLogAllParameters("load", -1, -1, *lhs.model, *rhs.model, first_divergence);

    auto initial_lhs = RunDiagnosticForward(lhs, microbatches[0]);
    auto initial_rhs = RunDiagnosticForward(rhs, microbatches[0]);
    LogLossComparison("initial_forward", -1, -1, initial_lhs.loss_value, initial_rhs.loss_value, first_divergence);
    auto initial_forward = CompareForwardDiagnostics("initial_forward", -1, -1, lhs.diagnostics, rhs.diagnostics,
                                                     first_divergence);
    const bool initial_loss_equal = std::memcmp(&initial_lhs.loss_value, &initial_rhs.loss_value, sizeof(float)) == 0;
    LOG(INFO) << "INITIAL_FORWARD_BITWISE_EQUAL value=" << (initial_loss_equal && initial_forward.BitwiseEqual())
              << " loss_equal=" << initial_loss_equal
              << " final_block_equal=" << (initial_forward.final_block_output.count_bitwise_mismatch == 0)
              << " ln_f_equal=" << (initial_forward.ln_f_output.count_bitwise_mismatch == 0)
              << " lm_head_input_equal=" << (initial_forward.lm_head_input.count_bitwise_mismatch == 0)
              << " logits_equal=" << (initial_forward.logits.count_bitwise_mismatch == 0);
    initial_lhs = DiagnosticForwardResult{};
    initial_rhs = DiagnosticForwardResult{};
    lhs.diagnostics.Clear();
    rhs.diagnostics.Clear();

    lhs.optimizer->ZeroGrad();
    rhs.optimizer->ZeroGrad();
    LOG(INFO) << "PAIRED_UPDATE_START update=0 microbatch_count=" << grad_accum_steps;

    TensorComparisonMetrics worst_logits_metrics;
    std::string worst_logits_stage = "none";
    auto observe_worst_logits = [&](const std::string &stage, const TensorComparisonMetrics &metrics) {
        if (worst_logits_stage == "none" || metrics.max_abs > worst_logits_metrics.max_abs) {
            worst_logits_stage = stage;
            worst_logits_metrics = metrics;
        }
    };
    observe_worst_logits("initial_forward", initial_forward.logits);

    for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
        const std::string forward_stage = "update0_micro" + std::to_string(micro_step) + "_forward";
        auto forward_lhs = RunDiagnosticForward(lhs, microbatches[micro_step]);
        auto forward_rhs = RunDiagnosticForward(rhs, microbatches[micro_step]);
        LogLossComparison(forward_stage, 0, micro_step, forward_lhs.loss_value, forward_rhs.loss_value,
                          first_divergence);
        auto forward_metrics = CompareForwardDiagnostics(forward_stage, 0, micro_step, lhs.diagnostics, rhs.diagnostics,
                                                         first_divergence);
        observe_worst_logits(forward_stage, forward_metrics.logits);

        auto lhs_backward_snapshots = RunBackwardWithSharedWeightObserver(lhs, *forward_lhs.loss);
        auto rhs_backward_snapshots = RunBackwardWithSharedWeightObserver(rhs, *forward_rhs.loss);
        const std::string backward_stage = "update0_micro" + std::to_string(micro_step) + "_backward";
        CompareBackwardContributionSnapshots(backward_stage + "_shared_weight_contributions", lhs_backward_snapshots,
                                             rhs_backward_snapshots, first_divergence, 0, micro_step);
        (void)CompareAndLogAllParameters(backward_stage, 0, micro_step, *lhs.model, *rhs.model, first_divergence);
    }

    lhs.optimizer->Step();
    rhs.optimizer->Step();
    (void)CompareAndLogAllParameters("update0_optimizer_step", 0, -1, *lhs.model, *rhs.model, first_divergence);

    lhs.optimizer->ZeroGrad();
    rhs.optimizer->ZeroGrad();
    auto next_lhs = RunDiagnosticForward(lhs, microbatches[0]);
    auto next_rhs = RunDiagnosticForward(rhs, microbatches[0]);
    LogLossComparison("update1_start_forward", 1, 0, next_lhs.loss_value, next_rhs.loss_value, first_divergence);
    auto next_forward = CompareForwardDiagnostics("update1_start_forward", 1, 0, lhs.diagnostics, rhs.diagnostics,
                                                  first_divergence);
    observe_worst_logits("update1_start_forward", next_forward.logits);

    LogWorstLogitsRow(worst_logits_stage, worst_logits_metrics);
    first_divergence.Log();
}


TEST_F(GPT2TrainingTest, LogitsConsistencyTiedWeights) {
    AssertTiedWeightsAndOptimizerDedup();

    if (!std::filesystem::exists(tied_logits_reference)) {
        GTEST_SKIP() << "Missing tied-weight GPT-2 logits reference: " << tied_logits_reference
                     << ". Generate a candidate explicitly with: cd build/Release && "
                     << "TINY_GENERATE_GPT2_REFERENCE=1 ./test_gpt2 "
                     << "--gtest_filter=GPT2TrainingTest.GenerateTiedWeightsReference. "
                     << "The generator writes .generated files and never overwrites the checked-in reference.";
    }

    auto final_logits = RunTiedReferenceTrainingAndForward(true);
    auto current = CopyTensorToLogitsBinary(*final_logits);
    LogitsBinary reference;
    try {
        reference = ReadLogitsBinaryFile(tied_logits_reference);
    } catch (const std::exception &e) {
        FAIL() << e.what();
    }
    ASSERT_EQ(reference.dims, current.dims) << "Reference dims do not match final forward logits";

    const auto metrics = CompareLogitsFull(reference, current, kDiagnosticTolerance);
    LogLogitsComparison("tied_weights_10_updates", metrics, kDiagnosticTolerance);
    EXPECT_EQ(metrics.reference_nan_count, 0);
    EXPECT_EQ(metrics.candidate_nan_count, 0);
    EXPECT_EQ(metrics.reference_inf_count, 0);
    EXPECT_EQ(metrics.candidate_inf_count, 0);
    EXPECT_EQ(metrics.count_gt_tolerance, 0)
        << "Tied-weight logits differ from reference; see LOGITS_FULL_COMPARE metrics above";
}

TEST_F(GPT2TrainingTest, GenerateTiedWeightsReference) {
    if (!EnvFlagEnabled("TINY_GENERATE_GPT2_REFERENCE")) {
        GTEST_SKIP() << "Set TINY_GENERATE_GPT2_REFERENCE=1 to write tied-weight GPT-2 reference .generated files";
    }
    ASSERT_FALSE(EnvFlagEnabled("TINY_GENERATE_GPT2_REFERENCE_FORMAL"))
        << "Formal reference generation/promotion is intentionally not implemented here. It must require a clean "
        << "working tree and committed generator code before writing a non-.generated reference.";

    AssertTiedWeightsAndOptimizerDedup();
    const std::string output_path = EnvOrDefault("TINY_GPT2_REFERENCE_OUTPUT", tied_logits_reference_generated);
    const std::string meta_path = EnvOrDefault("TINY_GPT2_REFERENCE_META", tied_logits_reference_meta_generated);
    const bool overwrite = EnvFlagEnabled("TINY_OVERWRITE_GPT2_REFERENCE_GENERATED");
    ASSERT_TRUE(EndsWith(output_path, ".generated"))
        << "Candidate logits output must end with .generated: " << output_path;
    ASSERT_TRUE(EndsWith(meta_path, ".generated"))
        << "Candidate metadata output must end with .generated: " << meta_path;
    ASSERT_FALSE(std::filesystem::exists(tied_logits_reference))
        << "Formal tied-weight reference already exists: " << tied_logits_reference
        << ". This generator only creates .generated candidates.";

    ReferenceTraceMetadata trace;
    try {
        trace = CollectReferenceTraceMetadata();
    } catch (const std::exception &e) {
        FAIL() << "Failed to collect reference trace metadata before writing outputs: " << e.what();
    }

    auto final_logits = RunTiedReferenceTrainingAndForward(true);
    const auto generated = CopyTensorToLogitsBinary(*final_logits);
    ReferenceWriteResult written;
    try {
        written = WriteLogitsBinaryFileAtomic(output_path, generated, overwrite);
    } catch (const std::exception &e) {
        FAIL() << e.what();
    }

    const auto metadata_text = BuildReferenceMetadata(generated, output_path, written.sha256, trace);
    try {
        WriteTextFileAtomic(meta_path, metadata_text, overwrite);
    } catch (const std::exception &e) {
        FAIL() << e.what();
    }
    const auto metadata_sha256 = Sha256File(meta_path);

    LOG(INFO) << "TIED_REFERENCE_GENERATED logits=" << output_path << " metadata=" << meta_path
              << " sha256=" << written.sha256 << " metadata_sha256=" << metadata_sha256;
}

TEST_F(GPT2TrainingTest, TiedReferenceCandidatePairwiseDiagnostics) {
    std::vector<std::string> candidate_paths = {
        "../../gpt2_tied_ref_run1.bin.generated",
        "../../gpt2_tied_ref_run2.bin.generated",
        "../../gpt2_tied_ref_run3.bin.generated",
    };
    if (const char *override_paths = std::getenv("TINY_GPT2_PAIRWISE_CANDIDATES")) {
        candidate_paths.clear();
        std::stringstream stream(override_paths);
        std::string path;
        while (std::getline(stream, path, ',')) {
            path = TrimWhitespace(path);
            if (!path.empty()) {
                candidate_paths.push_back(path);
            }
        }
        ASSERT_GE(candidate_paths.size(), 2) << "TINY_GPT2_PAIRWISE_CANDIDATES must contain at least two paths";
    }
    LOG(INFO) << "TIED_REFERENCE_PAIRWISE_CANDIDATE_COUNT count=" << candidate_paths.size();
    for (size_t i = 0; i < candidate_paths.size(); ++i) {
        LOG(INFO) << "TIED_REFERENCE_PAIRWISE_CANDIDATE index=" << i << " path=" << candidate_paths[i];
    }
    for (const auto &path : candidate_paths) {
        if (!std::filesystem::exists(path)) {
            GTEST_SKIP() << "Missing generated candidate for pairwise diagnostics: " << path;
        }
    }

    std::vector<LogitsBinary> candidates;
    candidates.reserve(candidate_paths.size());
    for (const auto &path : candidate_paths) {
        try {
            candidates.push_back(ReadLogitsBinaryFileAuto(path));
        } catch (const std::exception &e) {
            FAIL() << e.what();
        }
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            const auto metrics = CompareLogitsFull(candidates[i], candidates[j], kDiagnosticTolerance);
            const std::string label = "run" + std::to_string(i + 1) + "_vs_run" + std::to_string(j + 1);
            LogLogitsComparison(label, metrics, kDiagnosticTolerance);
            EXPECT_EQ(metrics.reference_nan_count, 0) << label;
            EXPECT_EQ(metrics.candidate_nan_count, 0) << label;
            EXPECT_EQ(metrics.reference_inf_count, 0) << label;
            EXPECT_EQ(metrics.candidate_inf_count, 0) << label;
        }
    }
}


// Historical reference check. The checked-in reference was generated from the old split-weight
// device-move behavior, so it is not a default correctness gate for tied-weight GPT-2.
// Run explicitly with --gtest_also_run_disabled_tests and this disabled test filter if needed.
TEST_F(GPT2TrainingTest, DISABLED_LogitsConsistencyLegacySplitWeights) {
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
