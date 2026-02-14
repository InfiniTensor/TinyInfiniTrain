#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kDefaultRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    // 与 dataset 一样，要求严格按预期长度读取，避免词表解析错位。
    ifs->read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(num_bytes));
    CHECK_EQ(static_cast<size_t>(ifs->gcount()), num_bytes) << "Failed to read enough bytes from tokenizer file";
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    CHECK(std::filesystem::exists(filepath)) << "Tokenizer file not found: " << filepath;
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    magic_number_ = BytesToType<uint32_t>(header, 0);
    const auto version = BytesToType<uint32_t>(header, 4);
    vocab_size_ = BytesToType<uint32_t>(header, 8);

    CHECK(version == static_cast<uint32_t>(Version::kV1) || version == static_cast<uint32_t>(Version::kV2))
        << "Unsupported tokenizer version: " << version;

    // v2 的 eot token 直接写在 header 中；v1 走 magic->eot 映射兼容旧文件。
    if (version == static_cast<uint32_t>(Version::kV2)) {
        eot_token_ = BytesToType<uint32_t>(header, 12);
    } else {
        auto eot_iter = kEotMap.find(magic_number_);
        CHECK(eot_iter != kEotMap.end()) << "Unsupported tokenizer magic number: " << magic_number_;
        eot_token_ = eot_iter->second;
    }

    // 词表表项格式：1字节长度 + 对应长度的 token 字节序列。
    token_table_.clear();
    token_table_.reserve(vocab_size_);
    for (uint32_t idx = 0; idx < vocab_size_; ++idx) {
        const auto token_length = ReadSeveralBytesFromIfstream(sizeof(uint8_t), &ifs)[0];
        auto token_bytes = ReadSeveralBytesFromIfstream(token_length, &ifs);
        token_table_.emplace_back(reinterpret_cast<const char *>(token_bytes.data()), token_bytes.size());
    }

    CHECK_EQ(token_table_.size(), vocab_size_);
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    CHECK_LT(token_id, token_table_.size()) << "token_id out of range: " << token_id;
    // 这里返回 token 的原始字符串片段（可能含前导空格或特殊字符）。
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt_iter = kPromptMap.find(magic_number_);
    CHECK(prompt_iter != kPromptMap.end()) << "Unsupported tokenizer magic number for prompt: " << magic_number_;
    const auto &prompt = prompt_iter->second;
    auto prompt_len = prompt.size();
    CHECK_LE(prompt_len, sequence_length) << "Prompt length exceeds sequence length";
    CHECK_LE(text_length, sequence_length) << "text_length must be <= sequence_length in this implementation";
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    // 固定 RNG 状态，保证采样行为可复现。
    uint64_t rng_state = kDefaultRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        // 1) 模型前向得到 (bs, seq_len, vocab)。
        auto logits = model.Forward({x})[0];
        // 2) 取当前时间步 t-1 的词表 logits（只用 batch0 做采样）。
        auto step_logits = logits->Slice({0, t - 1, 0}, {1, t, logits->Dims()[2]}, {1, 1, 1})
                               ->Contiguous()
                               ->View({logits->Dims()[2]});
        // 3) logits -> 概率分布。
        auto probabilities = infini_train::nn::function::Softmax(step_logits, 0);

        // 4) 当前采样函数在 CPU 上执行，先把概率搬到 CPU。
        auto probabilities_cpu = std::make_shared<infini_train::Tensor>(probabilities->To(Device(DeviceType::kCPU, 0)));
        auto *prob_ptr = static_cast<float *>(probabilities_cpu->DataPtr());
        const auto next_token = static_cast<uint32_t>(
            SampleMult(prob_ptr, static_cast<int>(probabilities_cpu->NumElements()), RandomF32(rng_state)));

        // 5) 把采样出的 token 写回所有 batch 的第 t 位，形成下一步输入。
        auto x_cpu = std::make_shared<infini_train::Tensor>(x->To(Device(DeviceType::kCPU, 0)));
        auto *x_cpu_ptr = static_cast<int64_t *>(x_cpu->DataPtr());
        for (int batch_idx = 0; batch_idx < static_cast<int>(batch_size); ++batch_idx) {
            x_cpu_ptr[batch_idx * static_cast<int>(sequence_length) + t] = static_cast<int64_t>(next_token);
        }
        x = std::make_shared<infini_train::Tensor>(x_cpu->To(device));

        std::cout << Decode(next_token);
    }
    std::cout << std::endl;
}
} // namespace infini_train
