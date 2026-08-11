#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

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
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
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
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
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
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    CHECK_LT(token_id, vocab_size_) << "token_id out of range: " << token_id;
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
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t kRngState = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        // 生成场景无 Backward：临时禁用参数梯度使前向不建 autograd 图（Function 即时释放），
        // 避免算子 saved_tensors_ 的循环引用在无 Backward 场景下导致显存逐步入泄漏
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

            static uint64_t rng_state = kRngState;
            const int next_token = SampleMult(probs.data(), static_cast<int>(vocab_size), RandomF32(rng_state));
            x_buff[b * sequence_length + t] = next_token;
            std::cout << Decode(next_token);
        }
    }
    std::cout << std::endl;
}
} // namespace infini_train
