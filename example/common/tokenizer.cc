#include "example/common/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
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
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    // Read 1024-byte header
    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    magic_number_ = BytesToType<uint32_t>(header, 0);
    // uint32_t version = BytesToType<uint32_t>(header, 4);  // unused
    vocab_size_ = BytesToType<uint32_t>(header, 8);

    CHECK(kEotMap.contains(magic_number_)) << "Unknown tokenizer magic: " << magic_number_;
    eot_token_ = kEotMap.at(magic_number_);

    // Read vocabulary: each entry is 1-byte length followed by that many bytes
    token_table_.resize(vocab_size_);
    for (uint32_t i = 0; i < vocab_size_; ++i) {
        const uint8_t len = ReadSeveralBytesFromIfstream(1, &ifs)[0];
        auto bytes = ReadSeveralBytesFromIfstream(len, &ifs);
        token_table_[i] = std::string(bytes.begin(), bytes.end());
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    CHECK_LT(token_id, token_table_.size()) << "Token ID out of range: " << token_id;
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
        // Forward pass: logits shape (batch_size, sequence_length, vocab_size)
        auto outputs = model.Forward({x});
        auto logits = outputs[0];
        const int64_t vocab_size = static_cast<int64_t>(logits->Dims().back());

        // Move logits to CPU and pick position t-1 of batch 0
        auto logits_cpu = logits->To(Device(DeviceType::kCPU, 0));
        const float *lp = static_cast<const float *>(logits_cpu.DataPtr()) + (t - 1) * vocab_size;

        // Numerically-stable softmax
        std::vector<float> probs(vocab_size);
        float mx = *std::max_element(lp, lp + vocab_size);
        float sum = 0.0f;
        for (int64_t i = 0; i < vocab_size; ++i) {
            probs[i] = std::expf(lp[i] - mx);
            sum += probs[i];
        }
        for (float &p : probs) { p /= sum; }

        // Multinomial sample
        const int next_tok = SampleMult(probs.data(), static_cast<int>(vocab_size), RandomF32(kRngState));

        // Update token in CPU buffer and refresh device tensor
        x_buff[t] = static_cast<int64_t>(next_tok);
        x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));

        // Decode and print
        std::cout << Decode(static_cast<uint32_t>(next_tok));
    }
    std::cout << std::endl;
}
} // namespace infini_train
