#include "example/common/tokenizer.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr size_t kHeaderSize = 1024;

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

class DisableParameterGradGuard {
public:
    explicit DisableParameterGradGuard(nn::Module &model) : parameters_(model.Parameters()) {
        requires_grad_.reserve(parameters_.size());
        for (const auto &parameter : parameters_) { requires_grad_.push_back(parameter->requires_grad()); }
        for (const auto &parameter : parameters_) { parameter->set_requires_grad(false); }
    }

    ~DisableParameterGradGuard() {
        for (size_t idx = 0; idx < parameters_.size(); ++idx) {
            parameters_[idx]->set_requires_grad(requires_grad_[idx]);
        }
    }

private:
    std::vector<std::shared_ptr<Tensor>> parameters_;
    std::vector<bool> requires_grad_;
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    CHECK_EQ(ifs->gcount(), static_cast<std::streamsize>(num_bytes)) << "Unexpected end of file";
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

size_t SampleMult(const float *probabilities, size_t n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    CHECK(std::filesystem::exists(filepath)) << "Tokenizer file not found: " << filepath;
    CHECK(std::filesystem::is_regular_file(filepath)) << "Tokenizer path is not a regular file: " << filepath;
    const uintmax_t file_size = std::filesystem::file_size(filepath);
    CHECK_GE(file_size, kHeaderSize) << "Tokenizer file is smaller than its header";

    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;
    const auto header = ReadSeveralBytesFromIfstream(kHeaderSize, &ifs);
    magic_number_ = BytesToType<uint32_t>(header, 0);
    const uint32_t version_value = BytesToType<uint32_t>(header, sizeof(uint32_t));
    vocab_size_ = BytesToType<uint32_t>(header, 2 * sizeof(uint32_t));

    CHECK(kEotMap.contains(magic_number_)) << "Unsupported tokenizer magic: " << magic_number_;
    CHECK(version_value == static_cast<uint32_t>(Version::kV1)
          || version_value == static_cast<uint32_t>(Version::kV2))
        << "Unsupported tokenizer version: " << version_value;
    CHECK_GT(vocab_size_, 0);
    CHECK_LE(static_cast<uintmax_t>(vocab_size_), file_size - kHeaderSize)
        << "Tokenizer vocabulary cannot fit in file";

    if (version_value == static_cast<uint32_t>(Version::kV2)) {
        eot_token_ = BytesToType<uint32_t>(header, 3 * sizeof(uint32_t));
    } else {
        eot_token_ = kEotMap.at(magic_number_);
    }
    CHECK_LT(eot_token_, vocab_size_) << "Tokenizer end token is outside the vocabulary";

    token_table_.reserve(vocab_size_);
    for (uint32_t token_id = 0; token_id < vocab_size_; ++token_id) {
        const int length_byte = ifs.get();
        CHECK_NE(length_byte, std::char_traits<char>::eof()) << "Missing length for token " << token_id;
        const size_t token_length = static_cast<uint8_t>(length_byte);
        std::string token(token_length, '\0');
        if (token_length > 0) {
            ifs.read(token.data(), token_length);
            CHECK_EQ(ifs.gcount(), static_cast<std::streamsize>(token_length))
                << "Truncated bytes for token " << token_id;
        }
        token_table_.push_back(std::move(token));
    }
    CHECK_EQ(ifs.peek(), std::char_traits<char>::eof()) << "Unexpected trailing tokenizer data";
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    if (token_id >= token_table_.size()) {
        return "";
    }
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device, uint64_t seed) const {
    CHECK_GT(batch_size, 0);
    CHECK_GT(sequence_length, 0);
    CHECK_LE(text_length, sequence_length);
    const uint64_t num_input_tokens = static_cast<uint64_t>(batch_size) * sequence_length;
    CHECK_LE(num_input_tokens, static_cast<uint64_t>(std::numeric_limits<size_t>::max()));
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (size_t i = 0; i < num_input_tokens; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    const auto &prompt = kPromptMap.at(magic_number_);
    const size_t prompt_len = prompt.size();
    CHECK_LE(prompt_len, text_length);
    for (size_t i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    DisableParameterGradGuard parameter_grad_guard(model);
    uint64_t rng_state = seed;
    LOG(INFO) << "start generate text:";
    for (size_t t = prompt_len; t < text_length; ++t) {
        const auto outputs = model.Forward({x});
        CHECK_GT(outputs.size(), 0);
        CHECK(outputs[0]);
        const auto &logits = outputs[0];
        CHECK_EQ(logits->Dims().size(), 3);
        CHECK_EQ(logits->Dims()[0], batch_size);
        CHECK_EQ(logits->Dims()[1], sequence_length);
        CHECK_EQ(logits->Dims()[2], vocab_size_);

        auto next_token_logits
            = logits->Slice({0, static_cast<int64_t>(t - 1), 0},
                            {1, static_cast<int64_t>(t), static_cast<int64_t>(vocab_size_)}, {1, 1, 1});
        auto probabilities = nn::function::Softmax(next_token_logits, -1);
        auto probabilities_cpu = probabilities->To(Device(DeviceType::kCPU, 0));
        auto *probabilities_ptr = static_cast<float *>(probabilities_cpu.DataPtr());
        const uint32_t next_token
            = static_cast<uint32_t>(SampleMult(probabilities_ptr, vocab_size_, RandomF32(rng_state)));

        x_buff[t] = next_token;
        std::cout << Decode(next_token);
        x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    }
    std::cout << std::endl;
}
} // namespace infini_train
