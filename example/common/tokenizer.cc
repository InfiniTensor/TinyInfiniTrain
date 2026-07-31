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
    // 检查词表文件是否存在
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }

    // 使用二进制方式打开文件
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open())
        << "Failed to open tokenizer file: " << filepath;

    // 读取前 1024 字节文件头
    const auto header =
        ReadSeveralBytesFromIfstream(1024, &ifs);

    CHECK(ifs)
        << "Failed to read tokenizer header";

    // 解析文件头
    magic_number_ =
        BytesToType<uint32_t>(header, 0);

    const uint32_t version_number =
        BytesToType<uint32_t>(header, 4);

    vocab_size_ =
        BytesToType<uint32_t>(header, 8);

    // 检查是否为支持的 Tokenizer 类型
    CHECK(kEotMap.contains(magic_number_))
        << "Unsupported tokenizer magic: "
        << magic_number_;

    const Version version =
        static_cast<Version>(version_number);

    // 不同版本确定 EOT token 的方式不同
    if (version == Version::kV1) {
        eot_token_ = kEotMap.at(magic_number_);
    } else if (version == Version::kV2) {
        eot_token_ =
            BytesToType<uint32_t>(header, 12);
    } else {
        LOG(FATAL)
            << "Unsupported tokenizer version: "
            << version_number;
    }

    // 为整个词表分配空间
    token_table_.resize(vocab_size_);

    // 逐个读取 token 文本
    for (uint32_t token_id = 0;
         token_id < vocab_size_;
         ++token_id) {

        // 每个 token 的第一个字节表示文本长度
        uint8_t length = 0;

        ifs.read(
            reinterpret_cast<char *>(&length),
            sizeof(length)
        );

        CHECK(ifs)
            << "Failed to read token length, token_id = "
            << token_id;

        // 再读取 length 个字节的文本
        std::vector<char> buffer(length);

        if (length > 0) {
            ifs.read(buffer.data(), length);

            CHECK(ifs)
                << "Failed to read token data, token_id = "
                << token_id;
        }

        token_table_[token_id] =
            std::string(buffer.begin(), buffer.end());
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
     if (token_id >= vocab_size_) {
        return "[INVALID_TOKEN]";
    }
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
    uint64_t rng_state = kRngState;
    LOG(INFO) << "start generate text:";
    auto cpu_device = Device();
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
auto parameters = model.Parameters();

        std::vector<bool> requires_grad_flags;
        requires_grad_flags.reserve(parameters.size());

        for (const auto &parameter : parameters) {
            requires_grad_flags.push_back(
                parameter->requires_grad()
            );
            parameter->set_requires_grad(false);
        }

        x = std::make_shared<infini_train::Tensor>(
            x->To(device)
        );

        auto logits = model.Forward({x})[0];

        const int64_t vocab_size =
            logits->Dims()[2];

        auto current_logits = logits->Slice(
            {0, t - 1, 0},
            {1, t, vocab_size},
            {1, 1, 1}
        );

        auto probabilities =
            nn::function::Softmax(
                current_logits,
                -1
            );

        auto probabilities_cpu =
            probabilities->To(cpu_device);

        for (size_t i = 0;
             i < parameters.size();
             ++i) {
            parameters[i]->set_requires_grad(
                requires_grad_flags[i]
            );
        }

        float *probs =
            static_cast<float *>(
                probabilities_cpu.DataPtr()
            );

        float coin = RandomF32(rng_state);

        int next_token = SampleMult(
            probs,
            static_cast<int>(vocab_size),
            coin
        );

        x = std::make_shared<infini_train::Tensor>(
            x->To(cpu_device)
        );

        auto data =
            static_cast<int64_t *>(x->DataPtr());

        data[t] = next_token;

        std::cout << Decode(next_token);
    }

    std::cout << std::endl;
}
} // namespace infini_train
