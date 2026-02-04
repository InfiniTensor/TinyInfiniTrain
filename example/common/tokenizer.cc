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

    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "Tokenizer file not found: " << filepath;
    }

    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    const size_t header_bytes = 1024;
    auto header = ReadSeveralBytesFromIfstream(header_bytes, &ifs);

    const auto file_magic = BytesToType<uint32_t>(header, 0);
    const auto version = BytesToType<uint32_t>(header, 4);
    const auto vocab_size = BytesToType<uint32_t>(header, 8);

    magic_number_ = version;
    vocab_size_ = vocab_size;

    const auto file_size = std::filesystem::file_size(filepath);
    const size_t remaining = (file_size > header_bytes) ? static_cast<size_t>(file_size - header_bytes) : 0;
    CHECK_GT(remaining, 0u) << "Empty vocab table in tokenizer file";

    std::vector<uint8_t> table_bytes(remaining);
    ifs.read(reinterpret_cast<char *>(table_bytes.data()), remaining);

    size_t pos = 0;
    token_table_.reserve(vocab_size_);

    bool parsed = false;
    if (remaining >= 4) {
        const uint32_t first_len = BytesToType<uint32_t>(table_bytes, 0);
        if (first_len > 0 && first_len < remaining) {
            pos = 0;
            try {
                for (uint32_t i = 0; i < vocab_size_ && pos + 4 <= remaining; ++i) {
                    uint32_t len = BytesToType<uint32_t>(table_bytes, pos);
                    pos += 4;
                    CHECK_LE(pos + len, remaining) << "Tokenizer entry length overflow";
                    std::string token(reinterpret_cast<const char *>(&table_bytes[pos]), len);
                    token_table_.push_back(std::move(token));
                    pos += len;
                }
                if (token_table_.size() == vocab_size_) parsed = true;
            } catch (...) {
                parsed = false;
            }
        }
    }

    if (!parsed) {
        token_table_.clear();
        std::string cur;
        for (size_t i = 0; i < table_bytes.size() && token_table_.size() < vocab_size_; ++i) {
            if (table_bytes[i] == '\0') {
                token_table_.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(static_cast<char>(table_bytes[i]));
            }
        }
        if (!cur.empty() && token_table_.size() < vocab_size_) token_table_.push_back(cur);
    }

    if (token_table_.size() > vocab_size_) token_table_.resize(vocab_size_);
    while (token_table_.size() < vocab_size_) token_table_.push_back("");

    auto it = kEotMap.find(magic_number_);
    if (it != kEotMap.end()) {
        eot_token_ = it->second;
    } else {
        eot_token_ = kGpt2Eot; // default
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    if (token_id >= token_table_.size()) return std::string();
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

    uint64_t rng_state = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        // prepare input on device
        auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));

        // forward
        auto outputs = model.Forward({x});
        auto logits = outputs[0];

        // move logits to CPU for sampling
        auto logits_cpu = logits->To(infini_train::Device(infini_train::DeviceType::kCPU, 0));
        const auto &ldims = logits_cpu.Dims();
        CHECK_EQ(ldims.size(), 3);
        const int B = static_cast<int>(ldims[0]);
        const int T = static_cast<int>(ldims[1]);
        const int V = static_cast<int>(ldims[2]);

        const float *logits_ptr = static_cast<const float *>(logits_cpu.DataPtr());

        // for each batch, sample from the distribution at position t
        std::vector<float> probs(V);
        for (int b = 0; b < B; ++b) {
            const float *row = logits_ptr + static_cast<size_t>(b * T + t) * V;
            // softmax (stable)
            float maxv = row[0];
            for (int i = 1; i < V; ++i) maxv = std::max(maxv, row[i]);
            double sum = 0.0;
            for (int i = 0; i < V; ++i) {
                probs[i] = std::exp(row[i] - maxv);
                sum += probs[i];
            }
            for (int i = 0; i < V; ++i) probs[i] = static_cast<float>(probs[i] / sum);

            // sample
            float coin = RandomF32(rng_state);
            int sampled = SampleMult(probs.data(), V, coin);

            // write sampled token into CPU buffer
            x_buff[static_cast<size_t>(b) * sequence_length + t] = static_cast<int64_t>(sampled);

            // print decoded text
            std::cout << Decode(static_cast<uint32_t>(sampled));
        }
    }
    std::cout << std::endl;
}
} // namespace infini_train
