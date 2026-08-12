#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

constexpr size_t kHeaderSize = 1024;
constexpr uint32_t kDatasetVersion = 1;

const std::unordered_map<uint32_t, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    CHECK_GT(sequence_length, 0);
    CHECK_LE(sequence_length, static_cast<size_t>(std::numeric_limits<int64_t>::max()));
    CHECK(std::filesystem::exists(path)) << "Dataset file not found: " << path;
    CHECK(std::filesystem::is_regular_file(path)) << "Dataset path is not a regular file: " << path;

    const uintmax_t file_size = std::filesystem::file_size(path);
    CHECK_GE(file_size, kHeaderSize) << "Dataset file is smaller than its header: " << path;

    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open dataset file: " << path;
    const auto header = ReadSeveralBytesFromIfstream(kHeaderSize, &ifs);
    const uint32_t magic = BytesToType<uint32_t>(header, 0);
    const uint32_t version = BytesToType<uint32_t>(header, sizeof(uint32_t));
    const uint32_t num_tokens = BytesToType<uint32_t>(header, 2 * sizeof(uint32_t));

    CHECK(kTypeMap.contains(magic)) << "Unsupported dataset magic: " << magic;
    CHECK_EQ(version, kDatasetVersion) << "Unsupported dataset version: " << version;
    const TinyShakespeareType type = kTypeMap.at(magic);
    const size_t encoded_token_size = kTypeToSize.at(type);
    const uintmax_t expected_size = kHeaderSize + static_cast<uintmax_t>(num_tokens) * encoded_token_size;
    CHECK_EQ(file_size, expected_size) << "Dataset file length does not match its header";
    CHECK_GT(num_tokens, 0);

    std::vector<int64_t> tokens(num_tokens);
    if (type == TinyShakespeareType::kUINT16) {
        std::vector<uint16_t> encoded_tokens(num_tokens);
        ifs.read(reinterpret_cast<char *>(encoded_tokens.data()), encoded_tokens.size() * sizeof(uint16_t));
        CHECK_EQ(ifs.gcount(), static_cast<std::streamsize>(encoded_tokens.size() * sizeof(uint16_t)))
            << "Unexpected end of dataset token data";
        for (size_t idx = 0; idx < encoded_tokens.size(); ++idx) { tokens[idx] = encoded_tokens[idx]; }
    } else {
        std::vector<uint32_t> encoded_tokens(num_tokens);
        ifs.read(reinterpret_cast<char *>(encoded_tokens.data()), encoded_tokens.size() * sizeof(uint32_t));
        CHECK_EQ(ifs.gcount(), static_cast<std::streamsize>(encoded_tokens.size() * sizeof(uint32_t)))
            << "Unexpected end of dataset token data";
        for (size_t idx = 0; idx < encoded_tokens.size(); ++idx) { tokens[idx] = encoded_tokens[idx]; }
    }

    const size_t num_samples = (static_cast<size_t>(num_tokens) - 1) / sequence_length;
    CHECK_GT(num_samples, 0) << "Dataset has too few tokens for sequence length " << sequence_length;
    CHECK_LE(num_samples, static_cast<size_t>(std::numeric_limits<int64_t>::max()));

    TinyShakespeareFile result;
    result.type = type;
    result.dims = {static_cast<int64_t>(num_samples), static_cast<int64_t>(sequence_length)};
    result.tensor = infini_train::Tensor({static_cast<int64_t>(num_tokens)}, DataType::kINT64);
    std::memcpy(result.tensor.DataPtr(), tokens.data(), tokens.size() * sizeof(int64_t));
    return result;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      num_samples_(static_cast<size_t>(text_file_.dims[0])) {}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, num_samples_);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
