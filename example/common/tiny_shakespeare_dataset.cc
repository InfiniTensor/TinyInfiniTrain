#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open dataset file: " << path;

    // Read 1024-byte header
    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    const uint32_t magic = BytesToType<uint32_t>(header, 0);
    // version at offset 4 (unused here)
    const uint32_t num_tokens = BytesToType<uint32_t>(header, 8);

    CHECK(kTypeMap.contains(magic)) << "Unknown dataset magic: " << magic;
    const TinyShakespeareType type = kTypeMap.at(magic);
    const size_t token_bytes = kTypeToSize.at(type);

    // Read raw token data and convert to int64
    std::vector<int64_t> tokens(num_tokens);
    for (uint32_t i = 0; i < num_tokens; ++i) {
        auto raw = ReadSeveralBytesFromIfstream(token_bytes, &ifs);
        if (type == TinyShakespeareType::kUINT16) {
            tokens[i] = static_cast<int64_t>(BytesToType<uint16_t>(raw, 0));
        } else {
            tokens[i] = static_cast<int64_t>(BytesToType<uint32_t>(raw, 0));
        }
    }

    const size_t num_windows = num_tokens / sequence_length;
    CHECK_GT(num_windows, 1) << "Not enough tokens for even one sample";

    // Build flat int64 tensor of shape [num_windows * sequence_length]
    // Tensor constructor takes dims; store flat then let operator[] slice
    const std::vector<int64_t> flat_dims = {static_cast<int64_t>(num_windows * sequence_length)};
    infini_train::Tensor tensor(flat_dims, infini_train::DataType::kINT64);
    std::memcpy(tensor.DataPtr(), tokens.data(), num_tokens * sizeof(int64_t));

    TinyShakespeareFile result;
    result.type = type;
    result.dims = {static_cast<int64_t>(num_windows), static_cast<int64_t>(sequence_length)};
    result.tensor = std::move(tensor);
    return result;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
      sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      num_samples_(static_cast<size_t>(text_file_.dims[0]) - 1) {}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
