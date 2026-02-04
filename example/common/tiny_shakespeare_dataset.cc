#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */

    TinyShakespeareFile out;
    if (!std::filesystem::exists(path)) {
        LOG(FATAL) << "File not found: " << path;
    }

    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    const size_t header_bytes = 1024;
    auto header = ReadSeveralBytesFromIfstream(header_bytes, &ifs);

    const auto version = BytesToType<uint32_t>(header, 4);
    CHECK(kTypeMap.find(static_cast<int>(version)) != kTypeMap.end())
        << "Unsupported tiny shakespeare version: " << version;
    const auto type = kTypeMap.at(static_cast<int>(version));
    out.type = type;

    const auto num_toks = BytesToType<uint32_t>(header, 8);
    CHECK_GT(num_toks, 0U);

    const size_t orig_type_size = kTypeToSize.at(type);

    const size_t token_bytes = static_cast<size_t>(num_toks) * orig_type_size;
    std::vector<uint8_t> tokens_bytes(token_bytes);
    ifs.read(reinterpret_cast<char *>(tokens_bytes.data()), token_bytes);
    CHECK_EQ(static_cast<size_t>(ifs.gcount()), token_bytes) << "Failed to read token data";

    const size_t sample_stride = sequence_length + 1; // each sample holds seq_len + 1 tokens
    const size_t num_samples = num_toks / sample_stride;
    CHECK_GT(num_samples, 0U) << "Not enough tokens for given sequence_length";

    std::vector<int64_t> storage(num_samples * sample_stride);

    for (size_t i = 0; i < num_toks; ++i) {
        int64_t val = 0;
        if (orig_type_size == 2) {
            val = static_cast<int64_t>(BytesToType<uint16_t>(tokens_bytes, i * orig_type_size));
        } else if (orig_type_size == 4) {
            val = static_cast<int64_t>(BytesToType<uint32_t>(tokens_bytes, i * orig_type_size));
        } else {
            LOG(FATAL) << "Unsupported token size: " << orig_type_size;
        }
        storage[i] = val;
    }

    const std::vector<int64_t> backing_dims = {static_cast<int64_t>(num_samples * sample_stride)};
    out.tensor = infini_train::Tensor(backing_dims, infini_train::DataType::kINT64);
    memcpy(out.tensor.DataPtr(), storage.data(), storage.size() * sizeof(int64_t));

    out.dims = {static_cast<int64_t>(num_samples), static_cast<int64_t>(sequence_length)};

    return out;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    text_file_ = ReadTinyShakespeareFile(filepath, sequence_length);
    sequence_length_ = sequence_length;
    const size_t sample_stride = sequence_length + 1;
    sequence_size_in_bytes_ = sample_stride * sizeof(int64_t);
    num_samples_ = static_cast<size_t>(text_file_.dims[0]);
}

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
