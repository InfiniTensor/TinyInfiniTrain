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
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    auto header_bytes = ReadSeveralBytesFromIfstream(1024, &ifs);
    const uint32_t magic = BytesToType<uint32_t>(header_bytes, 0);
    const uint32_t version = BytesToType<uint32_t>(header_bytes, 4);
    const uint32_t num_toks = BytesToType<uint32_t>(header_bytes, 8);
    (void)version;

    CHECK(kTypeMap.contains(magic)) << "Unsupported magic number: " << magic;
    const TinyShakespeareType type = kTypeMap.at(magic);
    const size_t token_size = kTypeToSize.at(type);

    CHECK_GT(num_toks, sequence_length) << "Not enough tokens for one sequence";
    infini_train::Tensor tensor(std::vector<int64_t>{static_cast<int64_t>(num_toks)}, DataType::kINT64);
    auto *dst = static_cast<int64_t *>(tensor.DataPtr());

    if (token_size == 2) {
        std::vector<uint16_t> tokens(num_toks);
        ifs.read(reinterpret_cast<char *>(tokens.data()), static_cast<std::streamsize>(num_toks * token_size));
        for (uint32_t i = 0; i < num_toks; ++i) {
            dst[i] = static_cast<int64_t>(tokens[i]);
        }
    } else {
        std::vector<uint32_t> tokens(num_toks);
        ifs.read(reinterpret_cast<char *>(tokens.data()), static_cast<std::streamsize>(num_toks * token_size));
        for (uint32_t i = 0; i < num_toks; ++i) {
            dst[i] = static_cast<int64_t>(tokens[i]);
        }
    }

    // dims[0] is used by operator[] as CHECK_LT(idx, dims[0] - 1)
    // Non-overlapping windows (llm.c semantics): sample idx covers tokens[idx*seq_len : (idx+1)*seq_len]
    const int64_t num_samples = static_cast<int64_t>(num_toks) / static_cast<int64_t>(sequence_length);
    std::vector<int64_t> dims{num_samples + 1, static_cast<int64_t>(sequence_length)};
    return TinyShakespeareFile{type, dims, std::move(tensor)};
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)), num_samples_(text_file_.dims[0] - 1) {}

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
