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
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    CHECK_EQ(ifs.gcount(), 1024) << "Truncated header in file: " << path;
    const int magic = BytesToType<int>(header, 0);
    const int num_toks = BytesToType<int>(header, 8);
    CHECK(kTypeMap.contains(magic)) << "Unsupported magic number: " << magic;
    CHECK_GE(num_toks, 0) << "Invalid num_toks in file: " << path;

    TinyShakespeareFile file;
    file.type = kTypeMap.at(magic);
    const size_t token_size = kTypeToSize.at(file.type);

    // 读 token 流并转为 int64 张量：CrossEntropy 的 target 要求 int64，
    // 且 operator[] 中 y 相对 x 偏移 sizeof(int64_t) = 8 字节，恰为一个 int64 token（预测下一 token）
    const size_t num_tok_bytes = static_cast<size_t>(num_toks) * token_size;
    auto token_bytes = ReadSeveralBytesFromIfstream(num_tok_bytes, &ifs);
    // 读取完整性校验（损坏/截断文件下避免短向量导致的越界读）
    CHECK_EQ(static_cast<size_t>(ifs.gcount()), num_tok_bytes) << "Truncated token data in file: " << path;
    const size_t num_samples = static_cast<size_t>(num_toks) / sequence_length;
    file.dims = {static_cast<int64_t>(num_samples), static_cast<int64_t>(sequence_length)};
    file.tensor = infini_train::Tensor(file.dims, DataType::kINT64);
    int64_t *tensor_data = static_cast<int64_t *>(file.tensor.DataPtr());
    for (size_t i = 0; i < num_samples * sequence_length; ++i) {
        tensor_data[i] = (token_size == 2) ? BytesToType<uint16_t>(token_bytes, i * 2)
                                           : BytesToType<int32_t>(token_bytes, i * 4);
    }
    return file;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
      sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      // 最后一个序列缺"下一 token"作标签，operator[] 以 CHECK_LT(idx, dims[0]-1) 限定可访问边界，
      // 故 Size 报告可访问样本数 = dims[0]-1（空数据集防护为 0），与 operator[] 的契约一致
      num_samples_(text_file_.dims[0] > 0 ? text_file_.dims[0] - 1 : 0) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x/y 与 text_file_.tensor 共享 buffer：x 取第 idx 个序列，y 偏移一个 token（预测下一 token）
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
