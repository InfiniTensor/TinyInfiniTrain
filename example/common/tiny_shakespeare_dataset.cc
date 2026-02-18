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
    // 这里要求“精确读取”指定字节数，避免文件截断时静默读到脏数据。
    ifs->read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(num_bytes));
    CHECK_EQ(static_cast<size_t>(ifs->gcount()), num_bytes) << "Failed to read enough bytes from dataset file";
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
    if (!std::filesystem::exists(path)) {
        LOG(FATAL) << "Dataset file not found: " << path;
    }

    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open dataset file: " << path;

    // header: [magic | version | num_tokens | ...]
    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    const auto magic = BytesToType<uint32_t>(header, 0);
    const auto version = BytesToType<uint32_t>(header, 4);
    const auto num_tokens = static_cast<size_t>(BytesToType<uint32_t>(header, 8));

    (void)version;
    CHECK_GT(num_tokens, sequence_length)
        << "Dataset token count must be larger than sequence length, got num_tokens=" << num_tokens
        << ", sequence_length=" << sequence_length;

    auto type_iter = kTypeMap.find(static_cast<int>(magic));
    CHECK(type_iter != kTypeMap.end()) << "Unsupported dataset magic number: " << magic;

    TinyShakespeareFile text_file;
    text_file.type = type_iter->second;
    // token 原始存储可能是 uint16/uint32，先整块读取，再逐个转成 int64。
    const size_t type_size = kTypeToSize.at(text_file.type);
    const auto token_bytes = ReadSeveralBytesFromIfstream(num_tokens * type_size, &ifs);

    // 统一转成 int64 token tensor，方便后续 embedding 直接索引。
    infini_train::Tensor token_tensor({static_cast<int64_t>(num_tokens)}, DataType::kINT64);
    auto *token_buffer = static_cast<int64_t *>(token_tensor.DataPtr());

    switch (text_file.type) {
    case TinyShakespeareType::kUINT16: {
        for (size_t idx = 0; idx < num_tokens; ++idx) {
            token_buffer[idx] = static_cast<int64_t>(BytesToType<uint16_t>(token_bytes, idx * sizeof(uint16_t)));
        }
        break;
    }
    case TinyShakespeareType::kUINT32: {
        for (size_t idx = 0; idx < num_tokens; ++idx) {
            token_buffer[idx] = static_cast<int64_t>(BytesToType<uint32_t>(token_bytes, idx * sizeof(uint32_t)));
        }
        break;
    }
    default:
        LOG(FATAL) << "Unsupported TinyShakespeare type";
    }

    // 构造后的逻辑形状：[(可取样本数 + 1), seq_len]。
    // 取样本 idx 时：x 从 idx 开始取 seq_len，y 从 idx+1 开始取 seq_len。
    const auto num_samples = num_tokens - sequence_length;
    text_file.dims = {static_cast<int64_t>(num_samples + 1), static_cast<int64_t>(sequence_length)};
    text_file.tensor = std::move(token_tensor);
    return text_file;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      // 每个样本窗口是 seq_len 个 int64 token。
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      // 最后一个 token 没有完整的 y 窗口，所以样本数是 dims[0]-1。
      num_samples_(static_cast<size_t>(text_file_.dims[0] - 1)) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    CHECK_EQ(text_file_.dims.size(), 2);
    CHECK_EQ(text_file_.dims[1], static_cast<int64_t>(sequence_length_));
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: 从 idx*seq_bytes 开始取 [idx, idx+seq_len)
    // y: 从 (idx*seq_bytes + sizeof(int64_t)) 开始取 [idx+1, idx+seq_len+1)
    // 这样天然形成 next-token prediction 对齐关系。
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
