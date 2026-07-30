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
     CHECK_GT(sequence_length, 0);

    // 检查文件是否存在
    if (!std::filesystem::exists(path)) {
        LOG(FATAL) << "File not found: " << path;
    }

    // 使用二进制方式打开文件
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << path;

    TinyShakespeareFile text_file;

    // 文件前 1024 字节是文件头
    const auto header =
        ReadSeveralBytesFromIfstream(1024, &ifs);

    CHECK(ifs.good())
        << "Failed to read dataset header: " << path;

    // 从文件头中解析三个整数
    const int32_t magic =
        BytesToType<int32_t>(header, 0);

    const int32_t version =
        BytesToType<int32_t>(header, 4);

    const int32_t num_tokens =
        BytesToType<int32_t>(header, 8);

    // 当前作业暂时不使用 version
    (void)version;

    CHECK_GT(num_tokens, 0);

    // 根据 magic 判断 token 原始数据类型
    CHECK(kTypeMap.contains(magic))
        << "Unsupported dataset magic number: " << magic;

    text_file.type = kTypeMap.at(magic);

    // 将全部 token 按 sequence_length 分组
    const int64_t num_sequences =
        static_cast<int64_t>(num_tokens / sequence_length);

    CHECK_GT(num_sequences, 0);

    text_file.dims = {
        num_sequences,
        static_cast<int64_t>(sequence_length)
    };

    const size_t num_values =
        static_cast<size_t>(num_sequences) * sequence_length;

    const size_t data_size_in_bytes =
        kTypeToSize.at(text_file.type) * num_values;

    /*
     * 文件中可能是 uint16 或 uint32，
     * 但模型内部统一使用 int64 保存 token_id。
     */
    text_file.tensor = infini_train::Tensor(
        text_file.dims,
        DataType::kINT64
    );

    auto *destination =
        static_cast<int64_t *>(text_file.tensor.DataPtr());

    // 根据文件类型创建对应的临时缓冲区
    std::variant<
        std::vector<uint16_t>,
        std::vector<int32_t>
    > buffer;

    if (text_file.type == TinyShakespeareType::kUINT16) {
        // GPT-2 最大序列长度
        CHECK_LE(sequence_length, 1024);

        buffer = std::vector<uint16_t>(num_values);
    } else if (
        text_file.type == TinyShakespeareType::kUINT32
    ) {
        // LLaMA 3 最大序列长度
        CHECK_LE(sequence_length, 8192);

        buffer = std::vector<int32_t>(num_values);
    }

    // 读取 token，并统一转换成 int64_t
    std::visit(
        [&](auto &tokens) {
            ifs.read(
                reinterpret_cast<char *>(tokens.data()),
                static_cast<std::streamsize>(
                    data_size_in_bytes
                )
            );

            CHECK(ifs.good() || ifs.eof())
                << "Failed to read dataset tokens";

            for (size_t i = 0; i < tokens.size(); ++i) {
                destination[i] =
                    static_cast<int64_t>(tokens[i]);
            }
        },
        buffer
    );
    return text_file;
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    : text_file_(
          ReadTinyShakespeareFile(
              filepath,
              sequence_length
          )
      ),
      sequence_length_(sequence_length),
      sequence_size_in_bytes_(
          sequence_length * sizeof(int64_t)
      ),
      num_samples_(text_file_.dims[0] - 1) {

    CHECK_EQ(
        text_file_.dims[1],
        static_cast<int64_t>(sequence_length_)
    );

    CHECK_EQ(
        static_cast<int>(text_file_.tensor.Dtype()),
        static_cast<int>(DataType::kINT64)
    );
    }
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
