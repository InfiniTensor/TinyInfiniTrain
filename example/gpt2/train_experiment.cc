#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#ifdef USE_CUDA
#include "cuda_runtime_api.h"
#endif

#include "example/common/tiny_shakespeare_dataset.h"
#include "example/common/tokenizer.h"
#include "example/gpt2/net.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

DEFINE_string(model_path, "", "Path to the llm.c GPT-2 checkpoint");
DEFINE_string(train_path, "", "Path to the complete training token file");
DEFINE_string(val_path, "", "Path to the complete validation token file");
DEFINE_string(tokenizer_path, "", "Path to the tokenizer file used for fixed-prompt generation");
DEFINE_string(output_dir, "experiment_output", "Directory for checkpoints, metrics, summaries, and generations");
DEFINE_string(device, "cuda", "Training device: cpu or cuda");
DEFINE_bool(resume, true, "Resume from output_dir/latest.ckpt when it exists");
DEFINE_uint64(batch_size, 2, "Batch size");
DEFINE_uint64(sequence_length, 64, "Sequence length");
DEFINE_uint64(max_epochs, 8, "Maximum number of complete train/validation epochs");
DEFINE_double(learning_rate, 2e-5, "Adam learning rate");
DEFINE_double(beta1, 0.9, "Adam beta1");
DEFINE_double(beta2, 0.999, "Adam beta2");
DEFINE_double(epsilon, 1e-8, "Adam epsilon");
DEFINE_uint64(early_stopping_patience, 3, "Validation epochs without improvement before stopping; 0 disables");
DEFINE_double(min_delta, 0.005, "Minimum validation-loss improvement");
DEFINE_uint64(seed, 1337, "Runner RNG seed saved in checkpoints");
DEFINE_uint64(generation_length, 64, "Fixed-prompt generation length (must not exceed sequence_length)");
DEFINE_double(max_runtime_seconds, 600.0, "Per-process runtime budget");
DEFINE_double(safety_margin_seconds, 90.0, "Reserved time for an atomic checkpoint and process shutdown");
DEFINE_uint64(checkpoint_interval_steps, 0, "Save latest every N train steps; 0 means epoch/window boundaries only");

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using infini_train::DataLoader;
using infini_train::DataType;
using infini_train::Device;
using infini_train::DeviceType;
using infini_train::Tensor;
using infini_train::Tokenizer;
using infini_train::nn::CrossEntropyLoss;
using infini_train::optimizers::Adam;

constexpr std::array<char, 8> kCheckpointMagic{'T', 'I', 'T', 'C', 'K', 'P', 'T', '\0'};
constexpr uint32_t kCheckpointVersion = 2;
constexpr uint32_t kEndianMarker = 0x01020304;
constexpr uint64_t kChecksumOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kChecksumPrime = 1099511628211ull;
constexpr char kStepsHeader[]
    = "epoch,global_step,batch_index,tokens,loss,compute_seconds,tokens_per_second,cuda_async_pool_peak_bytes";
constexpr char kEpochsHeader[]
    = "epoch,global_step,train_tokens,val_tokens,train_mean_loss,val_mean_loss,val_perplexity,val_perplexity_overflow,train_tokens_per_second,cuda_async_pool_peak_bytes,improved";

uint64_t UpdateChecksum(uint64_t checksum, const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    for (size_t idx = 0; idx < size; ++idx) {
        checksum ^= bytes[idx];
        checksum *= kChecksumPrime;
    }
    return checksum;
}

enum class Phase : uint32_t { kTrain = 0, kValidate = 1, kComplete = 2 };

struct Progress {
    Phase phase = Phase::kTrain;
    uint64_t epoch = 0;
    uint64_t global_step = 0;
    uint64_t next_batch = 0;
    double train_loss_sum = 0.0;
    uint64_t train_tokens = 0;
    double train_compute_seconds = 0.0;
    uint64_t total_train_tokens = 0;
    double total_train_compute_seconds = 0.0;
    double val_loss_sum = 0.0;
    uint64_t val_tokens = 0;
    double best_val_loss = std::numeric_limits<double>::infinity();
    uint64_t epochs_without_improvement = 0;
    uint64_t peak_cuda_bytes = 0;
    double final_train_loss = std::numeric_limits<double>::quiet_NaN();
    double final_val_loss = std::numeric_limits<double>::quiet_NaN();
};

struct StateEntry {
    std::string name;
    std::shared_ptr<Tensor> tensor;
    uint64_t payload_index = 0;
};

struct ModelCatalog {
    std::vector<StateEntry> entries;
    std::vector<std::string> canonical_names;
    std::vector<std::shared_ptr<Tensor>> unique_tensors;
};

struct FileIdentity {
    std::string path;
    uint64_t size = 0;
    uint64_t checksum = 0;
};

struct ExperimentContract {
    FileIdentity model;
    FileIdentity train;
    FileIdentity validation;
    FileIdentity tokenizer;
    uint64_t max_epochs = 0;
    uint64_t early_stopping_patience = 0;
    double min_delta = 0.0;
    uint64_t seed = 0;
    uint64_t generation_length = 0;
    std::string device;
};

struct DatasetCoverage {
    uint64_t samples = 0;
    uint64_t raw_tokens = 0;
    uint64_t usable_target_tokens = 0;
    uint64_t dropped_tokens = 0;
};

class BinaryWriter {
public:
    explicit BinaryWriter(const fs::path &path) : stream_(path, std::ios::binary | std::ios::trunc) {
        CHECK(stream_.is_open()) << "Failed to open checkpoint temporary file: " << path;
    }

    template <typename T> void Pod(const T &value) {
        static_assert(std::is_trivially_copyable_v<T>);
        Bytes(&value, sizeof(T));
    }

    void Bytes(const void *data, size_t size) {
        stream_.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
        CHECK(stream_) << "Checkpoint write failed";
        checksum_ = UpdateChecksum(checksum_, data, size);
    }

    void WriteChecksumTrailer() {
        stream_.write(reinterpret_cast<const char *>(&checksum_), sizeof(checksum_));
        CHECK(stream_) << "Checkpoint checksum trailer write failed";
    }

    void String(const std::string &value) {
        Pod(static_cast<uint64_t>(value.size()));
        Bytes(value.data(), value.size());
    }

    void Close() {
        stream_.flush();
        CHECK(stream_) << "Checkpoint flush failed";
        stream_.close();
        CHECK(!stream_.fail()) << "Checkpoint close failed";
    }

private:
    std::ofstream stream_;
    uint64_t checksum_ = kChecksumOffsetBasis;
};

class BinaryReader {
public:
    explicit BinaryReader(const fs::path &path) : stream_(path, std::ios::binary) {
        CHECK(stream_.is_open()) << "Failed to open checkpoint: " << path;
    }

    template <typename T> T Pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        Read(&value, sizeof(T));
        return value;
    }

    void Read(void *data, size_t size) {
        stream_.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
        CHECK_EQ(stream_.gcount(), static_cast<std::streamsize>(size)) << "Truncated checkpoint";
        checksum_ = UpdateChecksum(checksum_, data, size);
    }

    std::string String() {
        const uint64_t size = Pod<uint64_t>();
        CHECK_LE(size, static_cast<uint64_t>(std::numeric_limits<size_t>::max()));
        std::string value(static_cast<size_t>(size), '\0');
        Read(value.data(), value.size());
        return value;
    }

    void RequireChecksumTrailer() {
        uint64_t expected_checksum = 0;
        stream_.read(reinterpret_cast<char *>(&expected_checksum), sizeof(expected_checksum));
        CHECK_EQ(stream_.gcount(), static_cast<std::streamsize>(sizeof(expected_checksum)))
            << "Missing checkpoint checksum trailer";
        CHECK_EQ(checksum_, expected_checksum) << "Corrupt checkpoint body";
        CHECK_EQ(stream_.peek(), std::char_traits<char>::eof()) << "Unexpected trailing checkpoint data";
    }

private:
    std::ifstream stream_;
    uint64_t checksum_ = kChecksumOffsetBasis;
};

std::string JsonNumber(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

DatasetCoverage ReadDatasetCoverage(const fs::path &path, uint64_t samples) {
    std::ifstream input(path, std::ios::binary);
    CHECK(input.is_open()) << "Failed to read dataset coverage: " << path;
    input.seekg(2 * sizeof(uint32_t));
    uint32_t raw_tokens = 0;
    input.read(reinterpret_cast<char *>(&raw_tokens), sizeof(raw_tokens));
    CHECK_EQ(input.gcount(), static_cast<std::streamsize>(sizeof(raw_tokens))) << "Truncated dataset header";
    const uint64_t usable = samples * FLAGS_sequence_length;
    CHECK_GE(raw_tokens, usable + 1);
    return DatasetCoverage{.samples = samples,
                           .raw_tokens = raw_tokens,
                           .usable_target_tokens = usable,
                           .dropped_tokens = static_cast<uint64_t>(raw_tokens) - 1 - usable};
}

uint64_t ChecksumBytes(const void *data, size_t size) {
    return UpdateChecksum(kChecksumOffsetBasis, data, size);
}

FileIdentity IdentifyFile(const fs::path &path) {
    CHECK(fs::exists(path)) << "Required input file does not exist: " << path;
    CHECK(fs::is_regular_file(path)) << "Required input path is not a regular file: " << path;
    FileIdentity identity;
    identity.path = fs::weakly_canonical(path).string();
    identity.size = fs::file_size(path);

    std::ifstream input(path, std::ios::binary);
    CHECK(input.is_open()) << "Failed to hash input file: " << path;
    constexpr size_t kBufferSize = 1 << 20;
    std::vector<uint8_t> buffer(kBufferSize);
    uint64_t checksum = kChecksumOffsetBasis;
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        const size_t count = static_cast<size_t>(input.gcount());
        for (size_t idx = 0; idx < count; ++idx) {
            checksum ^= buffer[idx];
            checksum *= kChecksumPrime;
        }
    }
    CHECK(input.eof()) << "Failed while hashing input file: " << path;
    identity.checksum = checksum;
    return identity;
}

ExperimentContract BuildExperimentContract() {
    return ExperimentContract{
        .model = IdentifyFile(FLAGS_model_path),
        .train = IdentifyFile(FLAGS_train_path),
        .validation = IdentifyFile(FLAGS_val_path),
        .tokenizer = IdentifyFile(FLAGS_tokenizer_path),
        .max_epochs = FLAGS_max_epochs,
        .early_stopping_patience = FLAGS_early_stopping_patience,
        .min_delta = FLAGS_min_delta,
        .seed = FLAGS_seed,
        .generation_length = FLAGS_generation_length,
        .device = FLAGS_device,
    };
}

void WriteFileIdentity(BinaryWriter *writer, const FileIdentity &identity) {
    writer->String(identity.path);
    writer->Pod(identity.size);
    writer->Pod(identity.checksum);
}

void ValidateFileIdentity(BinaryReader *reader, const FileIdentity &expected) {
    CHECK_EQ(reader->String(), expected.path) << "Checkpoint input path mismatch";
    CHECK_EQ(reader->Pod<uint64_t>(), expected.size) << "Checkpoint input size mismatch";
    CHECK_EQ(reader->Pod<uint64_t>(), expected.checksum) << "Checkpoint input checksum mismatch";
}

void WriteExperimentContract(BinaryWriter *writer, const ExperimentContract &contract) {
    WriteFileIdentity(writer, contract.model);
    WriteFileIdentity(writer, contract.train);
    WriteFileIdentity(writer, contract.validation);
    WriteFileIdentity(writer, contract.tokenizer);
    writer->Pod(contract.max_epochs);
    writer->Pod(contract.early_stopping_patience);
    writer->Pod(contract.min_delta);
    writer->Pod(contract.seed);
    writer->Pod(contract.generation_length);
    writer->String(contract.device);
}

void ValidateExperimentContract(BinaryReader *reader, const ExperimentContract &expected) {
    ValidateFileIdentity(reader, expected.model);
    ValidateFileIdentity(reader, expected.train);
    ValidateFileIdentity(reader, expected.validation);
    ValidateFileIdentity(reader, expected.tokenizer);
    CHECK_EQ(reader->Pod<uint64_t>(), expected.max_epochs) << "Checkpoint max_epochs mismatch";
    CHECK_EQ(reader->Pod<uint64_t>(), expected.early_stopping_patience)
        << "Checkpoint early-stopping patience mismatch";
    CHECK_EQ(reader->Pod<double>(), expected.min_delta) << "Checkpoint min_delta mismatch";
    CHECK_EQ(reader->Pod<uint64_t>(), expected.seed) << "Checkpoint seed mismatch";
    CHECK_EQ(reader->Pod<uint64_t>(), expected.generation_length) << "Checkpoint generation length mismatch";
    CHECK_EQ(reader->String(), expected.device) << "Checkpoint device/backend mismatch";
}

fs::path TemporaryPath(const fs::path &destination) {
    const auto stamp = Clock::now().time_since_epoch().count();
    return destination.string() + ".tmp." + std::to_string(stamp);
}

void AtomicRename(const fs::path &temporary, const fs::path &destination) {
    std::error_code error;
    fs::rename(temporary, destination, error);
    CHECK(!error) << "Atomic rename failed from " << temporary << " to " << destination << ": " << error.message();
}

void AtomicWriteText(const fs::path &destination, const std::string &contents) {
    const fs::path temporary = TemporaryPath(destination);
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    CHECK(out.is_open()) << "Failed to open temporary output: " << temporary;
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    out.flush();
    CHECK(out) << "Failed to write temporary output: " << temporary;
    out.close();
    AtomicRename(temporary, destination);
}

void AtomicHardLink(const fs::path &source, const fs::path &destination) {
    CHECK(fs::exists(source)) << "Hard-link source does not exist: " << source;
    const fs::path temporary = TemporaryPath(destination);
    std::error_code error;
    fs::create_hard_link(source, temporary, error);
    CHECK(!error) << "Failed to create checkpoint hard link: " << error.message();
    AtomicRename(temporary, destination);
}

void AppendCsv(const fs::path &path, const std::string &header, const std::string &row) {
    const bool needs_header = !fs::exists(path) || fs::file_size(path) == 0;
    std::ofstream out(path, std::ios::app);
    CHECK(out.is_open()) << "Failed to open metrics CSV: " << path;
    if (needs_header) {
        out << header << '\n';
    }
    out << row << '\n';
    out.flush();
    CHECK(out) << "Failed to append metrics CSV: " << path;
}

std::pair<uint64_t, uint64_t> ParseCsvKey(const std::string &row) {
    const size_t first_comma = row.find(',');
    const size_t second_comma = row.find(',', first_comma == std::string::npos ? first_comma : first_comma + 1);
    CHECK(first_comma != std::string::npos && second_comma != std::string::npos) << "Malformed metrics row: " << row;
    size_t parsed = 0;
    const uint64_t first = std::stoull(row.substr(0, first_comma), &parsed);
    CHECK_EQ(parsed, first_comma) << "Malformed metrics key: " << row;
    const std::string second_text = row.substr(first_comma + 1, second_comma - first_comma - 1);
    const uint64_t second = std::stoull(second_text, &parsed);
    CHECK_EQ(parsed, second_text.size()) << "Malformed metrics key: " << row;
    return {first, second};
}

void ReconcileMetricsCsv(const fs::path &path, const std::string &expected_header, uint64_t maximum_key,
                         bool key_is_second_column, bool maximum_is_inclusive) {
    if (!fs::exists(path)) {
        return;
    }
    std::ifstream input(path);
    CHECK(input.is_open()) << "Failed to reconcile metrics: " << path;
    std::string header;
    CHECK(static_cast<bool>(std::getline(input, header))) << "Missing metrics header: " << path;
    CHECK_EQ(header, expected_header) << "Unexpected metrics schema: " << path;
    std::unordered_map<uint64_t, std::string> committed;
    std::string row;
    while (std::getline(input, row)) {
        if (row.empty()) {
            continue;
        }
        const auto [first, second] = ParseCsvKey(row);
        const uint64_t key = key_is_second_column ? second : first;
        if (maximum_is_inclusive ? key <= maximum_key : key < maximum_key) {
            committed[key] = row;
        }
    }
    CHECK(input.eof()) << "Failed while reconciling metrics: " << path;
    std::vector<uint64_t> keys;
    keys.reserve(committed.size());
    for (const auto &[key, _] : committed) {
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    std::ostringstream canonical;
    canonical << header << '\n';
    for (const uint64_t key : keys) {
        canonical << committed.at(key) << '\n';
    }
    AtomicWriteText(path, canonical.str());
}

ModelCatalog BuildModelCatalog(GPT2 &model) {
    auto state_dict = model.StateDict();
    CHECK(state_dict.contains("lm_head.weight"));
    CHECK(state_dict.contains("transformer.wte.weight"));
    CHECK_EQ(state_dict.at("lm_head.weight").get(), state_dict.at("transformer.wte.weight").get())
        << "GPT-2 token embedding and language-model head must remain tied after device transfer";
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> sorted(state_dict.begin(), state_dict.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    ModelCatalog catalog;
    std::unordered_map<const Tensor *, uint64_t> tensor_to_payload;
    for (auto &[name, tensor] : sorted) {
        CHECK(tensor) << "Null state tensor: " << name;
        auto [it, inserted] = tensor_to_payload.emplace(tensor.get(), catalog.unique_tensors.size());
        if (inserted) {
            catalog.unique_tensors.push_back(tensor);
            catalog.canonical_names.push_back(name);
        }
        catalog.entries.push_back(StateEntry{.name = name, .tensor = tensor, .payload_index = it->second});
    }
    CHECK(!catalog.entries.empty());
    return catalog;
}

std::vector<uint8_t> TensorBytes(const Tensor &tensor) {
    std::vector<uint8_t> bytes(tensor.SizeInBytes());
    if (tensor.GetDevice().Type() == DeviceType::kCPU) {
        std::memcpy(bytes.data(), tensor.DataPtr(), bytes.size());
    }
#ifdef USE_CUDA
    else if (tensor.GetDevice().Type() == DeviceType::kCUDA) {
        CHECK_EQ(cudaMemcpy(bytes.data(), tensor.DataPtr(), bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    }
#endif
    else {
        LOG(FATAL) << "Unsupported checkpoint tensor device";
    }
    return bytes;
}

void CopyBytesToTensor(const std::vector<uint8_t> &bytes, Tensor *tensor) {
    CHECK(tensor);
    CHECK_EQ(bytes.size(), tensor->SizeInBytes());
    if (tensor->GetDevice().Type() == DeviceType::kCPU) {
        std::memcpy(tensor->DataPtr(), bytes.data(), bytes.size());
    }
#ifdef USE_CUDA
    else if (tensor->GetDevice().Type() == DeviceType::kCUDA) {
        CHECK_EQ(cudaMemcpy(tensor->DataPtr(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice), cudaSuccess);
    }
#endif
    else {
        LOG(FATAL) << "Unsupported checkpoint tensor device";
    }
}

void WriteTensorMetadata(BinaryWriter *writer, const std::string &name, const Tensor &tensor) {
    writer->String(name);
    writer->Pod(static_cast<int32_t>(tensor.Dtype()));
    writer->Pod(static_cast<uint64_t>(tensor.Dims().size()));
    for (const int64_t dim : tensor.Dims()) {
        writer->Pod(dim);
    }
    writer->Pod(static_cast<uint64_t>(tensor.SizeInBytes()));
}

void ValidateTensorMetadata(BinaryReader *reader, const std::string &expected_name, const Tensor &tensor) {
    CHECK_EQ(reader->String(), expected_name) << "Checkpoint state name mismatch";
    CHECK_EQ(reader->Pod<int32_t>(), static_cast<int32_t>(tensor.Dtype())) << "Checkpoint dtype mismatch";
    const uint64_t num_dims = reader->Pod<uint64_t>();
    CHECK_EQ(num_dims, tensor.Dims().size()) << "Checkpoint rank mismatch";
    for (const int64_t expected_dim : tensor.Dims()) {
        CHECK_EQ(reader->Pod<int64_t>(), expected_dim) << "Checkpoint shape mismatch";
    }
    CHECK_EQ(reader->Pod<uint64_t>(), tensor.SizeInBytes()) << "Checkpoint byte-size mismatch";
}

std::string SerializeRng(const std::mt19937_64 &rng) {
    std::ostringstream out;
    out << rng;
    return out.str();
}

void DeserializeRng(const std::string &serialized, std::mt19937_64 *rng) {
    std::istringstream in(serialized);
    in >> *rng;
    CHECK(in) << "Invalid RNG checkpoint state";
}

void WriteConfig(BinaryWriter *writer, const GPT2Config &config) {
    writer->Pod(config.block_size);
    writer->Pod(config.vocab_size);
    writer->Pod(config.n_layer);
    writer->Pod(config.n_head);
    writer->Pod(config.n_embd);
}

void ValidateConfig(BinaryReader *reader, const GPT2Config &config) {
    CHECK_EQ(reader->Pod<int64_t>(), config.block_size);
    CHECK_EQ(reader->Pod<int64_t>(), config.vocab_size);
    CHECK_EQ(reader->Pod<int64_t>(), config.n_layer);
    CHECK_EQ(reader->Pod<int64_t>(), config.n_head);
    CHECK_EQ(reader->Pod<int64_t>(), config.n_embd);
}

void WriteProgress(BinaryWriter *writer, const Progress &progress) {
    writer->Pod(static_cast<uint32_t>(progress.phase));
    writer->Pod(progress.epoch);
    writer->Pod(progress.global_step);
    writer->Pod(progress.next_batch);
    writer->Pod(progress.train_loss_sum);
    writer->Pod(progress.train_tokens);
    writer->Pod(progress.train_compute_seconds);
    writer->Pod(progress.total_train_tokens);
    writer->Pod(progress.total_train_compute_seconds);
    writer->Pod(progress.val_loss_sum);
    writer->Pod(progress.val_tokens);
    writer->Pod(progress.best_val_loss);
    writer->Pod(progress.epochs_without_improvement);
    writer->Pod(progress.peak_cuda_bytes);
    writer->Pod(progress.final_train_loss);
    writer->Pod(progress.final_val_loss);
}

Progress ReadProgress(BinaryReader *reader) {
    Progress progress;
    const uint32_t phase = reader->Pod<uint32_t>();
    CHECK_LE(phase, static_cast<uint32_t>(Phase::kComplete));
    progress.phase = static_cast<Phase>(phase);
    progress.epoch = reader->Pod<uint64_t>();
    progress.global_step = reader->Pod<uint64_t>();
    progress.next_batch = reader->Pod<uint64_t>();
    progress.train_loss_sum = reader->Pod<double>();
    progress.train_tokens = reader->Pod<uint64_t>();
    progress.train_compute_seconds = reader->Pod<double>();
    progress.total_train_tokens = reader->Pod<uint64_t>();
    progress.total_train_compute_seconds = reader->Pod<double>();
    progress.val_loss_sum = reader->Pod<double>();
    progress.val_tokens = reader->Pod<uint64_t>();
    progress.best_val_loss = reader->Pod<double>();
    progress.epochs_without_improvement = reader->Pod<uint64_t>();
    progress.peak_cuda_bytes = reader->Pod<uint64_t>();
    progress.final_train_loss = reader->Pod<double>();
    progress.final_val_loss = reader->Pod<double>();
    return progress;
}

double SaveCheckpoint(const fs::path &destination, const GPT2 &model, const ModelCatalog &catalog, const Adam &optimizer,
                      const Progress &progress, const std::mt19937_64 &rng,
                      const ExperimentContract &experiment_contract) {
    const auto started = Clock::now();
#ifdef USE_CUDA
    if (!catalog.unique_tensors.empty() && catalog.unique_tensors[0]->GetDevice().Type() == DeviceType::kCUDA) {
        CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
#endif
    const fs::path temporary = TemporaryPath(destination);
    BinaryWriter writer(temporary);
    writer.Bytes(kCheckpointMagic.data(), kCheckpointMagic.size());
    writer.Pod(kCheckpointVersion);
    writer.Pod(kEndianMarker);
    WriteConfig(&writer, model.config());
    WriteExperimentContract(&writer, experiment_contract);
    writer.Pod(FLAGS_sequence_length);
    writer.Pod(FLAGS_batch_size);
    writer.Pod(optimizer.LearningRate());
    writer.Pod(optimizer.Beta1());
    writer.Pod(optimizer.Beta2());
    writer.Pod(optimizer.Epsilon());
    writer.Pod(optimizer.StepCount());
    WriteProgress(&writer, progress);
    writer.String(SerializeRng(rng));

    writer.Pod(static_cast<uint64_t>(catalog.entries.size()));
    for (const auto &entry : catalog.entries) {
        WriteTensorMetadata(&writer, entry.name, *entry.tensor);
        writer.Pod(entry.payload_index);
    }
    writer.Pod(static_cast<uint64_t>(catalog.unique_tensors.size()));
    for (size_t idx = 0; idx < catalog.unique_tensors.size(); ++idx) {
        writer.String(catalog.canonical_names[idx]);
        const auto bytes = TensorBytes(*catalog.unique_tensors[idx]);
        writer.Pod(static_cast<uint64_t>(bytes.size()));
        writer.Pod(ChecksumBytes(bytes.data(), bytes.size()));
        writer.Bytes(bytes.data(), bytes.size());
    }

    const auto &first = optimizer.FirstMoments();
    const auto &second = optimizer.SecondMoments();
    CHECK_EQ(first.size(), catalog.unique_tensors.size());
    CHECK_EQ(second.size(), catalog.unique_tensors.size());
    writer.Pod(static_cast<uint64_t>(first.size()));
    for (size_t idx = 0; idx < first.size(); ++idx) {
        const std::string first_name = "adam.m." + catalog.canonical_names[idx];
        const std::string second_name = "adam.v." + catalog.canonical_names[idx];
        WriteTensorMetadata(&writer, first_name, *first[idx]);
        const auto first_bytes = TensorBytes(*first[idx]);
        writer.Pod(ChecksumBytes(first_bytes.data(), first_bytes.size()));
        writer.Bytes(first_bytes.data(), first_bytes.size());
        WriteTensorMetadata(&writer, second_name, *second[idx]);
        const auto second_bytes = TensorBytes(*second[idx]);
        writer.Pod(ChecksumBytes(second_bytes.data(), second_bytes.size()));
        writer.Bytes(second_bytes.data(), second_bytes.size());
    }
    writer.WriteChecksumTrailer();
    writer.Close();
    AtomicRename(temporary, destination);
    return std::chrono::duration<double>(Clock::now() - started).count();
}

Progress LoadCheckpoint(const fs::path &path, const GPT2 &model, const ModelCatalog &catalog, Adam *optimizer,
                        std::mt19937_64 *rng, const ExperimentContract &experiment_contract) {
    BinaryReader reader(path);
    std::array<char, kCheckpointMagic.size()> magic{};
    reader.Read(magic.data(), magic.size());
    CHECK(magic == kCheckpointMagic) << "Invalid checkpoint magic";
    CHECK_EQ(reader.Pod<uint32_t>(), kCheckpointVersion) << "Unsupported checkpoint version";
    CHECK_EQ(reader.Pod<uint32_t>(), kEndianMarker) << "Checkpoint endianness mismatch";
    ValidateConfig(&reader, model.config());
    ValidateExperimentContract(&reader, experiment_contract);
    CHECK_EQ(reader.Pod<uint64_t>(), FLAGS_sequence_length);
    CHECK_EQ(reader.Pod<uint64_t>(), FLAGS_batch_size);
    CHECK_EQ(reader.Pod<float>(), optimizer->LearningRate());
    CHECK_EQ(reader.Pod<float>(), optimizer->Beta1());
    CHECK_EQ(reader.Pod<float>(), optimizer->Beta2());
    CHECK_EQ(reader.Pod<float>(), optimizer->Epsilon());
    const int64_t optimizer_step = reader.Pod<int64_t>();
    const Progress progress = ReadProgress(&reader);
    CHECK_EQ(optimizer_step, static_cast<int64_t>(progress.global_step))
        << "Checkpoint optimizer step and global step mismatch";
    DeserializeRng(reader.String(), rng);

    CHECK_EQ(reader.Pod<uint64_t>(), catalog.entries.size()) << "Checkpoint state count mismatch";
    for (const auto &entry : catalog.entries) {
        ValidateTensorMetadata(&reader, entry.name, *entry.tensor);
        CHECK_EQ(reader.Pod<uint64_t>(), entry.payload_index) << "Checkpoint tied-weight alias mismatch";
    }
    CHECK_EQ(reader.Pod<uint64_t>(), catalog.unique_tensors.size()) << "Checkpoint payload count mismatch";
    for (size_t idx = 0; idx < catalog.unique_tensors.size(); ++idx) {
        CHECK_EQ(reader.String(), catalog.canonical_names[idx]) << "Checkpoint canonical state name mismatch";
        CHECK_EQ(reader.Pod<uint64_t>(), catalog.unique_tensors[idx]->SizeInBytes());
        const uint64_t expected_checksum = reader.Pod<uint64_t>();
        std::vector<uint8_t> bytes(catalog.unique_tensors[idx]->SizeInBytes());
        reader.Read(bytes.data(), bytes.size());
        CHECK_EQ(ChecksumBytes(bytes.data(), bytes.size()), expected_checksum) << "Corrupt model tensor payload";
        CopyBytesToTensor(bytes, catalog.unique_tensors[idx].get());
    }

    CHECK_EQ(reader.Pod<uint64_t>(), catalog.unique_tensors.size()) << "Checkpoint optimizer state count mismatch";
    std::vector<std::shared_ptr<Tensor>> first;
    std::vector<std::shared_ptr<Tensor>> second;
    first.reserve(catalog.unique_tensors.size());
    second.reserve(catalog.unique_tensors.size());
    for (size_t idx = 0; idx < catalog.unique_tensors.size(); ++idx) {
        const auto &parameter = catalog.unique_tensors[idx];
        auto first_tensor = std::make_shared<Tensor>(parameter->Dims(), parameter->Dtype());
        auto second_tensor = std::make_shared<Tensor>(parameter->Dims(), parameter->Dtype());
        ValidateTensorMetadata(&reader, "adam.m." + catalog.canonical_names[idx], *first_tensor);
        const uint64_t first_checksum = reader.Pod<uint64_t>();
        std::vector<uint8_t> first_bytes(first_tensor->SizeInBytes());
        reader.Read(first_bytes.data(), first_bytes.size());
        CHECK_EQ(ChecksumBytes(first_bytes.data(), first_bytes.size()), first_checksum)
            << "Corrupt Adam first-moment payload";
        CopyBytesToTensor(first_bytes, first_tensor.get());
        ValidateTensorMetadata(&reader, "adam.v." + catalog.canonical_names[idx], *second_tensor);
        const uint64_t second_checksum = reader.Pod<uint64_t>();
        std::vector<uint8_t> second_bytes(second_tensor->SizeInBytes());
        reader.Read(second_bytes.data(), second_bytes.size());
        CHECK_EQ(ChecksumBytes(second_bytes.data(), second_bytes.size()), second_checksum)
            << "Corrupt Adam second-moment payload";
        CopyBytesToTensor(second_bytes, second_tensor.get());
        first.push_back(std::move(first_tensor));
        second.push_back(std::move(second_tensor));
    }
    reader.RequireChecksumTrailer();
    optimizer->LoadState(optimizer_step, first, second);
    return progress;
}

class DisableParameterGradGuard {
public:
    explicit DisableParameterGradGuard(const ModelCatalog &catalog) : parameters_(catalog.unique_tensors) {
        requires_grad_.reserve(parameters_.size());
        for (const auto &parameter : parameters_) {
            requires_grad_.push_back(parameter->requires_grad());
            parameter->set_requires_grad(false);
        }
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

uint64_t BatchCount(const TinyShakespeareDataset &dataset) {
    return (dataset.Size() + FLAGS_batch_size - 1) / FLAGS_batch_size;
}

infini_train::DataLoaderIterator IteratorAt(const DataLoader &loader, uint64_t batch_index) {
    auto iterator = loader.begin();
    for (uint64_t idx = 0; idx < batch_index; ++idx) {
        ++iterator;
    }
    return iterator;
}

float ReadScalar(const Tensor &tensor) {
    CHECK_EQ(tensor.NumElements(), 1);
    auto cpu = const_cast<Tensor &>(tensor).To(Device(DeviceType::kCPU, 0));
#ifdef USE_CUDA
    if (tensor.GetDevice().Type() == DeviceType::kCUDA) {
        CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
#endif
    return static_cast<const float *>(cpu.DataPtr())[0];
}

uint64_t PeakCudaBytes() {
#ifdef USE_CUDA
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) {
        uint64_t high = 0;
        if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &high) == cudaSuccess) {
            return high;
        }
    }
#endif
    return 0;
}

void ResetPeakCudaBytes() {
#ifdef USE_CUDA
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) {
        uint64_t zero = 0;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero);
    }
#endif
}

std::string GenerateFixedPrompt(Tokenizer &tokenizer, GPT2 &model, const ModelCatalog &catalog, const Device &device) {
    DisableParameterGradGuard no_grad(catalog);
    std::ostringstream captured;
    auto *previous = std::cout.rdbuf(captured.rdbuf());
    tokenizer.GenerateText(model, 1, static_cast<uint32_t>(FLAGS_sequence_length),
                           static_cast<uint32_t>(FLAGS_generation_length), device, FLAGS_seed);
    std::cout.rdbuf(previous);
    return captured.str();
}

std::string PhaseName(Phase phase) {
    switch (phase) {
    case Phase::kTrain: return "train";
    case Phase::kValidate: return "validate";
    case Phase::kComplete: return "complete";
    }
    return "invalid";
}

void WriteSummary(const fs::path &path, const Progress &progress, const GPT2Config &config,
                  const DatasetCoverage &train_coverage, const DatasetCoverage &val_coverage, bool resumed,
                  double wall_seconds, const std::string &run_state) {
    const double train_loss
        = progress.train_tokens == 0 ? progress.final_train_loss : progress.train_loss_sum / progress.train_tokens;
    const double val_loss = progress.val_tokens == 0 ? progress.final_val_loss : progress.val_loss_sum / progress.val_tokens;
    const double throughput = progress.total_train_compute_seconds > 0
                                  ? progress.total_train_tokens / progress.total_train_compute_seconds
                                  : 0.0;
    const bool perplexity_overflow
        = std::isfinite(val_loss) && val_loss > std::log(std::numeric_limits<double>::max());
    const double perplexity = std::isfinite(val_loss) && !perplexity_overflow
                                  ? std::exp(val_loss)
                                  : std::numeric_limits<double>::quiet_NaN();
    std::ostringstream out;
    out << "{\n"
        << "  \"status\": \"" << PhaseName(progress.phase) << "\",\n"
        << "  \"run_state\": \"" << run_state << "\",\n"
        << "  \"resumed\": " << (resumed ? "true" : "false") << ",\n"
        << "  \"device\": \"" << FLAGS_device << "\",\n"
        << "  \"epoch\": " << progress.epoch << ",\n"
        << "  \"global_step\": " << progress.global_step << ",\n"
        << "  \"next_batch\": " << progress.next_batch << ",\n"
        << "  \"train_mean_loss\": " << JsonNumber(train_loss) << ",\n"
        << "  \"validation_mean_loss\": " << JsonNumber(val_loss) << ",\n"
        << "  \"validation_perplexity\": " << JsonNumber(perplexity) << ",\n"
        << "  \"validation_perplexity_overflow\": " << (perplexity_overflow ? "true" : "false") << ",\n"
        << "  \"best_validation_loss\": " << JsonNumber(progress.best_val_loss) << ",\n"
        << "  \"train_tokens_per_second\": " << JsonNumber(throughput) << ",\n"
        << "  \"cuda_async_pool_peak_bytes\": " << progress.peak_cuda_bytes << ",\n"
        << "  \"wall_seconds_this_process\": " << JsonNumber(wall_seconds) << ",\n"
        << "  \"config\": {\"block_size\": " << config.block_size << ", \"vocab_size\": " << config.vocab_size
        << ", \"n_layer\": " << config.n_layer << ", \"n_head\": " << config.n_head
        << ", \"n_embd\": " << config.n_embd << ", \"batch_size\": " << FLAGS_batch_size
        << ", \"sequence_length\": " << FLAGS_sequence_length << ", \"max_epochs\": " << FLAGS_max_epochs
        << ", \"learning_rate\": " << JsonNumber(FLAGS_learning_rate) << ", \"seed\": " << FLAGS_seed << "},\n"
        << "  \"dataset_coverage\": {\n"
        << "    \"train\": {\"samples\": " << train_coverage.samples << ", \"raw_tokens\": "
        << train_coverage.raw_tokens << ", \"usable_target_tokens\": " << train_coverage.usable_target_tokens
        << ", \"dropped_tokens\": " << train_coverage.dropped_tokens << "},\n"
        << "    \"validation\": {\"samples\": " << val_coverage.samples << ", \"raw_tokens\": "
        << val_coverage.raw_tokens << ", \"usable_target_tokens\": " << val_coverage.usable_target_tokens
        << ", \"dropped_tokens\": " << val_coverage.dropped_tokens << "}\n"
        << "  },\n"
        << "  \"artifacts\": {\n"
        << "    \"steps_csv\": {\"path\": \"steps.csv\", \"exists\": "
        << (fs::exists(path.parent_path() / "steps.csv") ? "true" : "false") << "},\n"
        << "    \"epochs_csv\": {\"path\": \"epochs.csv\", \"exists\": "
        << (fs::exists(path.parent_path() / "epochs.csv") ? "true" : "false") << "},\n"
        << "    \"latest_checkpoint\": {\"path\": \"latest.ckpt\", \"exists\": "
        << (fs::exists(path.parent_path() / "latest.ckpt") ? "true" : "false") << "},\n"
        << "    \"best_checkpoint\": {\"path\": \"best.ckpt\", \"exists\": "
        << (fs::exists(path.parent_path() / "best.ckpt") ? "true" : "false") << "},\n"
        << "    \"final_checkpoint\": {\"path\": \"final.ckpt\", \"exists\": "
        << (fs::exists(path.parent_path() / "final.ckpt") ? "true" : "false") << "},\n"
        << "    \"baseline_generation\": {\"path\": \"baseline.txt\", \"exists\": "
        << (fs::exists(path.parent_path() / "baseline.txt") ? "true" : "false") << "},\n"
        << "    \"best_generation\": {\"path\": \"best.txt\", \"exists\": "
        << (fs::exists(path.parent_path() / "best.txt") ? "true" : "false") << "},\n"
        << "    \"final_generation\": {\"path\": \"final.txt\", \"exists\": "
        << (fs::exists(path.parent_path() / "final.txt") ? "true" : "false") << "}\n"
        << "  }\n"
        << "}\n";
    AtomicWriteText(path, out.str());
}

void ValidateFlags() {
    CHECK(!FLAGS_model_path.empty());
    CHECK(!FLAGS_train_path.empty());
    CHECK(!FLAGS_val_path.empty());
    CHECK(!FLAGS_tokenizer_path.empty());
    CHECK_GT(FLAGS_batch_size, 0);
    CHECK_GT(FLAGS_sequence_length, 0);
    CHECK_LE(FLAGS_sequence_length, std::numeric_limits<uint32_t>::max());
    CHECK_GT(FLAGS_max_epochs, 0);
    CHECK(std::isfinite(FLAGS_learning_rate));
    CHECK_GT(FLAGS_learning_rate, 0.0);
    CHECK_GT(static_cast<float>(FLAGS_learning_rate), 0.0f);
    CHECK(std::isfinite(FLAGS_beta1));
    CHECK_GE(FLAGS_beta1, 0.0);
    CHECK_LT(FLAGS_beta1, 1.0);
    CHECK(std::isfinite(FLAGS_beta2));
    CHECK_GE(FLAGS_beta2, 0.0);
    CHECK_LT(FLAGS_beta2, 1.0);
    CHECK(std::isfinite(FLAGS_epsilon));
    CHECK_GT(FLAGS_epsilon, 0.0);
    CHECK_GT(static_cast<float>(FLAGS_epsilon), 0.0f);
    CHECK(std::isfinite(FLAGS_min_delta));
    CHECK_GE(FLAGS_min_delta, 0.0);
    CHECK_GT(FLAGS_generation_length, 0);
    CHECK_LE(FLAGS_generation_length, FLAGS_sequence_length);
    CHECK(std::isfinite(FLAGS_max_runtime_seconds));
    CHECK_GT(FLAGS_max_runtime_seconds, 0.0);
    CHECK(std::isfinite(FLAGS_safety_margin_seconds));
    CHECK_GE(FLAGS_safety_margin_seconds, 0.0);
    CHECK_LT(FLAGS_safety_margin_seconds, FLAGS_max_runtime_seconds);
    CHECK(FLAGS_device == "cpu" || FLAGS_device == "cuda");
#ifndef USE_CUDA
    CHECK_EQ(FLAGS_device, "cpu") << "This binary was built without CUDA support";
#endif
}
} // namespace

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    ValidateFlags();

    const auto process_started = Clock::now();
    fs::create_directories(FLAGS_output_dir);
    const fs::path output_dir = FLAGS_output_dir;
    const fs::path latest_path = output_dir / "latest.ckpt";
    const fs::path best_path = output_dir / "best.ckpt";
    const fs::path summary_path = output_dir / "summary.json";
    const fs::path steps_path = output_dir / "steps.csv";
    const fs::path epochs_path = output_dir / "epochs.csv";

    Device device(DeviceType::kCPU, 0);
    if (FLAGS_device == "cuda") {
#ifdef USE_CUDA
        CHECK_EQ(cudaSetDevice(0), cudaSuccess);
        device = Device(DeviceType::kCUDA, 0);
#endif
    }

    auto model = GPT2::FromLLMC(FLAGS_model_path);
    model->To(device);
    model->TieWeights();
    CHECK_GE(model->config().block_size, static_cast<int64_t>(FLAGS_sequence_length));
    ModelCatalog catalog = BuildModelCatalog(*model);
    Adam optimizer(catalog.unique_tensors, static_cast<float>(FLAGS_learning_rate), static_cast<float>(FLAGS_beta1),
                   static_cast<float>(FLAGS_beta2), static_cast<float>(FLAGS_epsilon));
    CrossEntropyLoss loss_function;
    loss_function.To(device);
    auto train_dataset = std::make_shared<TinyShakespeareDataset>(FLAGS_train_path, FLAGS_sequence_length);
    auto val_dataset = std::make_shared<TinyShakespeareDataset>(FLAGS_val_path, FLAGS_sequence_length);
    const DatasetCoverage train_coverage = ReadDatasetCoverage(FLAGS_train_path, train_dataset->Size());
    const DatasetCoverage val_coverage = ReadDatasetCoverage(FLAGS_val_path, val_dataset->Size());
    DataLoader train_loader(train_dataset, FLAGS_batch_size);
    DataLoader val_loader(val_dataset, FLAGS_batch_size);
    Tokenizer tokenizer(FLAGS_tokenizer_path);
    const ExperimentContract experiment_contract = BuildExperimentContract();
    std::mt19937_64 rng(FLAGS_seed);
    Progress progress;
    const bool resumed = FLAGS_resume && fs::exists(latest_path);
    if (resumed) {
        progress = LoadCheckpoint(latest_path, *model, catalog, &optimizer, &rng, experiment_contract);
        CHECK(fs::exists(output_dir / "baseline.txt")) << "Baseline generation is missing for resumed experiment";
        ReconcileMetricsCsv(steps_path, kStepsHeader, progress.global_step, true, true);
        ReconcileMetricsCsv(epochs_path, kEpochsHeader, progress.epoch, false, false);
    } else {
        CHECK(!fs::exists(latest_path)) << "Existing checkpoint requires --resume=true or a new output_dir";
        CHECK(!fs::exists(steps_path) && !fs::exists(epochs_path))
            << "Existing metrics without a checkpoint require a new output_dir";
        AtomicWriteText(output_dir / "baseline.txt", GenerateFixedPrompt(tokenizer, *model, catalog, device));
    }

    ResetPeakCudaBytes();
    double last_checkpoint_seconds = 0.0;
    if (!resumed) {
        last_checkpoint_seconds
            = SaveCheckpoint(latest_path, *model, catalog, optimizer, progress, rng, experiment_contract);
    }

    const auto elapsed_seconds = [&]() { return std::chrono::duration<double>(Clock::now() - process_started).count(); };
    const auto window_exhausted = [&]() {
        return elapsed_seconds() + FLAGS_safety_margin_seconds + last_checkpoint_seconds >= FLAGS_max_runtime_seconds;
    };
    const auto save_latest = [&]() {
        progress.peak_cuda_bytes = std::max(progress.peak_cuda_bytes, PeakCudaBytes());
        last_checkpoint_seconds
            = SaveCheckpoint(latest_path, *model, catalog, optimizer, progress, rng, experiment_contract);
        WriteSummary(summary_path, progress, model->config(), train_coverage, val_coverage, resumed, elapsed_seconds(),
                     "running");
    };

    while (progress.phase != Phase::kComplete) {
        if (progress.epoch >= FLAGS_max_epochs) {
            progress.phase = Phase::kComplete;
            save_latest();
            break;
        }
        if (window_exhausted()) {
            save_latest();
            WriteSummary(summary_path, progress, model->config(), train_coverage, val_coverage, resumed,
                         elapsed_seconds(), "paused_runtime");
            LOG(INFO) << "Runtime window exhausted safely; resume from " << latest_path;
            return 0;
        }

        if (progress.phase == Phase::kTrain) {
            const uint64_t batches = BatchCount(*train_dataset);
            CHECK_LE(progress.next_batch, batches);
            auto iterator = IteratorAt(train_loader, progress.next_batch);
            while (progress.next_batch < batches) {
                auto [x_cpu, y_cpu] = *iterator;
                ++iterator;
                const uint64_t batch_index = progress.next_batch;
                const uint64_t tokens = x_cpu->NumElements();
                const auto compute_started = Clock::now();
                auto x = std::make_shared<Tensor>(x_cpu->To(device));
                auto y = std::make_shared<Tensor>(y_cpu->To(device));
                optimizer.ZeroGrad();
                auto logits = model->Forward({x})[0];
                auto loss = loss_function.Forward({logits, y})[0];
                const float loss_value = ReadScalar(*loss);
                CHECK(std::isfinite(loss_value)) << "Non-finite training loss at global step " << progress.global_step;
                loss->Backward();
                optimizer.Step();
#ifdef USE_CUDA
                if (device.Type() == DeviceType::kCUDA) {
                    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
                }
#endif
                const double compute_seconds = std::chrono::duration<double>(Clock::now() - compute_started).count();
                ++progress.global_step;
                ++progress.next_batch;
                progress.train_loss_sum += static_cast<double>(loss_value) * tokens;
                progress.train_tokens += tokens;
                progress.train_compute_seconds += compute_seconds;
                progress.total_train_tokens += tokens;
                progress.total_train_compute_seconds += compute_seconds;
                progress.peak_cuda_bytes = std::max(progress.peak_cuda_bytes, PeakCudaBytes());

                std::ostringstream row;
                row << progress.epoch << ',' << progress.global_step << ',' << batch_index << ',' << tokens << ','
                    << std::setprecision(9) << loss_value << ',' << compute_seconds << ',' << tokens / compute_seconds << ','
                    << progress.peak_cuda_bytes;
                AppendCsv(steps_path, kStepsHeader, row.str());

                if (FLAGS_checkpoint_interval_steps > 0
                    && progress.global_step % FLAGS_checkpoint_interval_steps == 0) {
                    save_latest();
                }
                if (window_exhausted()) {
                    save_latest();
                    WriteSummary(summary_path, progress, model->config(), train_coverage, val_coverage, resumed,
                                 elapsed_seconds(), "paused_runtime");
                    LOG(INFO) << "Runtime window exhausted safely during training";
                    return 0;
                }
            }
            progress.phase = Phase::kValidate;
            progress.next_batch = 0;
            save_latest();
            continue;
        }

        CHECK(progress.phase == Phase::kValidate);
        const uint64_t batches = BatchCount(*val_dataset);
        CHECK_LE(progress.next_batch, batches);
        auto iterator = IteratorAt(val_loader, progress.next_batch);
        {
            DisableParameterGradGuard no_grad(catalog);
            while (progress.next_batch < batches) {
                auto [x_cpu, y_cpu] = *iterator;
                ++iterator;
                auto x = std::make_shared<Tensor>(x_cpu->To(device));
                auto y = std::make_shared<Tensor>(y_cpu->To(device));
                auto logits = model->Forward({x})[0];
                auto loss = loss_function.Forward({logits, y})[0];
                const float loss_value = ReadScalar(*loss);
                CHECK(std::isfinite(loss_value))
                    << "Non-finite validation loss at epoch " << progress.epoch << ", batch " << progress.next_batch;
                const uint64_t tokens = x_cpu->NumElements();
                progress.val_loss_sum += static_cast<double>(loss_value) * tokens;
                progress.val_tokens += tokens;
                ++progress.next_batch;
                progress.peak_cuda_bytes = std::max(progress.peak_cuda_bytes, PeakCudaBytes());
                if (window_exhausted()) {
                    save_latest();
                    WriteSummary(summary_path, progress, model->config(), train_coverage, val_coverage, resumed,
                                 elapsed_seconds(), "paused_runtime");
                    LOG(INFO) << "Runtime window exhausted safely during validation";
                    return 0;
                }
            }
        }

        CHECK_GT(progress.train_tokens, 0);
        CHECK_GT(progress.val_tokens, 0);
        const double train_mean = progress.train_loss_sum / progress.train_tokens;
        const double val_mean = progress.val_loss_sum / progress.val_tokens;
        CHECK(std::isfinite(train_mean)) << "Non-finite epoch training loss";
        CHECK(std::isfinite(val_mean)) << "Non-finite epoch validation loss";
        const bool perplexity_overflow = val_mean > std::log(std::numeric_limits<double>::max());
        const double perplexity
            = perplexity_overflow ? std::numeric_limits<double>::infinity() : std::exp(val_mean);
        const double throughput = progress.train_tokens / progress.train_compute_seconds;
        const bool improved = val_mean + FLAGS_min_delta < progress.best_val_loss;
        if (improved) {
            progress.best_val_loss = val_mean;
            progress.epochs_without_improvement = 0;
        } else {
            ++progress.epochs_without_improvement;
        }

        std::ostringstream epoch_row;
        epoch_row << progress.epoch << ',' << progress.global_step << ',' << progress.train_tokens << ','
                  << progress.val_tokens << ',' << std::setprecision(12) << train_mean << ',' << val_mean << ','
                  << perplexity << ',' << (perplexity_overflow ? 1 : 0) << ',' << throughput << ','
                  << progress.peak_cuda_bytes << ',' << (improved ? 1 : 0);
        AppendCsv(epochs_path, kEpochsHeader, epoch_row.str());

        progress.final_train_loss = train_mean;
        progress.final_val_loss = val_mean;
        ++progress.epoch;
        progress.phase = Phase::kTrain;
        progress.next_batch = 0;
        progress.train_loss_sum = 0.0;
        progress.train_tokens = 0;
        progress.train_compute_seconds = 0.0;
        progress.val_loss_sum = 0.0;
        progress.val_tokens = 0;

        const bool early_stop = FLAGS_early_stopping_patience > 0
                                && progress.epochs_without_improvement >= FLAGS_early_stopping_patience;
        if (early_stop || progress.epoch >= FLAGS_max_epochs) {
            progress.phase = Phase::kComplete;
        }
        save_latest();
        if (improved) {
            AtomicHardLink(latest_path, best_path);
            AtomicWriteText(output_dir / "best.txt", GenerateFixedPrompt(tokenizer, *model, catalog, device));
        }
    }

    progress.phase = Phase::kComplete;
    AtomicWriteText(output_dir / "final.txt", GenerateFixedPrompt(tokenizer, *model, catalog, device));
    AtomicHardLink(latest_path, output_dir / "final.ckpt");
    WriteSummary(summary_path, progress, model->config(), train_coverage, val_coverage, resumed, elapsed_seconds(),
                 "complete");
    LOG(INFO) << "Experiment complete; summary: " << summary_path;
    return 0;
}
