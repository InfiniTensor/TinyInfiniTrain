#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>

#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/optimizer.h"

#include "example/common/tiny_shakespeare_dataset.h"
#include "example/common/tokenizer.h"
#include "example/gpt2/net.h"

namespace infini_train {

constexpr char kDeviceCPU[] = "cpu";
constexpr char kDeviceCUDA[] = "cuda";

class LogitsValidator {
public:
    /**
    * @brief 验证当前张量数据与参考文件中的logits是否匹配
    * 
    * @param logits 待验证的张量（设备不限，会自动转换为CPU）
    * @param filename 参考文件的二进制路径
    * @param tolerance 允许的数值绝对误差阈值（默认1e-4）
    * @return bool 验证结果（true表示所有采样点误差在允许范围内）
    * 
    * @note 二进制文件格式：
    * 1. 维度数量 (size_t, 8字节)
    * 2. 各维度值 (int64_t[num_dims], 8*num_dims字节)
    * 3. 张量数据 (float[num_elements], 4*num_elements字节) 
    * 
    * @warning 需确保：
    * - 输入张量内存有效
    * - 参考文件存在且格式正确
    * - 跨平台使用时注意字节序问题
    */
    static bool Validate(Tensor& logits, const std::string& filename, float tolerance = 1e-3) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile.is_open()) {
            LOG(ERROR) << "Failed to open reference file: " << filename;
            return false;
        }
        
        size_t num_dims;
        infile.read(reinterpret_cast<char*>(&num_dims), sizeof(size_t));
        std::vector<int64_t> ref_dims(num_dims);
        for (size_t i = 0; i < num_dims; i++) {
            infile.read(reinterpret_cast<char*>(&ref_dims[i]), sizeof(int64_t));
        }
        
        auto current_dims = logits.Dims();
        if (ref_dims != current_dims) {
            return false;
        }
        
        size_t num_elements = 1;
        for (auto dim : ref_dims) {
            num_elements *= dim;
        }
        std::vector<float> ref_data(num_elements);
        infile.read(reinterpret_cast<char*>(ref_data.data()), num_elements * sizeof(float));
        infile.close();
        
        auto cpu_logits = logits.To(Device(DeviceType::kCPU, 0));
        const float* current_data = static_cast<const float*>(cpu_logits.DataPtr());
        
        // 抽样比较策略
        const int sample_count = 100; // 抽取100个点进行比较
        std::vector<size_t> indices_to_check;

        for (int i = 0; i < sample_count; i++) {
            indices_to_check.push_back(i * num_elements / sample_count);
        }
        
        for (auto idx : indices_to_check) {
            float ref_val = ref_data[idx];
            float current_val = current_data[idx];
            float diff = std::abs(ref_val - current_val);
            
            if (diff > tolerance) {
                LOG(INFO) << "Logits mismatch at position " << idx 
                          << ": Reference=" << ref_val 
                          << ", Current=" << current_val
                          << ", Diff=" << diff;
                return false;
            }
        }
        
        LOG(INFO) << "Logits validation passed with " << sample_count << " samples";
        return true;
    }
};

// 测试类
class GPT2TrainingTest : public ::testing::Test {
protected:
    void SetUp() override {
        llmc_filepath = "../../Data/gpt2_124M.bin";
        input_bin = "../../Data/tinyshakespeare/tiny_shakespeare_train.bin";
        tokenizer_bin = "../../Data/gpt2_tokenizer.bin";
        logits_reference = "../../Data/gpt2_logits_reference.bin";

        device_flag = "cuda";
        model_name = "gpt2";
        batch_size = 2;
        sequence_length = 64;
        total_batch_size = 256;
        num_iteration = 10;    // 迭代次数
        text_length = 64;    // 生成文本长度
        learning_rate = 1e-4;    //学习率

        Initialize();

        LOG(INFO)<< "Initialize() finished!";
    }
    
    void Initialize() {
        if (device_flag == kDeviceCPU) {
            device = Device(DeviceType::kCPU, 0);
        } else {
            device = Device(DeviceType::kCUDA, 0);
        }

        model = GPT2::FromLLMC(llmc_filepath);
        model->To(device);

        train_loader = std::make_unique<DataLoader>(
            std::make_shared<TinyShakespeareDataset>(input_bin, sequence_length), batch_size);

        optimizer = std::make_unique<optimizers::SGD>(model->Parameters(), learning_rate);

        loss_fn = std::make_unique<nn::CrossEntropyLoss>();
        loss_fn->To(device);

        if (!tokenizer_bin.empty()) {
            tokenizer = std::make_unique<Tokenizer>(tokenizer_bin);
        }
    }
    
    void RunSingleStep() {
        auto train_iter = train_loader->begin();
        
        optimizer->ZeroGrad();
        float lossf = 0.0f;
        for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
            auto [x, y] = *train_iter;
            ++train_iter;
            x = std::make_shared<Tensor>(x->To(device));
            y = std::make_shared<Tensor>(y->To(device));

            auto outputs = model->Forward({x, y});

            logits = outputs[0];

            ASSERT_NE(logits, nullptr) << "First output is null";
            ASSERT_GT(logits->NumElements(), 0) << "Empty logits tensor";
            ASSERT_EQ(logits->Dims().size(), 3) << "Logits should be 3D (batch, seq, vocab)";
            ASSERT_NE(loss_fn, nullptr) << "Loss function not initialized!";

            auto loss = loss_fn->Forward({logits, y})[0];
            auto loss_cpu = loss->To(Device());
            lossf += static_cast<const float*>(loss_cpu.DataPtr())[0] / grad_accum_steps;

            loss->Backward();
        }
        optimizer->Step();
    }
    
    std::unique_ptr<GPT2> model;
    std::unique_ptr<DataLoader> train_loader;
    std::unique_ptr<optimizers::SGD> optimizer;
    std::unique_ptr<nn::CrossEntropyLoss> loss_fn;
    std::unique_ptr<Tokenizer> tokenizer;
    Device device;
    std::shared_ptr<Tensor> logits;
    int grad_accum_steps = 0;

    std::string llmc_filepath;
    std::string input_bin;
    std::string tokenizer_bin;
    std::string logits_reference;
    std::string device_flag;
    std::string model_name;
    int batch_size = 2;
    int sequence_length = 64;    
    int total_batch_size = 256;
    int num_iteration = 10;    // 迭代次数
    int text_length = 64;    // 生成文本长度
    int freq_generate_txt = 10;
    float learning_rate = 1e-4;
};

TEST_F(GPT2TrainingTest, LogitsConsistency) {
    const auto tokens_per_fwdbwd = batch_size * sequence_length;    // 梯度累积步数
    grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
    
    for (int step = 0; step < num_iteration + 1; ++step) {
        LOG(INFO)<<"epoch: " << step;
        // 执行训练
        RunSingleStep();

        /* tokenizer */
        if ((step + 1) % freq_generate_txt == 0) {
            if (!tokenizer) {
                continue;
            }
            tokenizer->GenerateText(*model, batch_size, sequence_length, text_length, device);
        }
    }

    // 验证 logits
    if (!logits_reference.empty()) {
        bool validation_passed = LogitsValidator::Validate(*logits, logits_reference);
        EXPECT_TRUE(validation_passed) << "Logits validation failed!";
    } else {
        FAIL() << "No reference logits provided! Cannot validate.";        
    }
}
} // namespace infini_train
