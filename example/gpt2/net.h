#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

struct GPT2Config {
    int64_t block_size = 1024;
    int64_t vocab_size = 50257;
    int64_t n_layer = 12;
    int64_t n_head = 12;
    int64_t n_embd = 768;
};

struct GPT2ForwardDiagnostics {
    bool capture_block_outputs = false;
    std::shared_ptr<infini_train::Tensor> transformer_input = nullptr;
    std::shared_ptr<infini_train::Tensor> final_block_output = nullptr;
    std::shared_ptr<infini_train::Tensor> ln_f_output = nullptr;
    std::shared_ptr<infini_train::Tensor> lm_head_input = nullptr;
    std::shared_ptr<infini_train::Tensor> logits = nullptr;
    std::vector<std::shared_ptr<infini_train::Tensor>> block_outputs;

    void Clear();
    void RecordBlockOutput(int64_t block_index, const std::shared_ptr<infini_train::Tensor> &output);
};

class NewGELU : public infini_train::nn::Module {
public:
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class CausalSelfAttention : public infini_train::nn::Module {
public:
    static constexpr char kCAttnLayerName[] = "c_attn";
    static constexpr char kCProjLayerName[] = "c_proj";

    explicit CausalSelfAttention(const GPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

    void To(infini_train::Device device) override;

protected:
    void To(infini_train::Device device,
            std::unordered_map<const infini_train::Tensor *, std::shared_ptr<infini_train::Tensor>> &moved_tensors)
        override;

private:
    GPT2Config config_;
    int64_t n_head_ = 0;
    int64_t n_embd_ = 0;
    std::shared_ptr<infini_train::Tensor> bias_ = nullptr;
};

class MLP : public infini_train::nn::Module {
public:
    static constexpr char kCFclayerName[] = "c_fc";
    static constexpr char kGeluLayerName[] = "gelu";
    static constexpr char kCProjLayerName[] = "c_proj";

    explicit MLP(const GPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class Block : public infini_train::nn::Module {
public:
    static constexpr char kLn1LayerName[] = "ln_1";
    static constexpr char kAttnLayerName[] = "attn";
    static constexpr char kLn2LayerName[] = "ln_2";
    static constexpr char kMlpLayerName[] = "mlp";

    explicit Block(const GPT2Config &config, int64_t diagnostic_index = -1,
                   GPT2ForwardDiagnostics **forward_diagnostics = nullptr);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    int64_t diagnostic_index_ = -1;
    GPT2ForwardDiagnostics **forward_diagnostics_ = nullptr;
};

class GPT2 : public infini_train::nn::Module {
public:
    static constexpr char kWTELayerName[] = "wte";
    static constexpr char kWPELayerName[] = "wpe";
    static constexpr char kHLayerName[] = "h";
    static constexpr char kLnFLayerName[] = "ln_f";
    static constexpr char kTransformerLayerName[] = "transformer";
    static constexpr char kLMHeadLayerName[] = "lm_head";

    enum class ModelType : int8_t {
        kGPT2,
        kGPT2Medium,
        kGPT2Large,
        kGPT2XL,
    };

    explicit GPT2(const GPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

    void SetForwardDiagnostics(GPT2ForwardDiagnostics *forward_diagnostics);

    static std::unique_ptr<GPT2> FromPretrained(ModelType model_type);
    static std::unique_ptr<GPT2> FromLLMC(const std::string &filepath);

private:
    GPT2Config config_;
    GPT2ForwardDiagnostics *forward_diagnostics_ = nullptr;
};
