#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class Sequential : public Module {
public:
    explicit Sequential(std::vector<std::unique_ptr<Module>> &&layers);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

class ModuleDict : public Module {
public:
    explicit ModuleDict(std::unordered_map<std::string, std::unique_ptr<Module>> &&modules);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};
} // namespace infini_train::nn
