#include "infini_train/include/optimizer.h"

#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &params) : params_(params) {}

void Optimizer::ZeroGrad() {
    for (auto param : params_) { param->ZeroGrad(); }
}

namespace optimizers {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate)
    : Optimizer(params), learning_rate_(learning_rate) {}

void SGD::Step() {
    for (auto param : params_) {
        auto device = param->GetDevice().Type();
        auto kernel = Dispatcher::Instance().GetKernel({device, "AccumulateGrad"});
        kernel.Call<void>(param->grad(), -learning_rate_, param);
    }
}

Adam::Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate, float beta1, float beta2, float eps)
    : Optimizer(params), t_(0), learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps) {

    for (const auto &param : params_) {
        m_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        v_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        m_.back()->Fill<float>(0.0f);
        v_.back()->Fill<float>(0.0f);
    }
}

void Adam::Step() {
    ++t_;

    for (size_t i = 0; i < params_.size(); ++i) {
        auto &param = params_[i];
        const auto &grad = param->grad();
        auto &m = m_[i];
        auto &v = v_[i];

        auto device = param->GetDevice().Type();
        auto kernel = Dispatcher::Instance().GetKernel({device, "AdamAccumulateGrad"});
        kernel.Call<void>(grad, param, m, v, learning_rate_, beta1_, beta2_, eps_, t_);
    }
}

void Adam::LoadState(int64_t step, const std::vector<std::shared_ptr<Tensor>> &first_moments,
                     const std::vector<std::shared_ptr<Tensor>> &second_moments) {
    CHECK_GE(step, 0);
    CHECK_EQ(first_moments.size(), params_.size());
    CHECK_EQ(second_moments.size(), params_.size());

    std::vector<std::shared_ptr<Tensor>> loaded_m;
    std::vector<std::shared_ptr<Tensor>> loaded_v;
    loaded_m.reserve(params_.size());
    loaded_v.reserve(params_.size());
    for (size_t idx = 0; idx < params_.size(); ++idx) {
        const auto &param = params_[idx];
        const auto &first = first_moments[idx];
        const auto &second = second_moments[idx];
        CHECK(first && second);
        CHECK(first->Dims() == param->Dims());
        CHECK(second->Dims() == param->Dims());
        CHECK_EQ(static_cast<int>(first->Dtype()), static_cast<int>(param->Dtype()));
        CHECK_EQ(static_cast<int>(second->Dtype()), static_cast<int>(param->Dtype()));
        loaded_m.push_back(std::make_shared<Tensor>(first->To(param->GetDevice())));
        loaded_v.push_back(std::make_shared<Tensor>(second->To(param->GetDevice())));
    }
    t_ = step;
    m_ = std::move(loaded_m);
    v_ = std::move(loaded_v);
}
} // namespace optimizers
} // namespace infini_train
