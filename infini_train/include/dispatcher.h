#pragma once

#include <iostream>
#include <map>
#include <type_traits>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {
class KernelFunction {
public:
    template <typename FuncT> explicit KernelFunction(FuncT &&func) : func_ptr_(reinterpret_cast<void *>(func)) {}

    template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
        // =================================== 作业 ===================================
        // TODO：实现通用kernel调用接口
        // 功能描述：将存储的函数指针转换为指定类型并调用
        // =================================== 作业 ===================================
        // 将内部保存的通用指针恢复为具体函数指针类型，再执行调用。
        CHECK(func_ptr_ != nullptr) << "Kernel function pointer is null";
        using FuncT = RetT (*)(ArgsT...);
        auto kernel = reinterpret_cast<FuncT>(func_ptr_);
        return kernel(std::forward<ArgsT>(args)...);
    }

private:
    void *func_ptr_ = nullptr;
};

class Dispatcher {
public:
    using KeyT = std::pair<DeviceType, std::string>;

    static Dispatcher &Instance();

    const KernelFunction &GetKernel(KeyT key) const;

    template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
        // =================================== 作业 ===================================
        // TODO：实现kernel注册机制
        // 功能描述：将kernel函数与设备类型、名称绑定
        // =================================== 作业 ===================================
        // 重复注册直接报错，避免同一 key 被覆盖导致行为不确定。
        CHECK(key_to_kernel_map_.find(key) == key_to_kernel_map_.end())
            << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
        key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};
} // namespace infini_train

#define REGISTER_KERNEL_IMPL_CONCAT_INNER(x, y) x##y
#define REGISTER_KERNEL_IMPL_CONCAT(x, y) REGISTER_KERNEL_IMPL_CONCAT_INNER(x, y)

#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    static const bool REGISTER_KERNEL_IMPL_CONCAT(_kernel_registered_, __COUNTER__) = []() {                          \
        /* 利用静态初始化在程序启动阶段完成注册，调用方无需手动初始化。 */                                       \
        infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                           \
        return true;                                                                                                    \
    }();
