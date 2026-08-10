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

        using FuncT = RetT (*)(ArgsT...);
        // TODO: 实现函数调用逻辑
        return reinterpret_cast<FuncT>(func_ptr_)(args...);
    }

private:
    void *func_ptr_ = nullptr;
};

class Dispatcher {
public:
    using KeyT = std::pair<DeviceType, std::string>;

    static Dispatcher &Instance() {
        static Dispatcher instance;
        return instance;
    }

    const KernelFunction &GetKernel(KeyT key) const {
        CHECK(key_to_kernel_map_.contains(key))
            << "Kernel not found: " << key.second << " on device: " << static_cast<int>(key.first);
        return key_to_kernel_map_.at(key);
    }

    template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
        // =================================== 作业 ===================================
        // TODO：实现kernel注册机制
        // 功能描述：将kernel函数与设备类型、名称绑定
        // =================================== 作业 ===================================
        CHECK(!key_to_kernel_map_.contains(key))
            << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);  // Fix CR#L52
        key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};
} // namespace infini_train

#define INFINI_KERNEL_CONCAT_(a, b) a##b
#define INFINI_KERNEL_CONCAT(a, b) INFINI_KERNEL_CONCAT_(a, b)
#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    /* =================================== 作业 =================================== */                               \
    /* TODO：实现自动注册宏 */                                                                                 \
    /* 功能描述：在全局静态区注册kernel，避免显式初始化代码 */                                  \
    /* =================================== 作业 =================================== */                               \
    static const bool INFINI_KERNEL_CONCAT(kernel_reg_, __COUNTER__) =                                                 \
        (infini_train::Dispatcher::Instance().Register(                                                                \
             std::make_pair(device, #kernel_name), kernel_func),                                                       \
         true);
