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
        using FuncT = RetT (*)(ArgsT...);
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
        CHECK(!key_to_kernel_map_.contains(key)) << "Kernel already registered: " << key.second;
        key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};
} // namespace infini_train

// Works at both file scope (file-scope const) and function scope (local const).
// The trailing semicolon is included so callers at file scope don't need one;
// callers at function scope that already add ';' end up with ';;' which is harmless.
#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    [[maybe_unused]] const bool kKernelRegistered_##kernel_name =                                                      \
        (infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func), true);
