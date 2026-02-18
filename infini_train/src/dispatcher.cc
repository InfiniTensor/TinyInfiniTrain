#include "infini_train/include/dispatcher.h"

namespace infini_train {

Dispatcher &Dispatcher::Instance() {
    static Dispatcher instance;
    return instance;
}

const KernelFunction &Dispatcher::GetKernel(KeyT key) const {
    auto iter = key_to_kernel_map_.find(key);
    CHECK(iter != key_to_kernel_map_.end())
        << "Kernel not found: " << key.second << " on device: " << static_cast<int>(key.first);
    return iter->second;
}

// Dispatcher &Dispatcher::Instance() {
//     static Dispatcher instance;
//     return instance;
// }

// bool Dispatcher::RegisterImpl(const DispatchKey &key, const std::function<void()> &kernel) {
//     CHECK(dispatch_table_.find(key) == dispatch_table_.end())
//         << "Kernel already registered for key " << std::get<1>(key);
//     dispatch_table_.emplace(key, kernel);
//     return true;
// }

// std::function<void()> Dispatcher::GetImpl(const DispatchKey &key) const {
//     auto it = dispatch_table_.find(key);
//     CHECK(it != dispatch_table_.end()) << "No kernel registered for key " << std::get<1>(key);
//     return it->second;
// }

} // namespace infini_train
