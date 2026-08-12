#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <typeindex>
#include <type_traits>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {
class KernelFunction {
public:
    template <typename FuncT> explicit KernelFunction(FuncT &&func) {
        using DecayedFuncT = std::decay_t<FuncT>;
        static_assert(std::is_pointer_v<DecayedFuncT> && std::is_function_v<std::remove_pointer_t<DecayedFuncT>>,
                      "KernelFunction only supports function pointers");
        invoker_ = std::make_shared<TypedInvoker<DecayedFuncT>>(std::forward<FuncT>(func));
    }

    template <typename RetT, class... ArgsT> RetT Call(ArgsT &&...args) const {
        static_assert(!std::is_reference_v<RetT>, "KernelFunction does not support reference return types");

        std::array<ErasedArgument, sizeof...(ArgsT)> erased_args{MakeErasedArgument<ArgsT>(args)...};
        CHECK(invoker_ != nullptr);
        CHECK(invoker_->return_type == std::type_index(typeid(RetT))) << "Kernel return type mismatch";
        CHECK_EQ(invoker_->argument_types.size(), erased_args.size()) << "Kernel argument count mismatch";
        for (size_t idx = 0; idx < erased_args.size(); ++idx) {
            CHECK(invoker_->argument_types[idx] == erased_args[idx].type) << "Kernel argument type mismatch at " << idx;
        }

        if constexpr (std::is_void_v<RetT>) {
            invoker_->Invoke(nullptr, erased_args);
        } else {
            std::optional<RetT> result;
            invoker_->Invoke(&result, erased_args);
            CHECK(result.has_value());
            return std::move(*result);
        }
    }

private:
    struct ErasedArgument {
        const void *value = nullptr;
        std::type_index type = typeid(void);
        bool is_lvalue = false;
        bool is_const = false;
    };

    struct Invoker {
        Invoker(std::type_index return_type, std::vector<std::type_index> argument_types)
            : return_type(return_type), argument_types(std::move(argument_types)) {}
        virtual ~Invoker() = default;

        virtual void Invoke(void *result, std::span<ErasedArgument> args) const = 0;

        const std::type_index return_type;
        const std::vector<std::type_index> argument_types;
    };

    template <typename FuncT> class TypedInvoker;

    template <typename RetT, typename... ParamsT> class TypedInvoker<RetT (*)(ParamsT...)> final : public Invoker {
    public:
        explicit TypedInvoker(RetT (*func)(ParamsT...))
            : Invoker(typeid(RetT), {std::type_index(typeid(std::remove_cvref_t<ParamsT>))...}), func_(func) {}

        void Invoke(void *result, std::span<ErasedArgument> args) const override {
            CHECK_EQ(args.size(), sizeof...(ParamsT));
            InvokeImpl(result, args, std::index_sequence_for<ParamsT...>{});
        }

    private:
        template <typename ParamT> static decltype(auto) Unpack(ErasedArgument &arg) {
            using BareT = std::remove_cvref_t<ParamT>;
            using ReferredT = std::remove_reference_t<ParamT>;

            if constexpr (std::is_lvalue_reference_v<ParamT>) {
                if constexpr (std::is_const_v<ReferredT>) {
                    return *static_cast<const BareT *>(arg.value);
                } else {
                    CHECK(arg.is_lvalue) << "Kernel argument must be an lvalue";
                    CHECK(!arg.is_const) << "Kernel argument must be mutable";
                    return *static_cast<BareT *>(const_cast<void *>(arg.value));
                }
            } else if constexpr (std::is_rvalue_reference_v<ParamT>) {
                CHECK(!arg.is_lvalue) << "Kernel argument must be an rvalue";
                if constexpr (std::is_const_v<ReferredT>) {
                    return std::move(*static_cast<const BareT *>(arg.value));
                } else {
                    CHECK(!arg.is_const) << "Kernel argument must be mutable";
                    return std::move(*static_cast<BareT *>(const_cast<void *>(arg.value)));
                }
            } else {
                static_assert(std::is_copy_constructible_v<BareT> || std::is_move_constructible_v<BareT>,
                              "By-value kernel arguments must be copyable or movable");
                if constexpr (std::is_copy_constructible_v<BareT>) {
                    return arg.is_lvalue || arg.is_const
                               ? BareT(*static_cast<const BareT *>(arg.value))
                               : BareT(std::move(*static_cast<BareT *>(const_cast<void *>(arg.value))));
                } else {
                    CHECK(!arg.is_lvalue && !arg.is_const) << "Move-only kernel argument must be a mutable rvalue";
                    return BareT(std::move(*static_cast<BareT *>(const_cast<void *>(arg.value))));
                }
            }
        }

        template <size_t... IndicesT>
        void InvokeImpl(void *result, std::span<ErasedArgument> args, std::index_sequence<IndicesT...>) const {
            if constexpr (std::is_void_v<RetT>) {
                std::invoke(func_, Unpack<ParamsT>(args[IndicesT])...);
            } else {
                auto &typed_result = *static_cast<std::optional<RetT> *>(result);
                typed_result.emplace(std::invoke(func_, Unpack<ParamsT>(args[IndicesT])...));
            }
        }

        RetT (*func_)(ParamsT...);
    };

    template <typename ArgT> static ErasedArgument MakeErasedArgument(std::remove_reference_t<ArgT> &arg) {
        return ErasedArgument{.value = std::addressof(arg),
                              .type = std::type_index(typeid(std::remove_cvref_t<ArgT>)),
                              .is_lvalue = std::is_lvalue_reference_v<ArgT>,
                              .is_const = std::is_const_v<std::remove_reference_t<ArgT>>};
    }

    std::shared_ptr<const Invoker> invoker_;
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
        CHECK(!key_to_kernel_map_.contains(key))
            << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
        key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
    }

private:
    std::map<KeyT, KernelFunction> key_to_kernel_map_;
};
} // namespace infini_train

#define INFINI_TRAIN_CONCAT_IMPL(lhs, rhs) lhs##rhs
#define INFINI_TRAIN_CONCAT(lhs, rhs) INFINI_TRAIN_CONCAT_IMPL(lhs, rhs)

#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                               \
    [[maybe_unused]] static const bool INFINI_TRAIN_CONCAT(infini_train_kernel_registered_, __COUNTER__) = []() {        \
        ::infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                            \
        return true;                                                                                                     \
    }();
