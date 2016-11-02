#ifndef THORIN_BE_LLVM_RUNTIME_H
#define THORIN_BE_LLVM_RUNTIME_H

#include <memory>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "thorin/world.h"

namespace thorin {

class CodeGen;

class Runtime {
public:
    Runtime(llvm::LLVMContext& context,
            llvm::Module& target,
            llvm::IRBuilder<>& builder);

    enum Platform {
        CPU_PLATFORM,
        CUDA_PLATFORM,
        OPENCL_PLATFORM
    };

    /// Emits a call to thorin_set_block_size.
    llvm::Value* set_block_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z);
    /// Emits a call to thorin_set_grid_size.
    llvm::Value* set_grid_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z);
    /// Emits a call to thorin_synchronize.
    llvm::Value* synchronize(llvm::Value* device);

    /// Emits a call to thorin_set_kernel_arg.
    llvm::Value* set_kernel_arg(llvm::Value* device, int arg, llvm::Value* ptr, llvm::Type* type);
    /// Emits a call to thorin_set_kernel_arg_ptr.
    llvm::Value* set_kernel_arg_ptr(llvm::Value* device, int arg, llvm::Value* ptr);
    /// Emits a call to thorin_set_kernel_arg_struct.
    llvm::Value* set_kernel_arg_struct(llvm::Value* device, int arg, llvm::Value* ptr, llvm::Type* type);
    /// Emits a call to thorin_load_kernel.
    llvm::Value* load_kernel(llvm::Value* device, llvm::Value* file, llvm::Value* kernel);
    /// Emits a call to thorin_launch_kernel.
    llvm::Value* launch_kernel(llvm::Value* device);

    /// Emits a call to thorin_parallel_for.
    llvm::Value* parallel_for(llvm::Value* num_threads, llvm::Value* lower, llvm::Value* upper,
                              llvm::Value* closure_ptr, llvm::Value* fun_ptr);
    /// Emits a call to thorin_spawn_thread.
    llvm::Value* spawn_thread(llvm::Value* closure_ptr, llvm::Value* fun_ptr);
    /// Emits a call to thorin_sync_thread.
    llvm::Value* sync_thread(llvm::Value* id);

    Continuation* emit_host_code(CodeGen &code_gen, Platform platform, Continuation* continuation);

protected:
    llvm::Function* get(const char* name);

    llvm::Module& target_;
    llvm::IRBuilder<>& builder_;
    const llvm::DataLayout& layout_;

    std::unique_ptr<llvm::Module> runtime_;
};

}

#endif

