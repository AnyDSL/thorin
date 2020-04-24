#ifndef THORIN_BE_LLVM_RUNTIME_H
#define THORIN_BE_LLVM_RUNTIME_H

#include <memory>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "thorin/world.h"

namespace thorin {

class CodeGen;

struct LaunchArgs {
    enum {
        Mem = 0,
        Device,
        Space,
        Config,
        Body,
        Return,
        Num
    };
};

class Runtime {
public:
    Runtime(llvm::LLVMContext& context,
            llvm::Module& target,
            llvm::IRBuilder<>& builder);

    enum Platform {
        CPU_PLATFORM,
        CUDA_PLATFORM,
        OPENCL_PLATFORM,
        HSA_PLATFORM
    };

    /// Emits a call to anydsl_launch_kernel.
    llvm::Value* launch_kernel(llvm::Value* device,
                               llvm::Value* file, llvm::Value* kernel,
                               llvm::Value* grid, llvm::Value* block,
                               llvm::Value* args, llvm::Value* sizes, llvm::Value* aligns, llvm::Value* types,
                               llvm::Value* num_args);

    /// Emits a call to anydsl_parallel_for.
    llvm::Value* parallel_for(llvm::Value* num_threads, llvm::Value* lower, llvm::Value* upper,
                              llvm::Value* closure_ptr, llvm::Value* fun_ptr);
    /// Emits a call to anydsl_fibers_spawn.
    llvm::Value* spawn_fibers(llvm::Value* num_threads, llvm::Value* num_blocks, llvm::Value* num_warps,
                              llvm::Value* closure_ptr, llvm::Value* fun_ptr);
    /// Emits a call to anydsl_spawn_thread.
    llvm::Value* spawn_thread(llvm::Value* closure_ptr, llvm::Value* fun_ptr);
    /// Emits a call to anydsl_sync_thread.
    llvm::Value* sync_thread(llvm::Value* id);

    Continuation* emit_host_code(CodeGen& code_gen,
                                 Platform platform,
                                 const std::string& ext,
                                 Continuation* continuation);

    llvm::Function* get(const char* name);

protected:
    llvm::Module& target_;
    llvm::IRBuilder<>& builder_;
    const llvm::DataLayout& layout_;

    std::unique_ptr<llvm::Module> runtime_;
};

}

#endif

