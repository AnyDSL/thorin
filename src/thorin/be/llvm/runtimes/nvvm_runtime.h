#ifndef THORIN_BE_LLVM_RURNTIMES_NVVM_H
#define THORIN_BE_LLVM_RURNTIMES_NVVM_H

#include "thorin/be/llvm/runtime.h"

namespace thorin {

class NVVMRuntime : public Runtime {
public:
    NVVMRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<> &builder);

    virtual llvm::Value* malloc(llvm::Value* size);
    virtual llvm::CallInst* free(llvm::Value* ptr);

    virtual llvm::CallInst* write(llvm::Value* ptr, llvm::Value* data, llvm::Value* length);
    virtual llvm::CallInst* read(llvm::Value* ptr, llvm::Value* data, llvm::Value* length);

    virtual llvm::CallInst* set_problem_size(llvm::Value* x, llvm::Value* y, llvm::Value* z);
    virtual llvm::CallInst* set_config_size(llvm::Value* x, llvm::Value* y, llvm::Value* z);
    virtual llvm::CallInst* synchronize();

    virtual llvm::CallInst* set_kernel_arg(llvm::Value* ptr);
    virtual llvm::CallInst* load_kernel(llvm::Value* module, llvm::Value* name);
    virtual llvm::CallInst* launch_kernel(llvm::Value* name);
};

}

#endif


