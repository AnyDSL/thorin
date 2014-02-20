#ifndef THORIN_BE_LLVM_RURNTIMES_SPIR_H
#define THORIN_BE_LLVM_RURNTIMES_SPIR_H

#include "thorin/be/llvm/runtime.h"

namespace thorin {

class SpirRuntime : public Runtime {
public:
    SpirRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<> &builder);

    virtual llvm::Value* malloc(llvm::Value* size);
    virtual llvm::CallInst* free(llvm::Value* ptr);

    virtual llvm::CallInst* write(llvm::Value* ptr, llvm::Value* data, llvm::Value* length);
    virtual llvm::CallInst* read(llvm::Value* ptr, llvm::Value* data, llvm::Value* length);

    virtual llvm::CallInst* set_problem_size(llvm::Value* x, llvm::Value* y, llvm::Value* z);
    virtual llvm::CallInst* set_config_size(llvm::Value* x, llvm::Value* y, llvm::Value* z);
    virtual llvm::CallInst* synchronize();

    virtual llvm::CallInst* set_kernel_arg(llvm::Value* ptr);
    virtual llvm::CallInst* load_kernel(llvm::Value* module, llvm::Value* data);
    virtual llvm::CallInst* launch_kernel(llvm::Value* name);

protected:
    virtual std::string get_module_name(Lambda*);

private:
    llvm::Value* size_of_kernel_arg_;
};

}

#endif


