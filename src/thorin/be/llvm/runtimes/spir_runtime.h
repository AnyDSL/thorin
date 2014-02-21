#ifndef THORIN_BE_LLVM_RUNTIMES_SPIR_H
#define THORIN_BE_LLVM_RUNTIMES_SPIR_H

#include "thorin/be/llvm/runtime.h"

namespace thorin {

class SpirRuntime : public KernelRuntime {
public:
    SpirRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder);

    virtual llvm::Value* malloc(llvm::Value* ptr);
    virtual llvm::Value* free(llvm::Value* ptr);

    virtual llvm::Value* write(llvm::Value* ptr, llvm::Value* data);
    virtual llvm::Value* read(llvm::Value* ptr, llvm::Value* data);

    virtual llvm::Value* set_problem_size(llvm::Value* x, llvm::Value* y, llvm::Value* z);
    virtual llvm::Value* set_config_size(llvm::Value* x, llvm::Value* y, llvm::Value* z);
    virtual llvm::Value* synchronize();

    virtual llvm::Value* set_kernel_arg(llvm::Value* ptr);
    virtual llvm::Value* set_mapped_kernel_arg(llvm::Value* ptr);
    virtual llvm::Value* load_kernel(llvm::Value* module, llvm::Value* data);
    virtual llvm::Value* launch_kernel(llvm::Value* name);

protected:
    virtual std::string get_module_name(Lambda*);

private:
    llvm::Value* size_of_kernel_arg_;
};

}

#endif


