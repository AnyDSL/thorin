#ifndef THORIN_BE_LLVM_RUNTIMES_NVVM_H
#define THORIN_BE_LLVM_RUNTIMES_NVVM_H

#include "thorin/be/llvm/runtime.h"

namespace thorin {

class NVVMRuntime : public KernelRuntime {
public:
    NVVMRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder);

    virtual llvm::Value* malloc(llvm::Value* device, llvm::Value* ptr);
    virtual llvm::Value* free(llvm::Value* device, llvm::Value* ptr);

    virtual llvm::Value* write(llvm::Value* device, llvm::Value* ptr, llvm::Value* data);
    virtual llvm::Value* read(llvm::Value* device, llvm::Value* ptr, llvm::Value* data);

    virtual llvm::Value* set_problem_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z);
    virtual llvm::Value* set_config_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z);
    virtual llvm::Value* synchronize(llvm::Value* device);

    virtual llvm::Value* set_kernel_arg(llvm::Value* device, llvm::Value* ptr);
    virtual llvm::Value* set_mapped_kernel_arg(llvm::Value* device, llvm::Value* ptr);
    virtual llvm::Value* load_kernel(llvm::Value* device, llvm::Value* module, llvm::Value* name);
    virtual llvm::Value* launch_kernel(llvm::Value* device, llvm::Value* name);

protected:
    virtual std::string get_module_name(Lambda*);
};

}

#endif


