#ifndef THORIN_BE_LLVM_RUNTIMES_SPIR_H
#define THORIN_BE_LLVM_RUNTIMES_SPIR_H

#include "thorin/be/llvm/runtime.h"

namespace thorin {

class SpirRuntime : public KernelRuntime {
public:
    SpirRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder);

    virtual llvm::Value* malloc(llvm::Value* device, llvm::Value* ptr);
    virtual llvm::Value* free(llvm::Value* device, llvm::Value* mem);

    virtual llvm::Value* write(llvm::Value* device, llvm::Value* mem, llvm::Value* data);
    virtual llvm::Value* read(llvm::Value* device, llvm::Value* mem, llvm::Value* data);

    virtual llvm::Value* set_problem_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z);
    virtual llvm::Value* set_config_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z);
    virtual llvm::Value* synchronize(llvm::Value* device);

    virtual llvm::Value* set_kernel_arg(llvm::Value* device, llvm::Value* ptr);
    virtual llvm::Value* set_kernel_arg_map(llvm::Value* device, llvm::Value* mem);
    virtual llvm::Value* set_texture(llvm::Value* device, llvm::Value* mem, llvm::Value* name, PrimTypeKind type);
    virtual llvm::Value* load_kernel(llvm::Value* device, llvm::Value* module, llvm::Value* data);
    virtual llvm::Value* launch_kernel(llvm::Value* device, llvm::Value* name);

protected:
    virtual std::string get_module_name(Lambda*);

private:
    llvm::Value* size_of_kernel_arg_;
};

}

#endif

