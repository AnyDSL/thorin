#ifndef THORIN_BE_LLVM_RUNTIMES_CUDA_H
#define THORIN_BE_LLVM_RUNTIMES_CUDA_H

#include "thorin/be/llvm/runtimes/nvvm_runtime.h"

namespace thorin {

class CUDARuntime : public NVVMRuntime {
public:
    CUDARuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder);

    virtual llvm::Value* load_kernel(llvm::Value* device, llvm::Value* module, llvm::Value* data);

protected:
    virtual std::string get_module_name(Lambda*);
};

}

#endif


