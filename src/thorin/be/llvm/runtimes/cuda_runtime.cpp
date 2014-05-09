#include "thorin/be/llvm/runtimes/cuda_runtime.h"

namespace thorin {

CUDARuntime::CUDARuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder)
    : NVVMRuntime(context, target, builder)
{}

llvm::Value* CUDARuntime::load_kernel(llvm::Value* device, llvm::Value* module, llvm::Value* data) {
    llvm::Value* load_args[] = { device, module, data };
    return builder_.CreateCall(get("nvvm_load_kernel_from_source"), load_args);
}

std::string CUDARuntime::get_module_name(Lambda* l) {
    return l->world().name() + ".cu";
}

}
