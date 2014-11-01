#include "thorin/be/llvm/runtimes/cuda_runtime.h"

namespace thorin {

CUDARuntime::CUDARuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder)
    : NVVMRuntime(context, target, builder)
{}

llvm::Value* CUDARuntime::load_kernel(llvm::Value* device, llvm::Value* file, llvm::Value* kernel) {
    llvm::Value* load_args[] = { device, file, kernel };
    return builder_.CreateCall(get("nvvm_load_cuda_kernel"), load_args);
}

std::string CUDARuntime::get_module_name(Lambda* l) {
    return l->world().name() + ".cu";
}

}
