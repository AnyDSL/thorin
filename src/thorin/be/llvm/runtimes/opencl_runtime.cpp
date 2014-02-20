#include "thorin/be/llvm/runtimes/opencl_runtime.h"

namespace thorin {

OpenCLRuntime::OpenCLRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<> &builder)
    : SpirRuntime(context, target, builder)
{}

llvm::CallInst* OpenCLRuntime::load_kernel(llvm::Value* module, llvm::Value* data) {
    llvm::Value* load_args[] = { module, data };
    return builder_.CreateCall(get("spir_build_program_and_kernel_from_source"), load_args);
}

}
