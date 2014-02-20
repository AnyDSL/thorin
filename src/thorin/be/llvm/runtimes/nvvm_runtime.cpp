#include "thorin/be/llvm/runtimes/nvvm_runtime.h"
#include "thorin/be/llvm/llvm.h"
#include "thorin/literal.h"

namespace thorin {

NVVMRuntime::NVVMRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<> &builder)
    : Runtime(context, target, builder, llvm::IntegerType::getInt64Ty(context), "nvvm.s")
{}

llvm::Value* NVVMRuntime::malloc(llvm::Value* ptr) {
    auto alloca = builder_.CreateAlloca(get_device_ptr_ty());
    auto device_ptr = builder_.CreateCall(get("nvvm_malloc_memory"), builder_.CreateBitCast(ptr, builder_.getInt8PtrTy()));
    builder_.CreateStore(device_ptr, alloca);
    return alloca;
}

llvm::CallInst* NVVMRuntime::free(llvm::Value* ptr) {
    auto loaded_device_ptr = builder_.CreateLoad(ptr);
    return builder_.CreateCall(get("nvvm_free_memory"), { loaded_device_ptr });
}

llvm::CallInst* NVVMRuntime::write(llvm::Value* ptr, llvm::Value* data) {
    auto loaded_device_ptr = builder_.CreateLoad(ptr);
    llvm::Value* mem_args[] = { loaded_device_ptr, builder_.CreateBitCast(data, builder_.getInt8PtrTy()) };
    return builder_.CreateCall(get("nvvm_write_memory"), mem_args);
}

llvm::CallInst* NVVMRuntime::read(llvm::Value* ptr, llvm::Value* data) {
    auto loaded_device_ptr = builder_.CreateLoad(ptr);
    llvm::Value* args[] = { loaded_device_ptr, builder_.CreateBitCast(data, builder_.getInt8PtrTy()) };
    return builder_.CreateCall(get("nvvm_read_memory"), args);
}

llvm::CallInst* NVVMRuntime::set_problem_size(llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* problem_size_args[] = { x, y, z };
    return builder_.CreateCall(get("nvvm_set_problem_size"), problem_size_args);
}

llvm::CallInst* NVVMRuntime::set_config_size(llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* config_args[] = { x, y, z };
    return builder_.CreateCall(get("nvvm_set_config_size"), config_args);
}

llvm::CallInst* NVVMRuntime::synchronize() {
    return builder_.CreateCall(get("nvvm_synchronize"));
}

llvm::CallInst* NVVMRuntime::set_kernel_arg(llvm::Value* ptr) {
    return builder_.CreateCall(get("nvvm_set_kernel_arg"), { ptr });
}

llvm::CallInst* NVVMRuntime::load_kernel(llvm::Value* module, llvm::Value* name) {
    llvm::Value* load_args[] = { module, name };
    return builder_.CreateCall(get("nvvm_load_kernel"), load_args);
}

llvm::CallInst* NVVMRuntime::launch_kernel(llvm::Value* name) {
    return builder_.CreateCall(get("nvvm_launch_kernel"), { name });
}

std::string NVVMRuntime::get_module_name(Lambda* l) {
    return l->world().name() + "_nvvm.ll";
}

}
