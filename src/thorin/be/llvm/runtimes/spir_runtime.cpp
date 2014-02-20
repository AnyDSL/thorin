#include "thorin/be/llvm/runtimes/spir_runtime.h"
#include "thorin/be/llvm/llvm.h"
#include "thorin/literal.h"

namespace thorin {

SpirRuntime::SpirRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<> &builder)
    : Runtime(context, target, builder, llvm::IntegerType::getInt64Ty(context), "spir.s")
{
    auto *DL = new llvm::DataLayout(target);
    size_of_kernel_arg_ = builder_.getInt64(DL->getTypeAllocSize(llvm::Type::getInt8PtrTy(context)));
}

llvm::Value* SpirRuntime::malloc(llvm::Value* ptr) {
    auto alloca = builder_.CreateAlloca(get_device_ptr_ty());
    auto device_ptr = builder_.CreateCall(get("spir_malloc_buffer"), builder_.CreateBitCast(ptr, builder_.getInt8PtrTy()));
    builder_.CreateStore(device_ptr, alloca);
    return alloca;
}

llvm::CallInst* SpirRuntime::free(llvm::Value* ptr) {
    auto loaded_device_ptr = builder_.CreateLoad(ptr);
    return builder_.CreateCall(get("spir_free_buffer"), { loaded_device_ptr });
}

llvm::CallInst* SpirRuntime::write(llvm::Value* ptr, llvm::Value* data) {
    auto loaded_device_ptr = builder_.CreateLoad(ptr);
    llvm::Value* mem_args[] = { loaded_device_ptr, builder_.CreateBitCast(data, builder_.getInt8PtrTy()) };
    return builder_.CreateCall(get("spir_write_buffer"), mem_args);
}

llvm::CallInst* SpirRuntime::read(llvm::Value* ptr, llvm::Value* data) {
    auto loaded_device_ptr = builder_.CreateLoad(ptr);
    llvm::Value* args[] = { loaded_device_ptr, builder_.CreateBitCast(data, builder_.getInt8PtrTy()) };
    return builder_.CreateCall(get("spir_read_buffer"), args);
}

llvm::CallInst* SpirRuntime::set_problem_size(llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* problem_size_args[] = { x, y, z };
    return builder_.CreateCall(get("spir_set_problem_size"), problem_size_args);
}

llvm::CallInst* SpirRuntime::set_config_size(llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* config_args[] = { x, y, z };
    return builder_.CreateCall(get("spir_set_config_size"), config_args);
}

llvm::CallInst* SpirRuntime::synchronize() {
    return builder_.CreateCall(get("spir_synchronize"));
}

llvm::CallInst* SpirRuntime::set_kernel_arg(llvm::Value* ptr) {
    llvm::Value* arg_args[] = { ptr, size_of_kernel_arg_ };
    return builder_.CreateCall(get("spir_set_kernel_arg"), arg_args);
}

llvm::CallInst* SpirRuntime::load_kernel(llvm::Value* module, llvm::Value* data) {
    llvm::Value* load_args[] = { module, data };
    return builder_.CreateCall(get("spir_build_program_and_kernel_from_binary"), load_args);
}

llvm::CallInst* SpirRuntime::launch_kernel(llvm::Value* name) {
    return builder_.CreateCall(get("spir_launch_kernel"), { name });
}

std::string SpirRuntime::get_module_name(Lambda* l) {
    return l->world().name() + "_spir.bc";
}

}
