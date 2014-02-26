#include "thorin/be/llvm/runtimes/spir_runtime.h"
#include "thorin/be/llvm/llvm.h"
#include "thorin/literal.h"

namespace thorin {

SpirRuntime::SpirRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder)
    : KernelRuntime(context, target, builder, llvm::IntegerType::getInt64Ty(context), "spir.s")
{
    AutoPtr<llvm::DataLayout> dl(new llvm::DataLayout(target));
    size_of_kernel_arg_ = builder_.getInt64(dl->getTypeAllocSize(llvm::Type::getInt8PtrTy(context)));
}

llvm::Value* SpirRuntime::malloc(llvm::Value* device, llvm::Value* ptr) {
    auto alloca = builder_.CreateAlloca(get_device_ptr_ty());
    llvm::Value* malloc_args[] = { device, builder_.CreateBitCast(ptr, builder_.getInt8PtrTy()) };
    auto device_ptr = builder_.CreateCall(get("spir_malloc_buffer"), malloc_args);
    builder_.CreateStore(device_ptr, alloca);
    return alloca;
}

llvm::Value* SpirRuntime::free(llvm::Value* device, llvm::Value* ptr) {
    auto loaded_device_ptr = builder_.CreateLoad(ptr);
    llvm::Value* free_args[] = { device, loaded_device_ptr };
    return builder_.CreateCall(get("spir_free_buffer"), { loaded_device_ptr });
}

llvm::Value* SpirRuntime::write(llvm::Value* device, llvm::Value* ptr, llvm::Value* data) {
    auto loaded_device_ptr = builder_.CreateLoad(ptr);
    llvm::Value* mem_args[] = { device, loaded_device_ptr, builder_.CreateBitCast(data, builder_.getInt8PtrTy()) };
    return builder_.CreateCall(get("spir_write_buffer"), mem_args);
}

llvm::Value* SpirRuntime::read(llvm::Value* device, llvm::Value* ptr, llvm::Value* data) {
    auto loaded_device_ptr = builder_.CreateLoad(ptr);
    llvm::Value* args[] = { device, loaded_device_ptr, builder_.CreateBitCast(data, builder_.getInt8PtrTy()) };
    return builder_.CreateCall(get("spir_read_buffer"), args);
}

llvm::Value* SpirRuntime::set_problem_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* problem_size_args[] = { device, x, y, z };
    return builder_.CreateCall(get("spir_set_problem_size"), problem_size_args);
}

llvm::Value* SpirRuntime::set_config_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* config_args[] = { device, x, y, z };
    return builder_.CreateCall(get("spir_set_config_size"), config_args);
}

llvm::Value* SpirRuntime::synchronize(llvm::Value* device) {
    return builder_.CreateCall(get("spir_synchronize"), { device });
}

llvm::Value* SpirRuntime::set_kernel_arg(llvm::Value* device, llvm::Value* ptr) {
    llvm::Value* arg_args[] = { device, ptr, size_of_kernel_arg_ };
    return builder_.CreateCall(get("spir_set_kernel_arg"), arg_args);
}

llvm::Value* SpirRuntime::set_mapped_kernel_arg(llvm::Value* device, llvm::Value* ptr) {
    llvm::Value* arg_args[] = { device, ptr };
    return builder_.CreateCall(get("spir_set_mapped_kernel_arg"), arg_args);
}

llvm::Value* SpirRuntime::load_kernel(llvm::Value* device, llvm::Value* module, llvm::Value* data) {
    llvm::Value* load_args[] = { device, module, data };
    return builder_.CreateCall(get("spir_build_program_and_kernel_from_binary"), load_args);
}

llvm::Value* SpirRuntime::launch_kernel(llvm::Value* device, llvm::Value* name) {
    llvm::Value* arg_args[] = { device, name };
    return builder_.CreateCall(get("spir_launch_kernel"), arg_args);
}

std::string SpirRuntime::get_module_name(Lambda* l) {
    return l->world().name() + ".spir.bc";
}

}
