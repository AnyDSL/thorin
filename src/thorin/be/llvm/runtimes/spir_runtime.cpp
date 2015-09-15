#include "thorin/be/llvm/runtimes/spir_runtime.h"

#include "thorin/primop.h"
#include "thorin/be/llvm/llvm.h"

namespace thorin {

SPIRRuntime::SPIRRuntime(llvm::LLVMContext& context, llvm::Module& target, llvm::IRBuilder<>& builder)
    : KernelRuntime(context, target, builder, llvm::IntegerType::getInt64Ty(context), THORIN_RUNTIME_PLATFORMS "spir.s")
{}

llvm::Value* SPIRRuntime::malloc(llvm::Value* device, llvm::Value* ptr) {
    llvm::Value* malloc_args[] = { device, builder_.CreatePointerCast(ptr, builder_.getInt8PtrTy()) };
    auto device_mem = builder_.CreateCall(get("spir_malloc_buffer"), malloc_args);
    return device_mem;
}

llvm::Value* SPIRRuntime::free(llvm::Value* device, llvm::Value* mem) {
    llvm::Value* free_args[] = { device, mem };
    return builder_.CreateCall(get("spir_free_buffer"), free_args);
}

llvm::Value* SPIRRuntime::write(llvm::Value* device, llvm::Value* mem, llvm::Value* data) {
    llvm::Value* mem_args[] = { device, mem, builder_.CreatePointerCast(data, builder_.getInt8PtrTy()) };
    return builder_.CreateCall(get("spir_write_buffer"), mem_args);
}

llvm::Value* SPIRRuntime::read(llvm::Value* device, llvm::Value* mem, llvm::Value* data) {
    llvm::Value* args[] = { device, mem, builder_.CreatePointerCast(data, builder_.getInt8PtrTy()) };
    return builder_.CreateCall(get("spir_read_buffer"), args);
}

llvm::Value* SPIRRuntime::set_problem_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* problem_size_args[] = { device, x, y, z };
    return builder_.CreateCall(get("spir_set_problem_size"), problem_size_args);
}

llvm::Value* SPIRRuntime::set_config_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* config_args[] = { device, x, y, z };
    return builder_.CreateCall(get("spir_set_config_size"), config_args);
}

llvm::Value* SPIRRuntime::synchronize(llvm::Value* device) {
    return builder_.CreateCall(get("spir_synchronize"), { device });
}

llvm::Value* SPIRRuntime::set_kernel_arg(llvm::Value* device, llvm::Value* ptr, llvm::Type* type) {
    auto layout = target_.getDataLayout();
    llvm::Value* arg_args[] = { device, ptr, builder_.getInt32(layout->getTypeAllocSize(type)) };
    return builder_.CreateCall(get("spir_set_kernel_arg"), arg_args);
}

llvm::Value* SPIRRuntime::set_kernel_arg_map(llvm::Value* device, llvm::Value* mem) {
    llvm::Value* arg_args[] = { device, mem };
    return builder_.CreateCall(get("spir_set_kernel_arg_map"), arg_args);
}

llvm::Value* SPIRRuntime::set_kernel_arg_struct(llvm::Value* device, llvm::Value* ptr, llvm::Type* type) {
    auto layout = target_.getDataLayout();
    llvm::Value* arg_args[] = { device, ptr, builder_.getInt32(layout->getTypeAllocSize(type)) };
    return builder_.CreateCall(get("spir_set_kernel_arg_struct"), arg_args);
}

llvm::Value* SPIRRuntime::set_texture(llvm::Value* device, llvm::Value* mem, llvm::Value* name, PrimTypeKind type) {
    assert(false && "textures only supported in CUDA/NVVM");
    llvm::Value* tex_args[] = { device, mem, name, builder_.getInt32(0) };
    return builder_.CreateCall(get("spir_set_kernel_arg_tex"), tex_args);
}

llvm::Value* SPIRRuntime::load_kernel(llvm::Value* device, llvm::Value* file, llvm::Value* kernel) {
    llvm::Value* load_args[] = { device, file, kernel };
    return builder_.CreateCall(get("spir_build_program_and_kernel_from_binary"), load_args);
}

llvm::Value* SPIRRuntime::launch_kernel(llvm::Value* device, llvm::Value* name) {
    llvm::Value* arg_args[] = { device, name };
    return builder_.CreateCall(get("spir_launch_kernel"), arg_args);
}

std::string SPIRRuntime::get_module_name(Lambda* l) {
    return l->world().name() + ".spir.bc";
}

}
