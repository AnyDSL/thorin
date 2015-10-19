#include "thorin/be/llvm/runtimes/nvvm_runtime.h"

#include "thorin/primop.h"
#include "thorin/be/llvm/llvm.h"

namespace thorin {

NVVMRuntime::NVVMRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder)
    : KernelRuntime(context, target, builder, llvm::IntegerType::getInt64Ty(context), THORIN_RUNTIME_PLATFORMS "runtime_defs.s")
{}

llvm::Value* NVVMRuntime::malloc(llvm::Value* device, llvm::Value* ptr) {
    return builder_.CreatePointerCast(ptr, builder_.getInt8PtrTy());
}

llvm::Value* NVVMRuntime::free(llvm::Value* device, llvm::Value* mem) {
    return nullptr;
}

llvm::Value* NVVMRuntime::write(llvm::Value* device, llvm::Value* mem, llvm::Value* data) {
    return nullptr;
}

llvm::Value* NVVMRuntime::read(llvm::Value* device, llvm::Value* mem, llvm::Value* data) {
    return nullptr;
}

llvm::Value* NVVMRuntime::set_problem_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* grid_size_args[] = { builder_.getInt32(1), device, x, y, z };
    return builder_.CreateCall(get("thorin_set_grid_size"), grid_size_args);
}

llvm::Value* NVVMRuntime::set_config_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* block_size_args[] = { builder_.getInt32(1), device, x, y, z };
    return builder_.CreateCall(get("thorin_set_block_size"), block_size_args);
}

llvm::Value* NVVMRuntime::synchronize(llvm::Value* device) {
    llvm::Value* sync_args[] = { builder_.getInt32(1), device };
    return builder_.CreateCall(get("thorin_synchronize"), sync_args);
}

llvm::Value* NVVMRuntime::set_kernel_arg(llvm::Value* device, llvm::Value* ptr, llvm::Type* type) {
    return nullptr;
}

llvm::Value* NVVMRuntime::set_kernel_arg_map(llvm::Value* device, llvm::Value* mem) {
    llvm::Value* arg_args[] = { builder_.getInt32(1), device, builder_.getInt32(0), mem, builder_.getInt32(0) };
    return builder_.CreateCall(get("thorin_set_kernel_arg"), arg_args);
}

llvm::Value* NVVMRuntime::set_kernel_arg_struct(llvm::Value* device, llvm::Value* ptr, llvm::Type* type) {
    return nullptr;
}

llvm::Value* NVVMRuntime::set_texture(llvm::Value* device, llvm::Value* ptr, llvm::Value* name, PrimTypeKind type) {
    return nullptr;

}

llvm::Value* NVVMRuntime::load_kernel(llvm::Value* device, llvm::Value* file, llvm::Value* kernel) {
    llvm::Value* load_args[] = { builder_.getInt32(1), device, file, kernel };
    return builder_.CreateCall(get("thorin_load_kernel"), load_args);
}

llvm::Value* NVVMRuntime::launch_kernel(llvm::Value* device, llvm::Value* name) {
    llvm::Value* arg_args[] = { builder_.getInt32(1), device };
    return builder_.CreateCall(get("thorin_launch_kernel"), arg_args);
}

std::string NVVMRuntime::get_module_name(Lambda* l) {
    return l->world().name() + ".nvvm";
}

}
