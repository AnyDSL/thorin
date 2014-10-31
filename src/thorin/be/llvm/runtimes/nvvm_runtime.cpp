#include "thorin/be/llvm/runtimes/nvvm_runtime.h"

#include "thorin/primop.h"
#include "thorin/be/llvm/llvm.h"

namespace thorin {

NVVMRuntime::NVVMRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder)
    : KernelRuntime(context, target, builder, llvm::IntegerType::getInt64Ty(context), THORIN_RUNTIME_PLATFORMS "nvvm.s")
{}

llvm::Value* NVVMRuntime::malloc(llvm::Value* device, llvm::Value* ptr) {
    llvm::Value* malloc_args[] = { device, builder_.CreateBitCast(ptr, builder_.getInt8PtrTy()) };
    auto device_mem = builder_.CreateCall(get("nvvm_malloc_memory"), malloc_args);
    return device_mem;
}

llvm::Value* NVVMRuntime::free(llvm::Value* device, llvm::Value* mem) {
    llvm::Value* free_args[] = { device, mem };
    return builder_.CreateCall(get("nvvm_free_memory"), free_args);
}

llvm::Value* NVVMRuntime::write(llvm::Value* device, llvm::Value* mem, llvm::Value* data) {
    llvm::Value* mem_args[] = { device, mem, builder_.CreateBitCast(data, builder_.getInt8PtrTy()) };
    return builder_.CreateCall(get("nvvm_write_memory"), mem_args);
}

llvm::Value* NVVMRuntime::read(llvm::Value* device, llvm::Value* mem, llvm::Value* data) {
    llvm::Value* args[] = { device, mem, builder_.CreateBitCast(data, builder_.getInt8PtrTy()) };
    return builder_.CreateCall(get("nvvm_read_memory"), args);
}

llvm::Value* NVVMRuntime::set_problem_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* problem_size_args[] = { device, x, y, z };
    return builder_.CreateCall(get("nvvm_set_problem_size"), problem_size_args);
}

llvm::Value* NVVMRuntime::set_config_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) {
    llvm::Value* config_args[] = { device, x, y, z };
    return builder_.CreateCall(get("nvvm_set_config_size"), config_args);
}

llvm::Value* NVVMRuntime::synchronize(llvm::Value* device) {
    return builder_.CreateCall(get("nvvm_synchronize"), { device });
}

llvm::Value* NVVMRuntime::set_kernel_arg(llvm::Value* device, llvm::Value* ptr, llvm::Type* type) {
    llvm::Value* arg_args[] = { device, ptr };
    return builder_.CreateCall(get("nvvm_set_kernel_arg"), arg_args);
}

llvm::Value* NVVMRuntime::set_kernel_arg_map(llvm::Value* device, llvm::Value* mem) {
    llvm::Value* arg_args[] = { device, mem };
    return builder_.CreateCall(get("nvvm_set_kernel_arg_map"), arg_args);
}

llvm::Value* NVVMRuntime::set_kernel_arg_struct(llvm::Value* device, llvm::Value* ptr, llvm::Type* type) {
    return set_kernel_arg(device, ptr, type);
}

llvm::Value* NVVMRuntime::set_texture(llvm::Value* device, llvm::Value* ptr, llvm::Value* name, PrimTypeKind type) {
    int32_t format;
    switch(type) {
        case PrimType_ps8:  case PrimType_qs8:  format = 0x08; break;
        case PrimType_pu8:  case PrimType_qu8:  format = 0x01; break;
        case PrimType_ps16: case PrimType_qs16: format = 0x09; break;
        case PrimType_pu16: case PrimType_qu16: format = 0x02; break;
        case PrimType_bool:
        case PrimType_ps32: case PrimType_qs32: format = 0x0a; break;
        case PrimType_pu32: case PrimType_qu32: format = 0x03; break;
        case PrimType_pf32: case PrimType_qf32: format = 0x20; break;
        case PrimType_ps64: case PrimType_qs64:
        case PrimType_pu64: case PrimType_qu64:
        case PrimType_pf64: case PrimType_qf64:
        default:
            THORIN_UNREACHABLE;
    }
    auto loaded_device_ptr = builder_.CreatePtrToInt(ptr, builder_.getInt64Ty());
    llvm::Value* formatVal = builder_.getInt32(format);
    llvm::Value* tex_args[] = { device, loaded_device_ptr, name, formatVal };
    return builder_.CreateCall(get("nvvm_set_kernel_arg_tex"), tex_args);

}

llvm::Value* NVVMRuntime::load_kernel(llvm::Value* device, llvm::Value* file, llvm::Value* kernel) {
    llvm::Value* load_args[] = { device, file, kernel };
    return builder_.CreateCall(get("nvvm_load_nvvm_kernel"), load_args);
}

llvm::Value* NVVMRuntime::launch_kernel(llvm::Value* device, llvm::Value* name) {
    llvm::Value* arg_args[] = { device, name };
    return builder_.CreateCall(get("nvvm_launch_kernel"), arg_args);
}

std::string NVVMRuntime::get_module_name(Lambda* l) {
    return l->world().name() + ".nvvm";
}

}
