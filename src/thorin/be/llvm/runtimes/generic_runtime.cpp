#include "thorin/be/llvm/runtimes/generic_runtime.h"

#include "thorin/primop.h"
#include "thorin/be/llvm/llvm.h"

namespace thorin {

GenericRuntime::GenericRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder)
    : Runtime(context, target, builder, THORIN_RUNTIME_PLATFORMS "generic.s")
{}

llvm::Value* GenericRuntime::mmap(uint32_t device, uint32_t addr_space, llvm::Value* ptr,
                                  llvm::Value* mem_offset, llvm::Value* mem_size, llvm::Value* elem_size) {
    llvm::Value* mmap_args[] = {
        builder_.getInt32(device),
        builder_.getInt32(addr_space),
        builder_.CreateBitCast(ptr, builder_.getInt8PtrTy()),
        builder_.CreateMul(mem_offset, elem_size),
        builder_.CreateMul(mem_size, elem_size)
    };
    return builder_.CreateCall(get("map_memory"), mmap_args);
}

llvm::Value* GenericRuntime::munmap(llvm::Value* mem) {
    return builder_.CreateCall(get("unmap_memory"), mem);
}

llvm::Value* GenericRuntime::parallel_for(llvm::Value* num_threads, llvm::Value* lower, llvm::Value* upper,
                                          llvm::Value* closure_ptr, llvm::Value* fun_ptr) {
    llvm::Value* parallel_args[] = {
        num_threads, lower, upper,
        builder_.CreateBitCast(closure_ptr, builder_.getInt8PtrTy()),
        builder_.CreateBitCast(fun_ptr, builder_.getInt8PtrTy())
    };
    return builder_.CreateCall(get("parallel_for"), parallel_args);
}

}
