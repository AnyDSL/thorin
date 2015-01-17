#include <llvm/IR/DataLayout.h>

#include "thorin/primop.h"
#include "thorin/be/llvm/llvm.h"
#include "thorin/be/llvm/runtimes/generic_runtime.h"

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

llvm::Value* GenericRuntime::parallel_create(llvm::Value* num_threads, llvm::Value* closure_ptr,
                                             uint64_t closure_size, llvm::Value* fun_ptr) {
    llvm::Value* parallel_args[] = {
        num_threads,
        builder_.CreateBitCast(closure_ptr, builder_.getInt8PtrTy()),
        builder_.getInt64(closure_size),
        builder_.CreateBitCast(fun_ptr, builder_.getInt8PtrTy())
    };
    return builder_.CreateCall(get("parallel_create"), parallel_args);
}

llvm::Value* GenericRuntime::parallel_join(llvm::Value* handle) {
    return builder_.CreateCall(get("parallel_join"), { handle });
}

}

