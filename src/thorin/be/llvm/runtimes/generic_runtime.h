#ifndef THORIN_BE_LLVM_RUNTIMES_GENERIC_H
#define THORIN_BE_LLVM_RUNTIMES_GENERIC_H

#include "thorin/be/llvm/runtime.h"

namespace thorin {

class GenericRuntime : public Runtime {
public:
    GenericRuntime(llvm::LLVMContext& context, llvm::Module& target, llvm::IRBuilder<>& builder);
    virtual ~GenericRuntime() {}

    virtual llvm::Value* mmap(uint32_t device, uint32_t addr_space, llvm::Value* ptr,
                              llvm::Value* mem_offset, llvm::Value* mem_size, llvm::Value* elem_size);
    virtual llvm::Value* munmap(llvm::Value* mem);
    virtual llvm::Value* parallel_for(llvm::Value* num_threads, llvm::Value* lower, llvm::Value* upper,
                                     llvm::Value* closure_ptr, llvm::Value* fun_ptr);
    virtual llvm::Value* parallel_spawn(llvm::Value* closure_ptr, llvm::Value* fun_ptr);
    virtual llvm::Value* parallel_sync(llvm::Value* id);
};

}

#endif

