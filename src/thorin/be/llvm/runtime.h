#ifndef THORIN_BE_LLVM_RURNTIME_H
#define THORIN_BE_LLVM_RURNTIME_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "thorin/world.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class CodeGen;

class Runtime {
protected:
    Runtime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<> &builder,
            llvm::Type* device_ptr_ty, const char* mod_name);

public:
    virtual ~Runtime() {}

    llvm::Type* get_device_ptr_ty() { return device_ptr_ty_; }

    // void* malloc(size);
    virtual llvm::Value* malloc(llvm::Value* size) = 0;
    // void free(void* ptr);
    virtual llvm::CallInst* free(llvm::Value* ptr) = 0;

    // void write(void* ptr, i8* data, i64 length);
    virtual llvm::CallInst* write(llvm::Value* ptr, llvm::Value* data, llvm::Value* length) = 0;
    // void read(void* ptr, i8* data, i64 length);
    virtual llvm::CallInst* read(llvm::Value* ptr, llvm::Value* data, llvm::Value* length) = 0;

    // void set_problem_size(i64, x, i64 y, i64 z);
    virtual llvm::CallInst* set_problem_size(llvm::Value* x, llvm::Value* y, llvm::Value* z) = 0;
    // void set_config_size(i64, x, i64 y, i64 z);
    virtual llvm::CallInst* set_config_size(llvm::Value* x, llvm::Value* y, llvm::Value* z) = 0;
    // void synchronize();
    virtual llvm::CallInst* synchronize() = 0;

    // void set_kernel_arg(void* ptr);
    virtual llvm::CallInst* set_kernel_arg(llvm::Value* ptr) = 0;
    // void load_kernel(char* module, char* name);
    virtual llvm::CallInst* load_kernel(llvm::Value* module, llvm::Value* name) = 0;
    // void launch_kernel(char* name);
    virtual llvm::CallInst* launch_kernel(llvm::Value* name) = 0;

    virtual Lambda* emit_host_code(CodeGen &code_gen, Lambda*) = 0;

protected:
    llvm::Function* get(const char* name);

    llvm::Module* target_;
    llvm::IRBuilder<> &builder_;
    llvm::Type* device_ptr_ty_;
    AutoPtr<llvm::Module> module_;
};

}

#endif

