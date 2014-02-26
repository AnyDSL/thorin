#ifndef THORIN_BE_LLVM_RUNTIME_H
#define THORIN_BE_LLVM_RUNTIME_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "thorin/world.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class CodeGen;

class Runtime {
protected:
    Runtime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder, const char* mod_name);
    virtual ~Runtime() {}

    llvm::Function* get(const char* name);

    llvm::Module* target_;
    llvm::IRBuilder<>& builder_;
    AutoPtr<llvm::Module> module_;
};

class KernelRuntime : public Runtime {
protected:
    KernelRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder,
                  llvm::Type* device_ptr_ty, const char* mod_name);

public:
    virtual ~KernelRuntime() {}

    llvm::Type* get_device_ptr_ty() { return device_ptr_ty_; }

    // void* malloc(void* ptr);
    virtual llvm::Value* malloc(llvm::Value* device, llvm::Value* size) = 0;
    // void free(void* ptr);
    virtual llvm::Value* free(llvm::Value* device, llvm::Value* ptr) = 0;

    // void write(void* ptr, i8* data);
    virtual llvm::Value* write(llvm::Value* device, llvm::Value* ptr, llvm::Value* data) = 0;
    // void read(void* ptr, i8* data);
    virtual llvm::Value* read(llvm::Value* device, llvm::Value* ptr, llvm::Value* data) = 0;

    // void set_problem_size(i64, x, i64 y, i64 z);
    virtual llvm::Value* set_problem_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) = 0;
    // void set_config_size(i64, x, i64 y, i64 z);
    virtual llvm::Value* set_config_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) = 0;
    // void synchronize();
    virtual llvm::Value* synchronize(llvm::Value* device) = 0;

    // void set_kernel_arg(void* ptr);
    virtual llvm::Value* set_kernel_arg(llvm::Value* device, llvm::Value* ptr) = 0;
    // void set_mapped_kernel_arg(void* ptr);
    virtual llvm::Value* set_mapped_kernel_arg(llvm::Value* device, llvm::Value* ptr) = 0;
    // void load_kernel(char* module, char* name);
    virtual llvm::Value* load_kernel(llvm::Value* device, llvm::Value* module, llvm::Value* name) = 0;
    // void launch_kernel(char* name);
    virtual llvm::Value* launch_kernel(llvm::Value* device, llvm::Value* name) = 0;

    virtual Lambda* emit_host_code(CodeGen &code_gen, Lambda*);

protected:
    virtual std::string get_module_name(Lambda*) = 0;

    llvm::Type* device_ptr_ty_;
};

}

#endif

