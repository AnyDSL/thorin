#ifndef THORIN_BE_LLVM_RURNTIME_H
#define THORIN_BE_LLVM_RURNTIME_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "thorin/world.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class CodeGen;

class RuntimeBase {
protected:
    RuntimeBase(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder, const char* mod_name);
    virtual ~RuntimeBase() {}

    llvm::Function* get(const char* name);

    llvm::Module* target_;
    llvm::IRBuilder<>& builder_;
    AutoPtr<llvm::Module> module_;
};

class GenericRuntime : public RuntimeBase {
public:
    GenericRuntime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder);
    virtual ~GenericRuntime() {}

    llvm::Value* map(uint32_t device, uint32_t addr_space, llvm::Value* ptr);
};

class Runtime : public RuntimeBase {
protected:
    Runtime(llvm::LLVMContext& context, llvm::Module* target, llvm::IRBuilder<>& builder,
            llvm::Type* device_ptr_ty, const char* mod_name);

public:
    virtual ~Runtime() {}

    llvm::Type* get_device_ptr_ty() { return device_ptr_ty_; }

    // void* malloc(void* ptr);
    virtual llvm::Value* malloc(llvm::Value* size) = 0;
    // void free(void* ptr);
    virtual llvm::Value* free(llvm::Value* ptr) = 0;

    // void write(void* ptr, i8* data);
    virtual llvm::Value* write(llvm::Value* ptr, llvm::Value* data) = 0;
    // void read(void* ptr, i8* data);
    virtual llvm::Value* read(llvm::Value* ptr, llvm::Value* data) = 0;

    // void set_problem_size(i64, x, i64 y, i64 z);
    virtual llvm::Value* set_problem_size(llvm::Value* x, llvm::Value* y, llvm::Value* z) = 0;
    // void set_config_size(i64, x, i64 y, i64 z);
    virtual llvm::Value* set_config_size(llvm::Value* x, llvm::Value* y, llvm::Value* z) = 0;
    // void synchronize();
    virtual llvm::Value* synchronize() = 0;

    // void set_kernel_arg(void* ptr);
    virtual llvm::Value* set_kernel_arg(llvm::Value* ptr) = 0;
    // void set_mapped_kernel_arg(void* ptr);
    virtual llvm::Value* set_mapped_kernel_arg(llvm::Value* ptr) = 0;
    // void load_kernel(char* module, char* name);
    virtual llvm::Value* load_kernel(llvm::Value* module, llvm::Value* name) = 0;
    // void launch_kernel(char* name);
    virtual llvm::Value* launch_kernel(llvm::Value* name) = 0;

    virtual Lambda* emit_host_code(CodeGen &code_gen, Lambda*);

protected:
    virtual std::string get_module_name(Lambda*) = 0;

    llvm::Type* device_ptr_ty_;
};

}

#endif

