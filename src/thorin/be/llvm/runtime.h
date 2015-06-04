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
    Runtime(llvm::LLVMContext& context, llvm::Module& target, llvm::IRBuilder<>& builder, const char* mod_name);
    virtual ~Runtime() {}

    llvm::Function* get(const char* name);

    llvm::Module& target_;
    llvm::IRBuilder<>& builder_;
    std::unique_ptr<llvm::Module> module_;
};

class KernelRuntime : public Runtime {
protected:
    KernelRuntime(llvm::LLVMContext& context, llvm::Module& target, llvm::IRBuilder<>& builder,
                  llvm::Type* device_ptr_ty, const char* mod_name);

public:
    virtual ~KernelRuntime() {}

    llvm::Type* get_device_ptr_ty() { return device_ptr_ty_; }

    // i32 malloc(i32 dev, void* ptr);
    virtual llvm::Value* malloc(llvm::Value* device, llvm::Value* size) = 0;
    // void free(i32 dev, i32 mem);
    virtual llvm::Value* free(llvm::Value* device, llvm::Value* ptr) = 0;

    // void write(i32 dev, i32 mem, i8* data);
    virtual llvm::Value* write(llvm::Value* device, llvm::Value* ptr, llvm::Value* data) = 0;
    // void read(i32 dev, i32 mem, i8* data);
    virtual llvm::Value* read(llvm::Value* device, llvm::Value* ptr, llvm::Value* data) = 0;

    // void set_problem_size(i32 dev, i32 x, i32 y, i32 z);
    virtual llvm::Value* set_problem_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) = 0;
    // void set_config_size(i32 dev, i32 x, i32 y, i32 z);
    virtual llvm::Value* set_config_size(llvm::Value* device, llvm::Value* x, llvm::Value* y, llvm::Value* z) = 0;
    // void synchronize(i32 dev);
    virtual llvm::Value* synchronize(llvm::Value* device) = 0;

    // void set_kernel_arg(i32 dev, void* ptr, i32 size);
    virtual llvm::Value* set_kernel_arg(llvm::Value* device, llvm::Value* ptr, llvm::Type* type) = 0;
    // void set_kernel_arg_map(i32 dev, i32 mem);
    virtual llvm::Value* set_kernel_arg_map(llvm::Value* device, llvm::Value* ptr) = 0;
    // void set_kernel_arg_struct(i32 dev, void* ptr, i32 size);
    virtual llvm::Value* set_kernel_arg_struct(llvm::Value* device, llvm::Value* ptr, llvm::Type* type) = 0;
    // void set_texture(i32 dev, i32 mem, i8* name, i32 type);
    virtual llvm::Value* set_texture(llvm::Value* device, llvm::Value* ptr, llvm::Value* name, PrimTypeKind type) = 0;
    // void load_kernel(i32 dev, char* file, char* kernel);
    virtual llvm::Value* load_kernel(llvm::Value* device, llvm::Value* file, llvm::Value* kernel) = 0;
    // void launch_kernel(i32 dev, char* name);
    virtual llvm::Value* launch_kernel(llvm::Value* device, llvm::Value* name) = 0;

    virtual Lambda* emit_host_code(CodeGen &code_gen, Lambda*);

protected:
    virtual std::string get_module_name(Lambda*) = 0;

    llvm::Type* device_ptr_ty_;
};

}

#endif

