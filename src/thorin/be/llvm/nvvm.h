#ifndef THORIN_BE_LLVM_NVVM_H
#define THORIN_BE_LLVM_NVVM_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class Load;

namespace llvm {

namespace llvm = ::llvm;

class NVVMCodeGen : public CodeGen {
public:
    NVVMCodeGen(World& world, const Cont2Config&, bool debug); // NVVM-specific optimizations are run in the runtime

protected:
    virtual void emit_function_decl_hook(Continuation*, llvm::Function*) override;
    virtual llvm::FunctionType* convert_fn_type(Continuation*) override;
    virtual llvm::Value* map_param(llvm::Function*, llvm::Argument*, const Param*) override;
    virtual void emit_function_start(Continuation*) override;

    virtual llvm::Value* emit_load(llvm::IRBuilder<>&,  const Load*) override;
    virtual llvm::Value* emit_store(llvm::IRBuilder<>&, const Store*) override;
    virtual llvm::Value* emit_lea(llvm::IRBuilder<>&,   const LEA*) override;

    virtual Continuation* emit_reserve(llvm::IRBuilder<>&, const Continuation*) override;

    virtual llvm::Value* emit_global(const Global*) override;

    virtual std::string get_alloc_name() const override { return "malloc"; }

private:
    llvm::Function* get_texture_handle_fun(llvm::IRBuilder<>&);
    llvm::GlobalVariable* resolve_global_variable(const Param*);

    const Cont2Config& kernel_config_;
    ParamMap<llvm::MDNode*> metadata_;
};

}

}

#endif
