#ifndef THORIN_BE_LLVM_NVVM_H
#define THORIN_BE_LLVM_NVVM_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class Load;

class NVVMCodeGen : public CodeGen {
public:
    NVVMCodeGen(World& world, const Cont2Config&);

protected:
    // NVVM-specific optimizations are run in the runtime
    virtual void optimize(int opt) override { if (opt > 0) CodeGen::optimize(1); }

    virtual void emit_function_decl_hook(Lam*, llvm::Function*) override;
    virtual llvm::FunctionType* convert_fn_type(Lam*) override;
    virtual llvm::Value* map_var(llvm::Function*, llvm::Argument*, const Def*) override;
    virtual void emit_function_start(llvm::BasicBlock*, Lam*) override;
    virtual llvm::Value* emit_global(const Global*) override;
    virtual llvm::Value* emit_load(const App*) override;
    virtual llvm::Value* emit_store(const App*) override;
    virtual llvm::Value* emit_lea(const App*) override;
    virtual Lam* emit_reserve(Lam*) override;
    virtual std::string get_alloc_name() const override { return "malloc"; }

private:
    llvm::Function* get_texture_handle_fun();
    llvm::GlobalVariable* resolve_global_variable(const Def*);

    const Cont2Config& kernel_config_;
    DefMap<llvm::MDNode*> metadata_;
};

}

#endif
