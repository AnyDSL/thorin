#ifndef THORIN_BE_LLVM_NVVM_H
#define THORIN_BE_LLVM_NVVM_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class Load;

namespace llvm {

namespace llvm = ::llvm;

class NVVMCodeGen : public CodeGen {
public:
    NVVMCodeGen(World& world, const Cont2Config&, int opt, bool debug);

    const char* file_ext() const override { return ".nvvm"; }

protected:
    void emit_fun_decl_hook(Continuation*, llvm::Function*) override;
    llvm::FunctionType* convert_fn_type(Continuation*) override;
    llvm::Value* map_param(llvm::Function*, llvm::Argument*, const Param*) override;
    void prepare(Continuation*, llvm::Function*) override;

    llvm::Value* emit_load(llvm::IRBuilder<>&,   const Load*) override;
    llvm::Value* emit_store(llvm::IRBuilder<>&,  const Store*) override;
    llvm::Value* emit_lea(llvm::IRBuilder<>&,    const LEA*) override;
    llvm::Value* emit_mathop(llvm::IRBuilder<>&, const MathOp*) override;

    Continuation* emit_reserve(llvm::IRBuilder<>&, const Continuation*) override;

    llvm::Value* emit_global(const Global*) override;

    std::string get_alloc_name() const override { return "malloc"; }

private:
    llvm::Function* get_texture_handle_fun(llvm::IRBuilder<>&);
    llvm::GlobalVariable* resolve_global_variable(const Param*);

    const Cont2Config& kernel_config_;
    ParamMap<llvm::MDNode*> metadata_;
};

}

}

#endif
