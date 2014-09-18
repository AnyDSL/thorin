#ifndef THORIN_BE_LLVM_NVVM_H
#define THORIN_BE_LLVM_NVVM_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class Load;

class NVVMCodeGen : public CodeGen {
public:
    NVVMCodeGen(World& world);

protected:
    virtual void emit_function_decl_hook(Lambda*, llvm::Function*) override;
    virtual llvm::FunctionType* convert_fn_type(Lambda*) override;
    virtual llvm::Value* map_param(llvm::Function*, llvm::Argument*, const Param*) override;
    virtual void emit_function_start(llvm::BasicBlock*, llvm::Function*, Lambda*) override;
    virtual llvm::Value* emit_load(Def) override;
    virtual llvm::Value* emit_store(Def) override;
    virtual llvm::Value* emit_lea(Def) override;
    virtual llvm::Value* emit_mmap(Def) override;
    virtual std::string get_alloc_name() const override { return "malloc"; }
    virtual std::string get_output_name(const std::string& name) const override { return name + ".nvvm"; }
    virtual std::string get_binary_output_name(const std::string& name) const override { return name + ".nvvm.bc"; }

private:
    llvm::Function* get_texture_handle_fun();
    llvm::GlobalVariable* resolve_global_variable(const Param*);

    HashMap<const Param*, llvm::MDNode*> metadata_;
};

}

#endif
