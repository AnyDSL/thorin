#ifndef THORIN_BE_LLVM_NVVM_H
#define THORIN_BE_LLVM_NVVM_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class Load;

class NVVMCodeGen : public CodeGen {
public:
    NVVMCodeGen(World& world);

protected:
    virtual llvm::Function* emit_function_decl(std::string&, Lambda*);
    virtual llvm::Function* emit_intrinsic_decl(std::string& name, Lambda* lambda);

    virtual llvm::Value* map_param(llvm::Function*, llvm::Argument*, const Param*);
    virtual void emit_function_start(llvm::BasicBlock*, llvm::Function*, Lambda*);

    virtual llvm::Value* emit_load(Def);
    virtual llvm::Value* emit_store(Def);
    virtual llvm::Value* emit_lea(Def);
    virtual llvm::Value* emit_map(Def);

    virtual std::string get_output_name(const std::string& name) const { return name + ".nvvm"; }
    virtual std::string get_binary_output_name(const std::string& name) const { return name + ".nvvm.bc"; }

private:
    llvm::Function* get_texture_handle_fun();
    llvm::GlobalVariable* resolve_global_variable(const Param*);

    HashMap<const Param*, llvm::MDNode*> metadata_;
};

}

#endif
