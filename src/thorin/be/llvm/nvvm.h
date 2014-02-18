#ifndef THORIN_BE_LLVM_NVVM_H
#define THORIN_BE_LLVM_NVVM_H

#include <unordered_map>
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

private:
    llvm::Function* get_texture_handle_fun();
    llvm::GlobalVariable* resolve_global_variable(const Param*);

    std::unordered_map<const Param*, llvm::MDNode*> metadata_;
};

}

#endif
