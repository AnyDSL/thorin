#ifndef THORIN_BE_LLVM_LLVM_H
#define THORIN_BE_LLVM_LLVM_H

#include <unordered_map>

#include <llvm/IR/IRBuilder.h>

#include "thorin/lambda.h"
#include "thorin/be/llvm/decls.h"

namespace llvm {
    class Type;
    class Value;
}

namespace thorin {

class World;

typedef LambdaMap<llvm::BasicBlock*> BBMap;

class CodeGen {
public:
    void emit();

protected:
    CodeGen(World& world, llvm::CallingConv::ID calling_convention);

    llvm::Type* map(const Type*);
    llvm::Value* emit(Def);
    llvm::Value* lookup(Def);
    llvm::AllocaInst* emit_alloca(llvm::Type*, const std::string&);

private:
    Lambda* emit_builtin(Lambda*);
    Lambda* emit_nvvm(Lambda*);
    Lambda* emit_spir(Lambda*);
    virtual llvm::Function* emit_function_decl(std::string&, Lambda*);
    virtual llvm::Function* emit_intrinsic_decl(std::string& name, Lambda* lambda) {
        return CodeGen::emit_function_decl(name, lambda);
    }

protected:
    World& world_;
    llvm::LLVMContext context_;
    AutoPtr<llvm::Module> module_;
    llvm::IRBuilder<> builder_;
    LLVMDecls llvm_decls_;
    llvm::CallingConv::ID calling_convention_;
    std::unordered_map<const Param*, llvm::Value*> params_;
    std::unordered_map<const Param*, llvm::PHINode*> phis_;
    std::unordered_map<const PrimOp*, llvm::Value*> primops_;
    std::unordered_map<Lambda*, llvm::Function*> fcts_;
};

//------------------------------------------------------------------------------

template<class T> 
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

void emit_llvm(World& world);

} // namespace thorin

#endif
