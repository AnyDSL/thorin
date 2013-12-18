#ifndef THORIN_BE_LLVM_H
#define THORIN_BE_LLVM_H

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
    CodeGen(World& world, llvm::CallingConv::ID calling_convention);

    template<typename DeclFunc, typename IntrinsicFunc>
    void emit(DeclFunc, IntrinsicFunc);
    llvm::Type* map(const Type*);
    llvm::Value* emit(Def);
    llvm::Value* lookup(Def);
    llvm::AllocaInst* emit_alloca(llvm::Type*, const std::string&);

private:
    Lambda* emit_builtin(Lambda*);
    Lambda* emit_nvvm(Lambda*);
    Lambda* emit_spir(Lambda*);

    llvm::Function* emit_nnvm_function_decl(llvm::LLVMContext&, llvm::Module*, std::string&, Lambda*);
    llvm::Function* emit_spir_function_decl(llvm::LLVMContext&, llvm::Module*, std::string&, Lambda*);

private:
    World& world;
    llvm::LLVMContext context;
    AutoPtr<llvm::Module> module;
    llvm::IRBuilder<> builder;
    LLVMDecls llvm_decls;
    llvm::CallingConv::ID calling_convention;
    std::unordered_map<const Param*, llvm::Value*> params;
    std::unordered_map<const Param*, llvm::PHINode*> phis;
    std::unordered_map<const PrimOp*, llvm::Value*> primops;
    std::unordered_map<Lambda*, llvm::Function*> fcts;
};

//------------------------------------------------------------------------------

template<class T> 
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

void emit_llvm(World& world);

} // namespace thorin

#endif
