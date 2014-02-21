#ifndef THORIN_BE_LLVM_LLVM_H
#define THORIN_BE_LLVM_LLVM_H

#include <unordered_map>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "thorin/lambda.h"
#include "thorin/be/llvm/runtime.h"

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
    virtual llvm::Function* emit_function_decl(std::string&, Lambda*);
    virtual llvm::Function* emit_intrinsic_decl(std::string& name, Lambda* lambda) {
        return CodeGen::emit_function_decl(name, lambda);
    }
    virtual llvm::Value* map_param(llvm::Function*, llvm::Argument* a, const Param*) { return a; }
    virtual void emit_function_start(llvm::BasicBlock*, llvm::Function*, Lambda*) {}

    virtual llvm::Value* emit_load(Def);
    virtual llvm::Value* emit_store(Def);
    virtual llvm::Value* emit_lea(Def);
    virtual llvm::Value* emit_memmap(Def);

    virtual std::string get_output_name(const std::string& name) const = 0;
    virtual std::string get_binary_output_name(const std::string& name) const = 0;

private:
    Lambda* emit_builtin(llvm::Function*, Lambda*);
    Lambda* emit_vectorized(llvm::Function*, Lambda*);

protected:
    World& world_;
    llvm::LLVMContext context_;
    AutoPtr<llvm::Module> module_;
    llvm::IRBuilder<> builder_;
    llvm::CallingConv::ID calling_convention_;
    std::unordered_map<const Param*, llvm::Value*> params_;
    std::unordered_map<const Param*, llvm::PHINode*> phis_;
    std::unordered_map<const PrimOp*, llvm::Value*> primops_;
    std::unordered_map<Lambda*, llvm::Function*> fcts_;
    std::set<llvm::Function*> fcts_to_remove_;
    AutoPtr<GenericRuntime> runtime_;
    AutoPtr<Runtime> nvvm_runtime_;
    AutoPtr<Runtime> spir_runtime_;
    AutoPtr<Runtime> opencl_runtime_;

    friend class Runtime;
    friend class NVVMRuntime;
    friend class SpirRuntime;
    friend class OpenCLRuntime;
};

//------------------------------------------------------------------------------

template<class T>
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

void emit_llvm(World& world);

} // namespace thorin

#endif
