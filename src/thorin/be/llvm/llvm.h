#ifndef THORIN_BE_LLVM_LLVM_H
#define THORIN_BE_LLVM_LLVM_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "thorin/lambda.h"
#include "thorin/be/llvm/runtime.h"
#include "thorin/be/llvm/runtimes/generic_runtime.h"

namespace thorin {

class World;

typedef LambdaMap<llvm::BasicBlock*> BBMap;

class CodeGen {
protected:
    CodeGen(World& world, llvm::CallingConv::ID function_calling_convention,
            llvm::CallingConv::ID device_calling_convention, llvm::CallingConv::ID kernel_calling_convention);

public:
    void emit(int opt);

protected:
    void optimize(int opt);

    llvm::Type* convert(Type);
    llvm::Value* emit(Def);
    llvm::Value* lookup(Def);
    llvm::AllocaInst* emit_alloca(llvm::Type*, const std::string&);
    virtual llvm::Function* emit_function_decl(std::string&, Lambda*);
    virtual llvm::Value* map_param(llvm::Function*, llvm::Argument* a, const Param*) { return a; }
    virtual void emit_function_start(llvm::BasicBlock*, llvm::Function*, Lambda*) {}

    virtual llvm::Value* emit_load(Def);
    virtual llvm::Value* emit_store(Def);
    virtual llvm::Value* emit_lea(Def);
    virtual llvm::Value* emit_mmap(Def);
    virtual llvm::Value* emit_munmap(Def);

    virtual std::string get_alloc_name() const = 0;
    virtual std::string get_output_name(const std::string& name) const = 0;
    virtual std::string get_binary_output_name(const std::string& name) const = 0;
    llvm::GlobalVariable* emit_global_memory(llvm::Type*, const std::string&, unsigned);
    llvm::Value* emit_shared_mmap(Def def, bool prefix=false);
    llvm::Value* emit_shared_munmap(Def def);

private:
    Lambda* emit_intrinsic(llvm::Function*, Lambda*);
    Lambda* emit_vectorize(llvm::Function*, Lambda*);

protected:
    World& world_;
    llvm::LLVMContext context_;
    AutoPtr<llvm::Module> module_;
    llvm::IRBuilder<> builder_;
    llvm::CallingConv::ID function_calling_convention_;
    llvm::CallingConv::ID device_calling_convention_;
    llvm::CallingConv::ID kernel_calling_convention_;
    HashMap<const Param*, llvm::Value*> params_;
    HashMap<const Param*, llvm::PHINode*> phis_;
    HashMap<const PrimOp*, llvm::Value*> primops_;
    TypeMap<llvm::Type*> types_;
    HashMap<Lambda*, llvm::Function*> fcts_;
    HashSet<llvm::Function*> fcts_to_remove_;

    AutoPtr<GenericRuntime> runtime_;
    AutoPtr<KernelRuntime> cuda_runtime_;
    AutoPtr<KernelRuntime> nvvm_runtime_;
    AutoPtr<KernelRuntime> spir_runtime_;
    AutoPtr<KernelRuntime> opencl_runtime_;
    Lambda* current_kernel_;

    friend class GenericRuntime;
    friend class KernelRuntime;
};

//------------------------------------------------------------------------------

template<class T>
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

void emit_llvm(World& world, int opt);

} // namespace thorin

#endif
