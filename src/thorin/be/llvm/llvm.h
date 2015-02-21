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
    World& world() const { return world_; }

protected:
    void optimize(int opt);

    llvm::Type* convert(Type);
    llvm::Value* emit(Def);
    llvm::Value* lookup(Def);
    llvm::AllocaInst* emit_alloca(llvm::Type*, const std::string&);
    llvm::Function* emit_function_decl(Lambda*);
    virtual void emit_function_decl_hook(Lambda*, llvm::Function*) {}
    virtual llvm::Value* map_param(llvm::Function*, llvm::Argument* a, const Param*) { return a; }
    virtual void emit_function_start(llvm::BasicBlock*, Lambda*) {}
    virtual llvm::FunctionType* convert_fn_type(Lambda*);

    virtual llvm::Value* emit_load(const Load*);
    virtual llvm::Value* emit_store(const Store*);
    virtual llvm::Value* emit_lea(const LEA*);
    virtual llvm::Value* emit_mmap(const Map*);

    virtual std::string get_alloc_name() const = 0;
    virtual std::string get_output_name(const std::string& name) const = 0;
    virtual std::string get_binary_output_name(const std::string& name) const = 0;
    llvm::GlobalVariable* emit_global_memory(llvm::Type*, const std::string&, unsigned);
    llvm::Value* emit_shared_mmap(Def def, bool prefix=false);

private:
    Lambda* emit_intrinsic(Lambda*);
    Lambda* emit_parallel(Lambda*);
    Lambda* emit_vectorize_continuation(Lambda*);
    Lambda* emit_atomic(Lambda*);
    void emit_vectorize(u32, llvm::Value*, llvm::Function*, llvm::CallInst*);
    llvm::Function* get_vectorize_tid();

protected:
    llvm::Value* create_loop(llvm::Value*, llvm::Value*, llvm::Value*, llvm::Function*, std::function<void(llvm::Value*)>);

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
    HashMap<Lambda*, llvm::Function*> fcts_;
    TypeMap<llvm::Type*> types_;
    std::vector<std::tuple<u32, llvm::Value*, llvm::Function*, llvm::CallInst*>> wfv_todo_;

    AutoPtr<GenericRuntime> runtime_;
    AutoPtr<KernelRuntime> cuda_runtime_;
    AutoPtr<KernelRuntime> nvvm_runtime_;
    AutoPtr<KernelRuntime> spir_runtime_;
    AutoPtr<KernelRuntime> opencl_runtime_;
    Lambda* entry_ = nullptr;

    friend class GenericRuntime;
    friend class KernelRuntime;
};

//------------------------------------------------------------------------------

template<class T>
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

void emit_llvm(World& world, int opt);

} // namespace thorin

#endif
