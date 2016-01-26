#ifndef THORIN_BE_LLVM_LLVM_H
#define THORIN_BE_LLVM_LLVM_H

#include <llvm/DIBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "thorin/lambda.h"
#include "thorin/be/llvm/runtime.h"

namespace thorin {

class World;

typedef LambdaMap<llvm::BasicBlock*> BBMap;

class CodeGen {
protected:
    CodeGen(World& world, llvm::GlobalValue::LinkageTypes function_import_linkage, llvm::GlobalValue::LinkageTypes function_export_linkage,
            llvm::CallingConv::ID function_calling_convention, llvm::CallingConv::ID device_calling_convention, llvm::CallingConv::ID kernel_calling_convention);

public:
    World& world() const { return world_; }
    void emit(int opt, bool debug);

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

    virtual std::string get_alloc_name() const = 0;
    virtual std::string get_output_name(const std::string& name) const = 0;
    virtual std::string get_binary_output_name(const std::string& name) const = 0;
    llvm::GlobalVariable* emit_global_variable(llvm::Type*, const std::string&, unsigned);
    Lambda* emit_reserve_shared(const Lambda*, bool prefix=false);

private:
    Lambda* emit_intrinsic(Lambda*);
    Lambda* emit_parallel(Lambda*);
    Lambda* emit_spawn(Lambda*);
    Lambda* emit_sync(Lambda*);
    Lambda* emit_vectorize_continuation(Lambda*);
    Lambda* emit_atomic(Lambda*);
    Lambda* emit_sizeof(Lambda*);
    Lambda* emit_select(Lambda*);
    Lambda* emit_shuffle(Lambda*);
    Lambda* emit_reinterpret(Lambda*);
    llvm::Value* emit_bitcast(Def, Type);
    virtual Lambda* emit_reserve(const Lambda*);
    void emit_result_phi(const Param*, llvm::Value*);
    void emit_vectorize(u32, llvm::Function*, llvm::CallInst*);

protected:
    void create_loop(llvm::Value*, llvm::Value*, llvm::Value*, llvm::Function*, std::function<void(llvm::Value*)>);

    World& world_;
    llvm::LLVMContext context_;
    AutoPtr<llvm::Module> module_;
    llvm::IRBuilder<> irbuilder_;
    llvm::DIBuilder dibuilder_;
    llvm::GlobalValue::LinkageTypes function_import_linkage_;
    llvm::GlobalValue::LinkageTypes function_export_linkage_;
    llvm::CallingConv::ID function_calling_convention_;
    llvm::CallingConv::ID device_calling_convention_;
    llvm::CallingConv::ID kernel_calling_convention_;
    HashMap<const Param*, llvm::Value*> params_;
    HashMap<const Param*, llvm::PHINode*> phis_;
    HashMap<const PrimOp*, llvm::Value*> primops_;
    HashMap<Lambda*, llvm::Function*> fcts_;
    TypeMap<llvm::Type*> types_;
    std::vector<std::tuple<u32, llvm::Function*, llvm::CallInst*>> wfv_todo_;

    AutoPtr<Runtime> runtime_;
    Lambda* entry_ = nullptr;

    friend class Runtime;
};

//------------------------------------------------------------------------------

template<class T>
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

void emit_llvm(World& world, int opt, bool debug);

} // namespace thorin

#endif
