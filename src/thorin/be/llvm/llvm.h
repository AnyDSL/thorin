#ifndef THORIN_BE_LLVM_LLVM_H
#define THORIN_BE_LLVM_LLVM_H

#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "thorin/continuation.h"
#include "thorin/be/llvm/runtime.h"

namespace thorin {

class World;

typedef ContinuationMap<llvm::BasicBlock*> BBMap;
typedef ContinuationMap<std::tuple<int, int, int>> Cont2Config;

class CodeGen {
protected:
    CodeGen(World& world, llvm::CallingConv::ID function_calling_convention, llvm::CallingConv::ID device_calling_convention, llvm::CallingConv::ID kernel_calling_convention, const Cont2Config& kernel_config);

public:
    World& world() const { return world_; }
    void emit(int opt, bool debug);

protected:
    void optimize(int opt);

    unsigned compute_variant_bits(const VariantType*);
    llvm::Type* convert(const Type*);
    llvm::Value* emit(const Def*);
    llvm::Value* lookup(const Def*);
    llvm::AllocaInst* emit_alloca(llvm::Type*, const std::string&);
    llvm::Function* emit_function_decl(Continuation*);
    virtual unsigned convert_addr_space(const AddrSpace);
    virtual void emit_function_decl_hook(Continuation*, llvm::Function*) {}
    virtual llvm::Value* map_param(llvm::Function*, llvm::Argument* a, const Param*) { return a; }
    virtual void emit_function_start(llvm::BasicBlock*, Continuation*) {}
    virtual llvm::FunctionType* convert_fn_type(Continuation*);

    virtual llvm::Value* emit_global(const Global*);
    virtual llvm::Value* emit_load(const Load*);
    virtual llvm::Value* emit_store(const Store*);
    virtual llvm::Value* emit_lea(const LEA*);
    virtual llvm::Value* emit_assembly(const Assembly* assembly);

    virtual std::string get_alloc_name() const = 0;
    virtual std::string get_output_name(const std::string& name) const = 0;
    llvm::GlobalVariable* emit_global_variable(llvm::Type*, const std::string&, unsigned, bool=false);
    Continuation* emit_reserve_shared(const Continuation*, bool=false);

private:
    Continuation* emit_peinfo(Continuation*);
    Continuation* emit_intrinsic(Continuation*);
    Continuation* emit_parallel(Continuation*);
    Continuation* emit_spawn(Continuation*);
    Continuation* emit_sync(Continuation*);
    Continuation* emit_vectorize_continuation(Continuation*);
    Continuation* emit_atomic(Continuation*);
    Continuation* emit_cmpxchg(Continuation*);
    llvm::Value* emit_bitcast(const Def*, const Type*);
    virtual Continuation* emit_reserve(const Continuation*);
    void emit_result_phi(const Param*, llvm::Value*);
    void emit_vectorize(u32, u32, llvm::Function*, llvm::CallInst*);

protected:
    void create_loop(llvm::Value*, llvm::Value*, llvm::Value*, llvm::Function*, std::function<void(llvm::Value*)>);
    llvm::Value* create_tmp_alloca(llvm::Type*, std::function<llvm::Value* (llvm::AllocaInst*)>);

    World& world_;
    llvm::LLVMContext context_;
    std::unique_ptr<llvm::Module> module_;
    llvm::IRBuilder<> irbuilder_;
    llvm::DIBuilder dibuilder_;
    llvm::CallingConv::ID function_calling_convention_;
    llvm::CallingConv::ID device_calling_convention_;
    llvm::CallingConv::ID kernel_calling_convention_;
    const Cont2Config& kernel_config_;
    ParamMap<llvm::Value*> params_;
    ParamMap<llvm::PHINode*> phis_;
    PrimOpMap<llvm::Value*> primops_;
    ContinuationMap<llvm::Function*> fcts_;
    TypeMap<llvm::Type*> types_;
#ifdef RV_SUPPORT
    std::vector<std::tuple<u32, u32, llvm::Function*, llvm::CallInst*>> vec_todo_;
#endif

    std::unique_ptr<Runtime> runtime_;
    Continuation* entry_ = nullptr;

    friend class Runtime;
};

//------------------------------------------------------------------------------

template<class T>
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

void emit_llvm(World& world, int opt, bool debug);

} // namespace thorin

#endif
