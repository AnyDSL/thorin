#ifndef THORIN_BE_LLVM_LLVM_H
#define THORIN_BE_LLVM_LLVM_H

#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

#include "thorin/config.h"
#include "thorin/continuation.h"
#include "thorin/analyses/schedule.h"
#include "thorin/be/llvm/runtime.h"
#include "thorin/be/kernel_config.h"
#include "thorin/transform/importer.h"

namespace thorin {

class World;

typedef ContinuationMap<llvm::BasicBlock*> BBMap;

class CodeGen {
protected:
    CodeGen(World& world,
            llvm::CallingConv::ID function_calling_convention,
            llvm::CallingConv::ID device_calling_convention,
            llvm::CallingConv::ID kernel_calling_convention,
            int opt, bool debug);
public:
    virtual ~CodeGen() {}

    /// @name getters
    //@{
    World& world() const { return world_; }
    llvm::LLVMContext& context() { return *context_; }
    llvm::Module& module() { return *module_; }
    const llvm::Module& module() const { return *module_; }
    virtual void emit(std::ostream& stream);
    int opt() const { return opt_; }
    bool debug() const { return debug_; }
    //@}

    std::unique_ptr<llvm::Module>& emit();

protected:
    llvm::Type* convert(const Type*);
    void emit(const Scope&);
    llvm::Value* emit(const Def*);
    void emit_epilogue(Continuation*);
    llvm::AllocaInst* emit_alloca(llvm::Type*, const std::string&);
    llvm::Value* emit_alloc(const Type*, const Def*);
    virtual llvm::Function* emit_function_decl(Continuation*);
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

    llvm::GlobalVariable* emit_global_variable(llvm::Type*, const std::string&, unsigned, bool=false);
    Continuation* emit_reserve_shared(const Continuation*, bool=false);
    void optimize();
    void verify() const;
    void create_loop(llvm::Value*, llvm::Value*, llvm::Value*, llvm::Function*, std::function<void(llvm::Value*)>);
    llvm::Value* create_tmp_alloca(llvm::Type*, std::function<llvm::Value* (llvm::AllocaInst*)>);

private:
    Continuation* emit_peinfo(Continuation*);
    Continuation* emit_intrinsic(Continuation*);
#if 0
    Continuation* emit_hls(Continuation*);
    Continuation* emit_parallel(Continuation*);
    Continuation* emit_fibers(Continuation*);
    Continuation* emit_spawn(Continuation*);
    Continuation* emit_sync(Continuation*);
    Continuation* emit_vectorize_continuation(Continuation*);
    void emit_vectorize(u32, llvm::Function*, llvm::CallInst*);
#endif
    Continuation* emit_atomic(Continuation*);
    Continuation* emit_cmpxchg(Continuation*);
    Continuation* emit_atomic_load(Continuation*);
    Continuation* emit_atomic_store(Continuation*);
    llvm::Value* emit_bitcast(const Def*, const Type*);
    virtual Continuation* emit_reserve(const Continuation*);
    void emit_result_phi(const Param*, llvm::Value*);

    World& world_;
    std::unique_ptr<llvm::LLVMContext> context_;
    std::unique_ptr<llvm::Module> module_;
    int opt_;
    bool debug_;

protected:
    std::unique_ptr<llvm::TargetMachine> machine_;
    llvm::IRBuilder<> irbuilder_;
    llvm::DIBuilder dibuilder_;
    llvm::DICompileUnit* dicompile_unit_;
    llvm::CallingConv::ID function_calling_convention_;
    llvm::CallingConv::ID device_calling_convention_;
    llvm::CallingConv::ID kernel_calling_convention_;
    ParamMap<llvm::Value*> params_;
    ParamMap<llvm::PHINode*> phis_;
    PrimOpMap<llvm::Value*> primops_;
    BBMap cont2bb_;
    Scheduler scheduler_;
    ContinuationMap<llvm::Function*> fcts_;
    TypeMap<llvm::Type*> types_;
#if THORIN_ENABLE_RV
    std::vector<std::tuple<u32, llvm::Function*, llvm::CallInst*>> vec_todo_;
#endif

    std::unique_ptr<Runtime> runtime_;
    Continuation* entry_ = nullptr;

    friend class Runtime;
};

//------------------------------------------------------------------------------

template<class T>
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

struct Backends {
    Backends(World& world, int opt, bool debug);

    Cont2Config kernel_config;
    std::vector<Continuation*> kernels;

    // TODO use arrays + loops for this
    Importer cuda;
    Importer nvvm;
    Importer opencl;
    Importer amdgpu;
    Importer hls;

    // TODO use arrays + loops for this
    std::unique_ptr<CodeGen> cpu_cg;
    std::unique_ptr<CodeGen> cuda_cg;
    std::unique_ptr<CodeGen> nvvm_cg;
    std::unique_ptr<CodeGen> opencl_cg;
    std::unique_ptr<CodeGen> amdgpu_cg;
    std::unique_ptr<CodeGen> hls_cg;
};

} // namespace thorin

#endif
