#ifndef THORIN_BE_LLVM_LLVM_H
#define THORIN_BE_LLVM_LLVM_H

#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

#include "thorin/config.h"
#include "thorin/def.h"
#include "thorin/be/llvm/runtime.h"
#include "thorin/be/kernel_config.h"
#include "thorin/transform/importer.h"

namespace thorin {

class World;

typedef LamMap<llvm::BasicBlock*> BBMap;

class CodeGen {
protected:
    CodeGen(World& world, llvm::CallingConv::ID function_calling_convention, llvm::CallingConv::ID device_calling_convention, llvm::CallingConv::ID kernel_calling_convention);

public:
    virtual ~CodeGen() {}

    World& world() const { return world_; }
    std::unique_ptr<llvm::Module>& emit(int opt, bool debug);
    virtual void emit(std::ostream& stream, int opt, bool debug);

protected:
    virtual void optimize(int opt);

    unsigned compute_variant_bits(const VariantType*);
    unsigned compute_variant_op_bits(const Type*);
    llvm::Type* convert(const Type*);
    llvm::Value* emit(const Def*);
    llvm::Value* lookup(const Def*);
    llvm::AllocaInst* emit_alloca(llvm::Type*, const std::string&);
    llvm::Value* emit_alloc(const Type*, const Def*);
    llvm::Function* emit_function_decl(Lam*);
    virtual unsigned convert_addr_space(const AddrSpace);
    virtual void emit_function_decl_hook(Lam*, llvm::Function*) {}
    virtual llvm::Value* map_param(llvm::Function*, llvm::Argument* a, const Def*) { return a; }
    virtual void emit_function_start(llvm::BasicBlock*, Lam*) {}
    virtual llvm::FunctionType* convert_fn_type(Lam*);

    virtual llvm::Value* emit_global(const Global*);
    virtual llvm::Value* emit_load(const Load*);
    virtual llvm::Value* emit_store(const Store*);
    virtual llvm::Value* emit_lea(const LEA*);
    virtual llvm::Value* emit_assembly(const Assembly* assembly);

    virtual std::string get_alloc_name() const = 0;
    llvm::GlobalVariable* emit_global_variable(llvm::Type*, const std::string&, unsigned, bool=false);
    Lam* emit_reserve_shared(const Lam*, bool=false);

private:
    Lam* emit_peinfo(Lam*);
    Lam* emit_intrinsic(Lam*);
    Lam* emit_hls(Lam*);
    Lam* emit_parallel(Lam*);
    Lam* emit_spawn(Lam*);
    Lam* emit_sync(Lam*);
    Lam* emit_vectorize_lam(Lam*);
    Lam* emit_atomic(Lam*);
    Lam* emit_cmpxchg(Lam*);
    llvm::Value* emit_bitcast(const Def*, const Type*);
    virtual Lam* emit_reserve(const Lam*);
    void emit_result_phi(const Def*, llvm::Value*);
    void emit_vectorize(u32, llvm::Function*, llvm::CallInst*);

protected:
    void create_loop(llvm::Value*, llvm::Value*, llvm::Value*, llvm::Function*, std::function<void(llvm::Value*)>);
    llvm::Value* create_tmp_alloca(llvm::Type*, std::function<llvm::Value* (llvm::AllocaInst*)>);

    World& world_;
    llvm::LLVMContext context_;
    std::unique_ptr<llvm::TargetMachine> machine_;
    std::unique_ptr<llvm::Module> module_;
    llvm::IRBuilder<> irbuilder_;
    llvm::DIBuilder dibuilder_;
    llvm::CallingConv::ID function_calling_convention_;
    llvm::CallingConv::ID device_calling_convention_;
    llvm::CallingConv::ID kernel_calling_convention_;
    DefMap<llvm::Value*> params_;
    DefMap<llvm::PHINode*> phis_;
    DefMap<llvm::Value*> defs_;
    LamMap<llvm::Function*> fcts_;
    TypeMap<llvm::Type*> types_;
#if THORIN_ENABLE_RV
    std::vector<std::tuple<u32, llvm::Function*, llvm::CallInst*>> vec_todo_;
#endif

    std::unique_ptr<Runtime> runtime_;
    Lam* entry_ = nullptr;

    friend class Runtime;
};

//------------------------------------------------------------------------------

template<class T>
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

struct Backends {
    Backends(World& world);

    Cont2Config kernel_config;
    std::vector<Lam*> kernels;

    Importer cuda;
    Importer nvvm;
    Importer opencl;
    Importer amdgpu;
    Importer hls;

    std::unique_ptr<CodeGen> cpu_cg;
    std::unique_ptr<CodeGen> cuda_cg;
    std::unique_ptr<CodeGen> nvvm_cg;
    std::unique_ptr<CodeGen> opencl_cg;
    std::unique_ptr<CodeGen> amdgpu_cg;
    std::unique_ptr<CodeGen> hls_cg;
};

} // namespace thorin

#endif
