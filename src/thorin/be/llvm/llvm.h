#ifndef THORIN_BE_LLVM_LLVM_H
#define THORIN_BE_LLVM_LLVM_H

#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

#include "thorin/config.h"
#include "thorin/continuation.h"
#include "thorin/analyses/schedule.h"
#include "thorin/be/codegen.h"
#include "thorin/be/emitter.h"
#include "thorin/be/llvm/runtime.h"
#include "thorin/be/kernel_config.h"

namespace thorin {

class World;

namespace llvm {

namespace llvm = ::llvm;

using BB = std::pair<llvm::BasicBlock*, std::unique_ptr<llvm::IRBuilder<>>>;

class CodeGen : public thorin::CodeGen, public thorin::Emitter<llvm::Value*, llvm::Type*, BB, CodeGen> {
protected:
    CodeGen(
        Thorin&,
        llvm::CallingConv::ID function_calling_convention,
        llvm::CallingConv::ID device_calling_convention,
        llvm::CallingConv::ID kernel_calling_convention,
        int opt, bool debug);
public:
    virtual ~CodeGen() {}

    /// @name getters
    //@{
    llvm::LLVMContext& context() { return *context_; }
    llvm::Module& module() { return *module_; }
    const llvm::Module& module() const { return *module_; }
    llvm::TargetMachine& machine() { return *machine_; }
    int opt() const { return opt_; }
    //@}

    const char* file_ext() const override { return ".ll"; }
    void emit_stream(std::ostream& stream) override;
    // Note: This moves the context and module of the class,
    // rendering the current CodeGen object invalid.
    std::pair<
        std::unique_ptr<llvm::LLVMContext>,
        std::unique_ptr<llvm::Module>> emit_module();
    llvm::Function* prepare(const Scope&);
    virtual void prepare(Continuation*, llvm::Function*);
    llvm::Value* emit_constant(const Def* def);
    llvm::Value* emit_bb(BB&, const Def* def);
    llvm::Value* emit_builder(llvm::IRBuilder<>&, const Def* def);
    virtual llvm::Function* emit_fun_decl(Continuation*);
    bool is_valid(llvm::Value* value) { return value != nullptr; }
    void finalize(const Scope&);
    void finalize(Continuation*) {}
    void emit_epilogue(Continuation*);

protected:
    /// @name convert
    //@{
    llvm::Type* convert(const Type*);
    virtual unsigned convert_addr_space(const AddrSpace);
    virtual llvm::FunctionType* convert_fn_type(Continuation*);
    //@}

    llvm::AllocaInst* emit_alloca(llvm::IRBuilder<>&, llvm::Type*, const std::string&);
    llvm::Value*      emit_alloc (llvm::IRBuilder<>&, const Type*, const Def*);
    virtual void emit_fun_decl_hook(Continuation*, llvm::Function*) {}
    virtual llvm::Value* map_param(llvm::Function*, llvm::Argument* a, const Param*) { return a; }

    virtual llvm::Value* emit_mathop  (llvm::IRBuilder<>&, const MathOp*);
    virtual llvm::Value* emit_load    (llvm::IRBuilder<>&, const Load*);
    virtual llvm::Value* emit_store   (llvm::IRBuilder<>&, const Store*);
    virtual llvm::Value* emit_lea     (llvm::IRBuilder<>&, const LEA*);
    virtual llvm::Value* emit_assembly(llvm::IRBuilder<>&, const Assembly* assembly);

    virtual Continuation* emit_reserve(llvm::IRBuilder<>&, const Continuation*);
    Continuation* emit_reserve_shared(llvm::IRBuilder<>&, const Continuation*, bool=false);

    virtual std::string get_alloc_name() const = 0;
    llvm::BasicBlock* cont2bb(Continuation* cont) { return cont2bb_[cont].first; }

    virtual llvm::Value* emit_global(const Global*);
    llvm::GlobalVariable* emit_global_variable(llvm::Type*, const std::string&, unsigned, bool=false);

    void optimize();
    void verify() const;
    void create_loop(llvm::IRBuilder<>&, llvm::Value*, llvm::Value*, llvm::Value*, llvm::Function*, std::function<void(llvm::Value*)>);
    llvm::Value* create_tmp_alloca(llvm::IRBuilder<>&, llvm::Type*, std::function<llvm::Value* (llvm::AllocaInst*)>);

    llvm::Value* call_math_function(llvm::IRBuilder<>&, const MathOp*, const std::string&);

private:
    Continuation* emit_peinfo(llvm::IRBuilder<>&, Continuation*);
    Continuation* emit_intrinsic(llvm::IRBuilder<>&, Continuation*);
    Continuation* emit_hls(llvm::IRBuilder<>&, Continuation*);
    Continuation* emit_parallel(llvm::IRBuilder<>&, Continuation*);
    Continuation* emit_fibers(llvm::IRBuilder<>&, Continuation*);
    Continuation* emit_spawn(llvm::IRBuilder<>&, Continuation*);
    Continuation* emit_sync(llvm::IRBuilder<>&, Continuation*);
    Continuation* emit_vectorize_continuation(llvm::IRBuilder<>&, Continuation*);
    Continuation* emit_atomic(llvm::IRBuilder<>&, Continuation*);
    Continuation* emit_cmpxchg(llvm::IRBuilder<>&, Continuation*, bool);
    Continuation* emit_fence(llvm::IRBuilder<>&, Continuation*);
    Continuation* emit_atomic_load(llvm::IRBuilder<>&, Continuation*);
    Continuation* emit_atomic_store(llvm::IRBuilder<>&, Continuation*);
    llvm::Value* emit_bitcast(llvm::IRBuilder<>&, const Def*, const Type*);
    void emit_vectorize(u32, llvm::Function*, llvm::CallInst*);
    void emit_phi_arg(llvm::IRBuilder<>&, const Param*, llvm::Value*);

    // Note: The module and context have to be stored as pointers, so
    // that ownership of the module can be moved to the JIT (LLVM currently
    // has no std::move or std::swap implementation for modules and contexts).
    std::unique_ptr<llvm::LLVMContext> context_;
    std::unique_ptr<llvm::Module> module_;

    int opt_;

protected:
    std::unique_ptr<llvm::TargetMachine> machine_;
    llvm::DIBuilder dibuilder_;
    llvm::DICompileUnit* dicompile_unit_;
    llvm::CallingConv::ID function_calling_convention_;
    llvm::CallingConv::ID device_calling_convention_;
    llvm::CallingConv::ID kernel_calling_convention_;
    llvm::DIScope* discope_ = nullptr;
    std::unique_ptr<Runtime> runtime_;
    std::optional<BB> entry_prelude_;
    std::optional<BB> entry_prelude_end_;
    llvm::Value* return_buf_;
    bool has_alloca_;
    std::vector<llvm::CallInst*> potential_tailcalls_;
#if THORIN_ENABLE_RV
    std::vector<std::tuple<u32, llvm::Function*, llvm::CallInst*>> vec_todo_;
#endif

    friend class Runtime;
};

//------------------------------------------------------------------------------

template<class T>
llvm::ArrayRef<T> llvm_ref(const Array<T>& array) { return llvm::ArrayRef<T>(array.begin(), array.end()); }

} // namespace llvm_be

} // namespace thorin

#endif
