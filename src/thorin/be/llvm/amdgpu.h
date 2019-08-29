#ifndef THORIN_BE_LLVM_AMDGPU_H
#define THORIN_BE_LLVM_AMDGPU_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class Load;

class AMDGPUCodeGen : public CodeGen {
public:
    AMDGPUCodeGen(World& world, const Cont2Config&);

protected:
    virtual void emit_function_decl_hook(Continuation*, llvm::Function*) override;
    virtual unsigned convert_addr_space(const AddrSpace) override;
    virtual llvm::Value* emit_global(const Global*) override;
    virtual Continuation* emit_reserve(const Continuation*) override;
    virtual std::string get_alloc_name() const override { return "malloc"; }
    virtual llvm::AtomicOrdering get_atomic_ordering() const override { return llvm::AtomicOrdering::Monotonic; }
    virtual llvm::SyncScope::ID get_atomic_sync_scope(const AddrSpace) const override;

    const Cont2Config& kernel_config_;
};

}

#endif
