#ifndef THORIN_BE_LLVM_AMDGPU_H
#define THORIN_BE_LLVM_AMDGPU_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class Load;

class AMDGPUCodeGen : public CodeGen {
public:
    AMDGPUCodeGen(World& world, const Cont2Config&);

protected:
    virtual void emit_function_decl_hook(Lam*, llvm::Function*) override;
    virtual unsigned convert_addr_space(const AddrSpace) override;
    virtual llvm::Value* emit_global(const Global*) override;
    virtual Lam* emit_reserve(const Lam*) override;
    virtual std::string get_alloc_name() const override { return "malloc"; }

    const Cont2Config& kernel_config_;
};

}

#endif
