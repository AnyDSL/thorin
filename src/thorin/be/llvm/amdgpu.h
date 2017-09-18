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
    virtual std::string get_output_name(const std::string& name) const override { return name + ".amdgpu"; }
};

}

#endif
