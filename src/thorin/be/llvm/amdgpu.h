#ifndef THORIN_BE_LLVM_AMDGPU_H
#define THORIN_BE_LLVM_AMDGPU_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class Load;

namespace llvm {

namespace llvm = ::llvm;

class AMDGPUCodeGen : public CodeGen {
public:
    AMDGPUCodeGen(World& world, const Lam2Config&, int opt, bool debug);

    const char* file_ext() const override { return ".amdgpu"; }

protected:
    void emit_fun_decl_hook(Lam*, llvm::Function*) override;
    llvm::Function* emit_fun_decl(Lam*) override;
    llvm::Value* emit_global(const Global*) override;
    llvm::Value* emit_mathop(llvm::IRBuilder<>&, const MathOp*) override;
    Lam* emit_reserve(llvm::IRBuilder<>&, const Lam*) override;
    std::string get_alloc_name() const override { return "malloc"; }

    const Lam2Config& kernel_config_;
};

}

}

#endif
