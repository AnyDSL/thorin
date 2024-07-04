#ifndef THORIN_BE_LLVM_AMDGPU_H
#define THORIN_BE_LLVM_AMDGPU_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

namespace llvm {

namespace llvm = ::llvm;

class AMDGPUCodeGen : public CodeGen {
public:
    AMDGPUCodeGen(Thorin&, llvm::CallingConv::ID, llvm::CallingConv::ID, llvm::CallingConv::ID, const Cont2Config&, int opt, bool debug);

    const char* file_ext() const override { return ".amdgpu"; }

protected:
    void emit_fun_decl_hook(Continuation*, llvm::Function*) override;
    virtual llvm::Function* emit_fun_decl(Continuation*) = 0;
    llvm::Value* emit_global(const Global*) override;
    llvm::Value* emit_mathop(llvm::IRBuilder<>&, const MathOp*) override;
    llvm::Value* emit_reserve(llvm::IRBuilder<>&, const Continuation*) override;
    std::string get_alloc_name() const override { return "malloc"; }

    const Cont2Config& kernel_config_;
};

}

}

#endif
