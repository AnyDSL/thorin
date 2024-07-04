#ifndef THORIN_BE_LLVM_AMDGPU_HSA_H
#define THORIN_BE_LLVM_AMDGPU_HSA_H

#include "thorin/be/llvm/amdgpu.h"

namespace thorin {

namespace llvm {

namespace llvm = ::llvm;

class AMDGPUHSACodeGen : public AMDGPUCodeGen {
public:
    AMDGPUHSACodeGen(Thorin& thorin, const Cont2Config&, int opt, bool debug);

protected:
    llvm::Function* emit_fun_decl(Continuation*) override;
};

}

}

#endif
