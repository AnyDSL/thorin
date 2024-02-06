#ifndef THORIN_BE_LLVM_AMDGPU_PAL_H
#define THORIN_BE_LLVM_AMDGPU_PAL_H

#include "thorin/be/llvm/amdgpu.h"

namespace thorin {

namespace llvm {

namespace llvm = ::llvm;

class AMDGPUPALCodeGen : public AMDGPUCodeGen {
public:
    AMDGPUPALCodeGen(World& world, const Cont2Config&, int opt, bool debug);

protected:
    llvm::Function* emit_fun_decl(Continuation*) override;
};

}

}

#endif
