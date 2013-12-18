#ifndef THORIN_BE_LLVM_NVVM_H
#define THORIN_BE_LLVM_NVVM_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class NVVMCodeGen : public CodeGen {
public:
    NVVMCodeGen(World& world);

    virtual llvm::Function* emit_function_decl(std::string&, Lambda*);
    virtual llvm::Function* emit_intrinsic_decl(std::string& name, Lambda* lambda);
};

}

#endif
