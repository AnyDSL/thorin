#ifndef THORIN_BE_LLVM_NVVM_H
#define THORIN_BE_LLVM_NVVM_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class NVVMCodeGen : public CodeGen {
public:
    NVVMCodeGen(World& world)
        : CodeGen(world, llvm::CallingConv::PTX_Device)
    {}

    virtual llvm::Function* emit_function_decl(std::string&, Lambda*);
};

}

#endif
