#ifndef THORIN_BE_LLVM_CPU_H
#define THORIN_BE_LLVM_CPU_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class CPUCodeGen : public CodeGen {
public:
    CPUCodeGen(World& world)
        : CodeGen(world, llvm::CallingConv::C)
    {}

    virtual llvm::Function* emit_function_decl(std::string&, Lambda*);
};

}

#endif
