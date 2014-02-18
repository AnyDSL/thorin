#ifndef THORIN_BE_LLVM_OPENCL_H
#define THORIN_BE_LLVM_OPENCL_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class OpenCLCodeGen : public CodeGen {
public:
    OpenCLCodeGen(World& world);

    void emit();
};

}

#endif
