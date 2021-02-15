#ifndef THORIN_BE_LLVM_OPENCL_H
#define THORIN_BE_LLVM_OPENCL_H

#include "c.h"

namespace thorin::c {

class OpenCLCodeGen : public CodeGen {
public:
    OpenCLCodeGen(World& world, const Cont2Config&, int opt, bool debug);
};

}

#endif
