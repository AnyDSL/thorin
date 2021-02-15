#ifndef THORIN_BE_LLVM_CUDA_H
#define THORIN_BE_LLVM_CUDA_H

#include "c.h"

namespace thorin::c {

class CUDACodeGen : public CodeGen {
public:
    CUDACodeGen(World &world, const Cont2Config &, int opt, bool debug);
};

}

#endif
