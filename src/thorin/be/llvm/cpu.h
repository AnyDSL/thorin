#ifndef THORIN_BE_LLVM_CPU_H
#define THORIN_BE_LLVM_CPU_H

#include "thorin/be/llvm/llvm.h"

namespace thorin::llvm {

namespace llvm = ::llvm;

class CPUCodeGen : public CodeGen {
public:
    CPUCodeGen(World& world, int opt, bool debug);

protected:
    std::string get_alloc_name() const override { return "anydsl_alloc"; }
};

}

#endif
