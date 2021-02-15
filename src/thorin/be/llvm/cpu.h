#ifndef THORIN_BE_LLVM_CPU_H
#define THORIN_BE_LLVM_CPU_H

#include "thorin/be/llvm/llvm.h"

namespace thorin::llvm_be {

class CPUCodeGen : public CodeGen {
public:
    CPUCodeGen(World& world, int opt, bool debug);

protected:
    virtual std::string get_alloc_name() const override { return "anydsl_alloc"; }
};

}

#endif
