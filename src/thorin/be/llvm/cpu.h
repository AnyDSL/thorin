#ifndef THORIN_BE_LLVM_CPU_H
#define THORIN_BE_LLVM_CPU_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class CPUCodeGen : public CodeGen {
public:
    CPUCodeGen(World& world, std::string cpu_target_name="");

    static std::vector<std::string> GetTargets();

protected:
    virtual std::string get_alloc_name() const override { return "anydsl_alloc"; }
};

}

#endif
