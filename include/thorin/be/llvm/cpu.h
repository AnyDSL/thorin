#ifndef THORIN_BE_LLVM_CPU_H
#define THORIN_BE_LLVM_CPU_H

#include "thorin/be/llvm/llvm.h"

namespace thorin::llvm {

namespace llvm = ::llvm;

class CPUCodeGen : public CodeGen {
public:
    CPUCodeGen(Thorin&, int opt, bool debug, std::string& target_triple, std::string& target_cpu, std::string& target_attr);

protected:
    std::string get_alloc_name() const override { return "anydsl_alloc"; }
};

}

#endif
