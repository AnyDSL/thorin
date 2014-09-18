#ifndef THORIN_BE_LLVM_CPU_H
#define THORIN_BE_LLVM_CPU_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class CPUCodeGen : public CodeGen {
public:
    CPUCodeGen(World& world);

protected:
    virtual std::string get_alloc_name() const override { return "thorin_malloc"; }
    virtual std::string get_output_name(const std::string& name) const override { return name + ".ll"; }
    virtual std::string get_binary_output_name(const std::string& name) const override { return name + ".bc"; }
};

}

#endif
