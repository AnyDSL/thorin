#ifndef THORIN_BE_LLVM_OPENCL_H
#define THORIN_BE_LLVM_OPENCL_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class OpenCLCodeGen : public CodeGen {
public:
    OpenCLCodeGen(World& world);

    void emit();

protected:
    virtual std::string get_output_name(const std::string& name) const { return name + ".cl"; }
    virtual std::string get_binary_output_name(const std::string& name) const { return name + ".cl.bc"; }
};

}

#endif
