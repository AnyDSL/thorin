#ifndef THORIN_BE_LLVM_OPENCL_H
#define THORIN_BE_LLVM_OPENCL_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class OpenCLCodeGen : public CodeGen {
public:
    OpenCLCodeGen(World& world, const Cont2Config&);

    void emit(bool debug);

protected:
    virtual std::string get_alloc_name() const { THORIN_UNREACHABLE; /*alloc not supported in OpenCL*/; }
    virtual std::string get_output_name(const std::string& name) const { return name + ".cl"; }

    const Cont2Config& kernel_config_;
};

}

#endif
