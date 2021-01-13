#ifndef THORIN_BE_LLVM_OPENCL_H
#define THORIN_BE_LLVM_OPENCL_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class OpenCLCodeGen : public CodeGen {
public:
    OpenCLCodeGen(World& world, const Cont2Config&, int opt, bool debug);

    void emit(std::ostream& stream) override;

protected:
    virtual std::string get_alloc_name() const override { THORIN_UNREACHABLE; /*alloc not supported in OpenCL*/; }

    const Cont2Config& kernel_config_;
};

}

#endif
