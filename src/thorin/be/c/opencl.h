#ifndef THORIN_BE_LLVM_OPENCL_H
#define THORIN_BE_LLVM_OPENCL_H

#include "../backends.h"
#include "../llvm/llvm.h"

namespace thorin::c_be {

class OpenCLCodeGen : public thorin::CodeGen {
public:
    OpenCLCodeGen(World& world, const Cont2Config&, int opt, bool debug);

    void emit(std::ostream& stream) override;

protected:
    const Cont2Config& kernel_config_;
};

}

#endif
