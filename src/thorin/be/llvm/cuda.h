#ifndef THORIN_BE_LLVM_CUDA_H
#define THORIN_BE_LLVM_CUDA_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class CUDACodeGen : public CodeGen {
public:
    CUDACodeGen(World& world, const Cont2Config&);

    void emit(std::ostream& stream, int opt, bool debug) override;

protected:
    virtual std::string get_alloc_name() const { return "malloc"; }

    const Cont2Config& kernel_config_;
};

}

#endif
