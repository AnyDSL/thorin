#ifndef THORIN_BE_LLVM_CUDA_H
#define THORIN_BE_LLVM_CUDA_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class CUDACodeGen : public CodeGen {
public:
    CUDACodeGen(World& world, const Cont2Config&);

    void emit(bool debug);

protected:
    virtual std::string get_alloc_name() const { return "malloc"; }
    virtual std::string get_output_name(const std::string& name) const { return name + ".cu"; }
};

}

#endif
