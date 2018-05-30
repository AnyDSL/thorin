#ifndef THORIN_BE_LLVM_NVPTX_H
#define THORIN_BE_LLVM_NVPTX_H

#include "thorin/be/llvm/nvvm.h"

namespace thorin {

class NVPTXCodeGen : public NVVMCodeGen {
public:
    NVPTXCodeGen(World& world, const Cont2Config&);

protected:
    void emit(std::ostream& stream, int opt, bool debug) override;
};

}

#endif
