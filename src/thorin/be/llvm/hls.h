#ifndef THORIN_BE_LLVM_HLS_H
#define THORIN_BE_LLVM_HLS_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class HLSCodeGen : public CodeGen {
public:
    HLSCodeGen(World& world, const Cont2Config&);

    void emit(std::ostream& stream, int opt, bool debug) override;

protected:
    virtual std::string get_alloc_name() const { THORIN_UNREACHABLE; /*alloc not supported in HLS*/; }

    const Cont2Config& kernel_config_;
};

}

#endif
