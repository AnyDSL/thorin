#ifndef THORIN_BE_LLVM_HLS_H
#define THORIN_BE_LLVM_HLS_H

#include "thorin/be/llvm/llvm.h"

namespace thorin::llvm_be {

class HLSCodeGen : public CodeGen {
public:
    HLSCodeGen(World& world, const Cont2Config&, int opt, bool debug);

    void emit(std::ostream& stream) override;

protected:
    virtual std::string get_alloc_name() const override { THORIN_UNREACHABLE; /*alloc not supported in HLS*/; }

    const Cont2Config& kernel_config_;
};

}

#endif
