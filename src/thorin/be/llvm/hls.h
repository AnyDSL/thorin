#ifndef THORIN_BE_LLVM_HLS_H
#define THORIN_BE_LLVM_HLS_H

#include "thorin/be/llvm/llvm.h"

namespace thorin {

class HLSCodeGen : public CodeGen {
public:
    HLSCodeGen(World& world, const Cont2Config&);

    void emit(bool debug);

protected:
    virtual std::string get_alloc_name() const { THORIN_UNREACHABLE; /*alloc not supported in HLS*/; }
    virtual std::string get_output_name(const std::string& name) const { return name + ".hls"; }

    const Cont2Config& kernel_config_;
};

}

#endif
