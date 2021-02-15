#ifndef THORIN_BE_C_H
#define THORIN_BE_C_H

#include <cstdint>
#include <iostream>

#include "thorin/be/backends.h"

namespace thorin {

class World;

namespace c_be {

enum class Lang : uint8_t {
    C99,        ///< Flag for C99
    HLS,        ///< Flag for HLS
    CUDA,       ///< Flag for CUDA
    OPENCL      ///< Flag for OpenCL
};

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World& world, const Cont2Config& kernel_config, Lang lang, bool debug)
    : thorin::CodeGen(world, debug)
    , kernel_config_(kernel_config)
    , lang_(lang)
    , debug_(debug) {}

    void emit(std::ostream& stream) override;

private:
    const Cont2Config& kernel_config_;
    Lang lang_;
    bool debug_;
};

void emit_c_int(World&, Stream& stream);

}

}

#endif
