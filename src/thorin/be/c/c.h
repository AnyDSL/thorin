#ifndef THORIN_BE_C_H
#define THORIN_BE_C_H

#include <cstdint>
#include <iostream>

#include "thorin/be/kernel_config.h"

namespace thorin {

class World;

namespace c_be {

enum class Lang : uint8_t {
    C99,        ///< Flag for C99
    HLS,        ///< Flag for HLS
    CUDA,       ///< Flag for CUDA
    OPENCL      ///< Flag for OpenCL
};

void emit_c(World&, const Cont2Config& kernel_config, std::ostream& stream, Lang lang, bool debug);
void emit_c_int(World&, Stream& stream);

}

}

#endif
