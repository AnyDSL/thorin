#ifndef THORIN_BE_C_H
#define THORIN_BE_C_H

#include <cstdint>
#include <iostream>

#include "thorin/world.h"
#include "thorin/be/kernel_config.h"

namespace thorin {

class World;

enum class Lang : uint8_t {
    C99,    ///< Flag for C99
    HLS,    ///< Flag for HLS
    CUDA,   ///< Flag for CUDA
    OPENCL  ///< Flag for OpenCL
};

void emit_c(World&, const Cont2Config& kernel_config, std::ostream& stream, Lang lang, bool debug);

}

#endif
