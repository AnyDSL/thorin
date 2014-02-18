#ifndef THORIN_BE_C_H
#define THORIN_BE_C_H

#include <iostream>

namespace thorin {

class World;

enum LangType {
    C99     = 1 << 0,   ///< Flag for C99
    OPENCL  = 1 << 1    ///< Flag for OpenCL
};

void emit_c(World&, std::ostream& stream = std::cout, LangType lang = C99);

}

#endif
