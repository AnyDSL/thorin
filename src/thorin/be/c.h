#ifndef THORIN_BE_C_H
#define THORIN_BE_C_H

#include <iostream>

namespace thorin {

class World;

void emit_c(World&, std::ostream& stream = std::cout);

}

#endif
