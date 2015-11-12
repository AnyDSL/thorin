#ifndef THORIN_BE_AIR_H
#define THORIN_BE_AIR_H

#include <iostream>

#include "thorin/def.h"
#include "thorin/type.h"

namespace thorin {

class Scope;

std::ostream& emit_head(const Lambda*, std::ostream& = std::cout);
std::ostream& emit_jump(const Lambda*, std::ostream& = std::cout);

}

#endif
