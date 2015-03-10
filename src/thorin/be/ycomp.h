#ifndef YCOMP_BE_H
#define YCOMP_BE_H

#include <iostream>

#include "thorin/def.h"
#include "thorin/type.h"

namespace thorin {

class Scope;

void emit_ycomp(const Scope&, bool scheduled = false, std::ostream& = std::cout);
void emit_ycomp(const World&, bool scheduled = false, std::ostream& = std::cout);

}

#endif
