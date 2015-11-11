#ifndef THORIN_BE_AIR_H
#define THORIN_BE_AIR_H

#include <iostream>

#include "thorin/def.h"
#include "thorin/type.h"

namespace thorin {

class Scope;

std::ostream& emit_thorin(const Scope&, std::ostream& = std::cout);
std::ostream& emit_thorin(const World&, std::ostream& = std::cout);
//std::ostream& emit_type(Type, std::ostream& = std::cout);
//std::ostream& emit_def(Def, std::ostream& = std::cout);
//std::ostream& emit_name(Def, std::ostream& = std::cout);
//std::ostream& emit_assignment(const PrimOp*, std::ostream& = std::cout);
std::ostream& emit_head(const Lambda*, std::ostream& = std::cout);
std::ostream& emit_jump(const Lambda*, std::ostream& = std::cout);

}

#endif
