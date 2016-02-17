#ifndef THORIN_TRANSFORM_IMPORT_H
#define THORIN_TRANSFORM_IMPORT_H

#include "thorin/def.h"

namespace thorin {

Type import(World& to, Type);
const Def* import(World& to, const Def*);

}

#endif
