#ifndef THORIN_TRANSFORM_IMPORT_H
#define THORIN_TRANSFORM_IMPORT_H

#include "thorin/def.h"

namespace thorin {

const Type* import(World& to, const Type* otype);
Def import(World& to, Def odef);

}

#endif
