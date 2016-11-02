#ifndef THORIN_TRANSFORM_IMPORT_H
#define THORIN_TRANSFORM_IMPORT_H

#include "thorin/def.h"

namespace thorin {

const Def* import(World& to, const Def*);
const Def* import(World& to, Type2Type&, Def2Def&, const Def*);

}

#endif
