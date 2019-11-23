#ifndef THORIN_AD_PULLBACK_H
#define THORIN_AD_PULLBACK_H

#include <thorin/def.h>

namespace thorin {

const Def* pullback_fn(const Def* fn);

const Def* j_call(const Def* fn);

}

#endif // THORIN_AD_PULLBACK_H
