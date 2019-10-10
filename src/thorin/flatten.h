#ifndef THORIN_FLATTEN_H
#define THORIN_FLATTEN_H

#include "thorin/def.h"

namespace thorin {

class Flattener {
public:
    /// Flattens a sigma/array/pack/tuple.
    const Def* flatten(const Def* def);

    Def2Def old2new;
};

/// Applies the reverse transformation on a pack/tuple, given the original type.
const Def* unflatten(const Def* def, const Def* type);
/// Same as unflatten, but uses the operands of a flattened pack/tuple directly.
const Def* unflatten(Defs ops, const Def* type);

}

#endif
