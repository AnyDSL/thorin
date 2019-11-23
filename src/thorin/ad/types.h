#ifndef THORIN_AD_TYPES_H
#define THORIN_AD_TYPES_H

#include <thorin/def.h>

namespace thorin {

/// \returns the tangent vector type to the given primal, nullptr if it cannot be determined.
const Def* tangent_vector_type(const Def* primal_type);

}

#endif // THORIN_AD_TYPES_H
