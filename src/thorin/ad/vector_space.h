#ifndef THORIN_AD_VECTOR_SPACE_H
#define THORIN_AD_VECTOR_SPACE_H

#include <thorin/def.h>

namespace thorin {

/// \returns the 1 of the given vector_type, nullptr if it is no tangent vector.
const Def* tangent_vector_lit_one(const Def* vector_type);

}

#endif // THORIN_AD_VECTOR_SPACE_H
