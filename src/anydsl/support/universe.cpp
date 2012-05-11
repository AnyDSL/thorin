#include "anydsl/support/universe.h"

namespace anydsl {

Universe::Universe() 
    : dummy_(false)
#define ANYDSL_U_TYPE(T) ,T##_(new PrimType(*this, PrimType_##T, #T))
#define ANYDSL_F_TYPE(T) ,T##_(new PrimType(*this, PrimType_##T, #T))
#include "anydsl/tables/primtypetable.h"
{
}

Universe::~Universe() {
    for (size_t i = 0; i < Num_PrimTypes; ++i)
        delete primTypes_[i];
}

} // namespace anydsl
