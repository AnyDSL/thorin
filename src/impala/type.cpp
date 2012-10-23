#include "impala/type.h"

using anydsl2::Pi;

namespace impala {

World::World() 
    : noret_(consume(new NoRet(*this))->as<NoRet>())
    , error_(consume(new TypeError(*this))->as<TypeError>())
{
    typekeeper(noret_);
    typekeeper(error_);
}

const anydsl2::Type* return_type(const anydsl2::Pi* pi) {
    if (!pi->empty()) {
        if (const Pi* ret = pi->elems().back()->isa<Pi>()) {
            if (ret->num_elems() == 1)
                return ret->elem(0);
        }
    }

    return ((impala::World&) pi->world()).noret();
}

} // namespace impala
