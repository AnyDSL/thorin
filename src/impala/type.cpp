#include "impala/type.h"

using anydsl2::Pi;

namespace impala {

World::World() 
    : noret_(consume(new NoRet(*this))->as<NoRet>())
    , type_void_(consume(new Void(*this))->as<Void>())
    , type_error_(consume(new TypeError(*this))->as<TypeError>())
{
    typekeeper(noret_);
    typekeeper(type_void_);
    typekeeper(type_error_);
}

const anydsl2::Type* return_type(const anydsl2::Pi* pi) {
    if (!pi->empty()) {
        if (const Pi* ret = pi->elems().back()->isa<Pi>()) {
            if (ret->num_elems() == 1)
                return ret->elem(0);
            if (ret->num_elems() == 0)
                return ((impala::World&) pi->world()).type_void();
        }
    }

    return ((impala::World&) pi->world()).noret();
}

} // namespace impala
