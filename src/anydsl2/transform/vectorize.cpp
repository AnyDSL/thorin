#include "anydsl2/transform/vectorize.h"

#include "anydsl2/type.h"
#include "anydsl2/world.h"

namespace anydsl2 {

const Type* vectorize(const Type* type, size_t length) {
    World& world = type->world();

    if (const PrimType* primtype = type->isa<PrimType>()) {
        assert(primtype->length() == 1);
        return world.type(primtype->primtype_kind(), length);
    }

    if (const Ptr* ptr = type->isa<Ptr>()) {
        assert(ptr->length() == 1);
        return world.ptr(ptr->referenced_type(), length);
    }

    Array<const Type*> new_elems(type->size());
    for_all2 (&new_elem, new_elems, elem, type->elems())
        new_elem = vectorize(elem, length);

    return world.rebuild(type, new_elems);
}

}
