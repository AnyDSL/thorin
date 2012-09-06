#include "anydsl/type.h"

#include "anydsl/literal.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind kind)
    : Type(world, kind, 0)
{
    debug = kind2str(this->primtype_kind());
}

//------------------------------------------------------------------------------

CompoundType::CompoundType(World& world, int kind, size_t num)
    : Type(world, kind, num)
{}

CompoundType::CompoundType(World& world, int kind, ArrayRef<const Type*> elems)
    : Type(world, kind, elems.size())
{
    size_t x = 0;
    for_all (elem, elems)
        setOp(x++, elem);
}

//------------------------------------------------------------------------------

size_t Sigma::hash() const {
    if (named_)
        return boost::hash_value(this);
    else
        return CompoundType::hash();
}

bool Sigma::equal(const Def* other) const {
    if (named_)
        return this == other;
    else
        return CompoundType::equal(other);
}

//------------------------------------------------------------------------------

size_t Pi::ho_begin() const {
    for (size_t i = 0, e = size(); i != e; ++i)
        if (elem(i)->isa<Pi>())
            return i;

    return ho_end();
}

void Pi::ho_next(size_t& i) const {
    ++i;

    while (i < size())
        if (elem(i)->isa<Pi>())
            return;
        else
            ++i;
}

size_t Pi::fo_begin() const {
    for (size_t i = 0, e = size(); i != e; ++i)
        if (!elem(i)->isa<Pi>())
            return i;

    return fo_end();
}

void Pi::fo_next(size_t& i) const {
    ++i;

    while (i < size())
        if (!elem(i)->isa<Pi>())
            return;
        else
            ++i;
}

//------------------------------------------------------------------------------

} // namespace anydsl
