#include "anydsl/type.h"

#include "anydsl/literal.h"

namespace anydsl {

bool Type::equal(const Type* other) const {
    if (this->index() != other->index())
        return false;

    return true;
}

size_t Type::hash() const {
    size_t seed = 0;
    boost::hash_combine(seed, index());

    return seed;
}


bool CompoundType::equal(const Type* other) const {
    if (!Type::equal(other))
        return false;

    const CompoundType* c = other->as<CompoundType>();

    if (this->types().size() != c->types().size())
        return false;

    bool result = true;
    for (size_t i = 0, e = types().size(); i != e && result; ++i)
        result &= this->get(i) == c->get(i);

    return result;
}

size_t CompoundType::hash() const {
    size_t seed = Type::hash();

    boost::hash_combine(seed, types().size());

    for (size_t i = 0, e = types().size(); i != e; ++i)
        boost::hash_combine(seed, get(i));

    return seed;
}


//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind kind)
    : Type(world, (IndexKind) kind)
{
    debug = kind2str(this->kind());
}

//------------------------------------------------------------------------------

const Type* CompoundType::get(PrimLit* c) const { 
    anydsl_assert(isInteger(lit2type(c->kind())), "must be an integer constant");
    return get(c->box().get_u64()); 
}

//------------------------------------------------------------------------------

} // namespace anydsl
