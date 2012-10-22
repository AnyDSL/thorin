#include "anydsl2/type.h"

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/world.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

const Ptr* Type::to_ptr() const { return world().ptr(this); }

//------------------------------------------------------------------------------

PrimType::PrimType(World& world, PrimTypeKind kind)
    : Type(world, kind, 0)
{
    debug = kind2str(primtype_kind());
}

//------------------------------------------------------------------------------

CompoundType::CompoundType(World& world, int kind, size_t num_generics, size_t num_elems)
    : Type(world, kind, num_generics + num_elems)
    , num_generics_(num_generics)
{}

CompoundType::CompoundType(World& world, int kind, ArrayRef<const Type*> elems)
    : Type(world, kind, elems.size())
    , num_generics_(0)
{
    size_t x = 0;
    for_all (elem, elems)
        set(x++, elem);
}

CompoundType::CompoundType(World& world, int kind, ArrayRef<const Generic*> generics, 
                                                   ArrayRef<const Type*> elems)
    : Type(world, kind, generics.size() + elems.size())
    , num_generics_(generics.size())
{
    size_t x = 0;
    for_all (generic, generics)
        set(x++, generic);
    for_all (elem, elems)
        set(x++, elem);
}

//------------------------------------------------------------------------------

size_t Sigma::hash() const {
    return named_ ? boost::hash_value(this) : CompoundType::hash();
}

bool Sigma::equal(const Node* other) const {
    return named_ ? this == other : CompoundType::equal(other);
}

//------------------------------------------------------------------------------

template<bool first_order>
bool Pi::classify_order() const {
    for_all (elem, elems())
        if (first_order ^ (elem->isa<Pi>() == 0))
            return false;

    return true;
}

bool Pi::is_fo() const { return classify_order<true>(); }
bool Pi::is_ho() const { return classify_order<false>(); }

//------------------------------------------------------------------------------

Generic::Generic(Lambda* lambda, size_t index)
    : Type(lambda->world(), Node_Generic, 0)
    , lambda_(lambda)
    , index_(index)
{}

size_t Generic::hash() const { return boost::hash_value(this); }
bool Generic::equal(const Node* other) const { return this == other; }

//------------------------------------------------------------------------------

} // namespace anydsl2
