#include "thorin/type.h"

#include <algorithm>
#include <iostream>
#include <sstream>

#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/world.h"
#include "thorin/be/thorin.h"

namespace thorin {

//------------------------------------------------------------------------------

size_t Type::hash() const {
    size_t seed = hash_combine(hash_begin((int) kind()), size());
    for (auto elem : elems_)
        seed = hash_combine(seed, elem->gid());
    return seed;
}

bool Type::equal(const Type* other) const {
    bool result = this->kind() == other->kind() && this->size() == other->size();
    for (size_t i = 0, e = size(); result && i != e; ++i)
        result &= this->elems_[i] == other->elems_[i];
    return result;
}

void Type::dump() const { emit_type(this); std::cout << std::endl; }
size_t Type::length() const { return as<VectorType>()->length(); }
const Type* Type::elem_via_lit(const Def& def) const { return elem(def->primlit_value<size_t>()); }

int Type::order() const {
    if (kind() == Node_Ptr)
        return 0;

    int sub = 0;
    for (auto elem : elems())
        sub = std::max(sub, elem->order());

    if (kind() == Node_Pi)
        return sub + 1;

    return sub;
}

//------------------------------------------------------------------------------

const VectorType* VectorType::scalarize() const {
    if (auto ptr = isa<Ptr>())
        return world().ptr(ptr->referenced_type());
    return world().type(as<PrimType>()->primtype_kind());
}

//------------------------------------------------------------------------------

size_t Ptr::hash() const {
    return hash_combine(hash_combine(VectorType::hash(), (size_t)device()), (size_t)addr_space());
}

bool Ptr::equal(const Type* other) const {
    if(!VectorType::equal(other))
        return false;
    auto ptr = other->as<Ptr>();
    return ptr->device() == device() && ptr->addr_space() == addr_space();
}

//------------------------------------------------------------------------------

CompoundType::CompoundType(World& world, NodeKind kind, size_t size)
    : Type(world, kind, size, false /*TODO named sigma*/)
{}

CompoundType::CompoundType(World& world, NodeKind kind, ArrayRef<const Type*> elems)
    : Type(world, kind, elems.size(), false)
{
    size_t x = 0;
    for (auto elem : elems) {
        if (elem->is_generic())
            is_generic_ = true;
        set(x++, elem);
    }
}

//------------------------------------------------------------------------------

bool Pi::is_returning() const {
    bool ret = false;
    for (auto elem : elems()) {
        switch (elem->order()) {
            case 0: continue;
            case 1:
                if (!ret) {
                    ret = true;
                    continue;
                } // else fall-through
            default:
                return false;
        }
    }
    return true;
}

//------------------------------------------------------------------------------

}
