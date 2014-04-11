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

const Type*& GenericMap::operator [] (const Generic* generic) const {
    size_t i = generic->index();
    if (i >= types_.size())
        types_.resize(i+1, nullptr);
    return types_[i];
}

bool GenericMap::is_empty() const {
    for (size_t i = 0, e = types_.size(); i != e; ++i)
        if (!types_[i])
            return false;

    return true;
}

std::string GenericMap::to_string() const {
    std::ostringstream o;
    bool first = true;
    for (size_t i = 0, e = types_.size(); i != e; ++i)
        if (auto type = types_[i]) {
            if (first)
                first = false;
            else
                o << ", ";
            o << '_' << i << " = "; 
            type->dump();
        }

    return o.str();
}

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
const Type* Type::elem_via_lit(Def def) const { return elem(def->primlit_value<size_t>()); }

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

bool Type::check_with(const Type* other) const {
    if (this == other || this->isa<Generic>() || other->isa<Generic>())
        return true;

    if (this->kind() != other->kind() || this->size() != other->size())
        return false;

    for (size_t i = 0, e = size(); i != e; ++i)
        if (!this->elem(i)->check_with(other->elem(i)))
            return false;

    return true;
}

bool Type::infer_with(GenericMap& map, const Type* other) const {
    size_t num_elems = this->size();
    assert(this->isa<Generic>() || num_elems == other->size());
    assert(this->isa<Generic>() || this->kind() == other->kind());

    if (this == other)
        return true;

    if (auto generic = this->isa<Generic>()) {
        const Type*& mapped = map[generic];
        if (!mapped) {
            mapped = other;
            return true;
        } else
            return mapped == other;
    }

    for (size_t i = 0; i < num_elems; ++i) {
        if (!this->elem(i)->infer_with(map, other->elem(i)))
            return false;
    }

    return true;
}

const Type* Type::specialize(const GenericMap& map) const {
    if (auto generic = this->isa<Generic>()) {
        if (auto substitute = map[generic])
            return substitute;
        else
            return this;
    } else if (empty())
        return this;

    Array<const Type*> new_elems(size());
    for (size_t i = 0, e = size(); i != e; ++i)
        new_elems[i] = elem(i)->specialize(map);

    return world().rebuild(this, new_elems);
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
