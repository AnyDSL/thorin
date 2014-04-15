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

size_t TypeNode::hash() const {
    size_t seed = hash_combine(hash_begin((int) kind()), size());
    for (auto elem : elems_)
        seed = hash_combine(seed, elem->gid());
    return seed;
}

bool TypeNode::equal(const TypeNode* other) const {
    bool result = this->kind() == other->kind() && this->size() == other->size();
    for (size_t i = 0, e = size(); result && i != e; ++i)
        result &= this->elems_[i] == other->elems_[i];
    return result;
}

void TypeNode::dump() const { emit_type(Type(this)); std::cout << std::endl; }
size_t TypeNode::length() const { return as<VectorTypeNode>()->length(); }
Type TypeNode::elem_via_lit(const Def& def) const { return elem(def->primlit_value<size_t>()); }

int TypeNode::order() const {
    if (kind() == Node_PtrType)
        return 0;

    int sub = 0;
    for (auto elem : elems())
        sub = std::max(sub, elem->order());

    if (kind() == Node_FnType)
        return sub + 1;

    return sub;
}

//------------------------------------------------------------------------------

VectorType VectorTypeNode::scalarize() const {
    if (auto ptr = isa<PtrTypeNode>())
        return world().ptr_type(ptr->referenced_type());
    return world().type(as<PrimTypeNode>()->primtype_kind());
}

//------------------------------------------------------------------------------

size_t PtrTypeNode::hash() const {
    return hash_combine(hash_combine(VectorTypeNode::hash(), (size_t)device()), (size_t)addr_space());
}

bool PtrTypeNode::equal(const TypeNode* other) const {
    if(!VectorTypeNode::equal(other))
        return false;
    auto ptr = other->as<PtrTypeNode>();
    return ptr->device() == device() && ptr->addr_space() == addr_space();
}

//------------------------------------------------------------------------------

CompoundTypeNode::CompoundTypeNode(World& world, NodeKind kind, ArrayRef<Type> elems)
    : TypeNode(world, kind, elems.size(), false)
{
    size_t x = 0;
    for (auto elem : elems) {
        if (elem->is_generic())
            is_generic_ = true;
        set(x++, elem);
    }
}

//------------------------------------------------------------------------------

bool FnTypeNode::is_returning() const {
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
