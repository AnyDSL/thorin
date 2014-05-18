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

void TypeNode::bind(TypeVar type_var) const {
    assert(!type_var->is_unified());
    type_vars_.push_back(type_var); 
    type_var->bound_at_ = this; 
}

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

void TypeNode::dump() const { emit_type(Type(this)); std::cout << std::endl; }
size_t TypeNode::length() const { return as<VectorTypeNode>()->length(); }
Type TypeNode::elem_via_lit(const Def& def) const { return elem(def->primlit_value<size_t>()); }
const TypeNode* TypeNode::unify() const { return world().unify_base(this); }
TypeVarSet TypeNode::free_type_vars() const { TypeVarSet bound, free; free_type_vars(bound, free); return free; }

void TypeNode::free_type_vars(TypeVarSet& bound, TypeVarSet& free) const {
    for (auto type_var : type_vars())
        bound.insert(*type_var);

    for (auto elem : elems()) {
        if (auto type_var = elem->isa<TypeVarNode>()) {
            if (!bound.contains(type_var))
                free.insert(type_var);
        } else
            elem->free_type_vars(bound, free);
    }
}

VectorType VectorTypeNode::scalarize() const {
    if (auto ptr = isa<PtrTypeNode>())
        return world().ptr_type(ptr->referenced_type());
    return world().type(as<PrimTypeNode>()->primtype_kind());
}

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

/*
 * hash
 */

size_t TypeNode::hash() const {
    size_t seed = hash_combine(hash_combine(hash_begin((int) kind()), size()), num_type_vars());
    for (auto elem : elems_)
        seed = hash_combine(seed, elem->hash());
    return seed;
}

size_t PtrTypeNode::hash() const {
    return hash_combine(hash_combine(VectorTypeNode::hash(), (size_t)device()), (size_t)addr_space());
}

//------------------------------------------------------------------------------

/*
 * equal
 */

bool TypeNode::equal(const TypeNode* other) const {
    bool result = this->kind() == other->kind() && this->size() == other->size() 
        && this->num_type_vars() == other->num_type_vars();

    if (result) {
        for (size_t i = 0, e = num_type_vars(); result && i != e; ++i) {
            assert(this->type_var(i)->equiv_ == nullptr);
            this->type_var(i)->equiv_ = *other->type_var(i);
        }

        for (size_t i = 0, e = size(); result && i != e; ++i)
            result &= this->elems_[i] == other->elems_[i];

        for (auto var : type_vars())
            var->equiv_ = nullptr;
    }

    return result;
}

bool PtrTypeNode::equal(const TypeNode* other) const {
    if(!VectorTypeNode::equal(other))
        return false;
    auto ptr = other->as<PtrTypeNode>();
    return ptr->device() == device() && ptr->addr_space() == addr_space();
}

bool TypeVarNode::equal(const TypeNode* other) {
    if (auto typevar = other->isa<TypeVarNode>())
        return this->equiv_ == typevar;
    return false;
}

//------------------------------------------------------------------------------

}
