#include "anydsl2/type.h"

#include <algorithm>
#include <sstream>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/world.h"
#include "anydsl2/be/air.h"

namespace anydsl2 {

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
    if (this == other || this->isa<Generic>())
        return true;

    if (this->kind() != other->kind() || this->size() != other->size())
        return false;

    for (size_t i = 0, e = size(); i != e; ++i)
        if (!this->elem(i)->check_with(other->elem(i)))
            return false;

    return true;
}

bool Type::infer_with(GenericMap& map, const Type* other) const {
    if (auto genericref = this->isa<GenericRef>())
        return genericref->generic()->infer_with(map, other);
    if (auto genericref = other->isa<GenericRef>())
        other = genericref->generic();

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

    if (auto genref = this->isa<GenericRef>())
        return genref->generic()->specialize(map);

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

CompoundType::CompoundType(World& world, int kind, size_t size)
    : Type(world, kind, size, false /*TODO named sigma*/)
{}

CompoundType::CompoundType(World& world, int kind, ArrayRef<const Type*> elems)
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

GenericRef::GenericRef(World& world, const Generic* generic, Lambda* lambda)
    : Type(world, Node_GenericRef, 1, true)
    , lambda_(lambda)
{
#if 0
    lambda_->generic_refs_.push_back(this);
#endif
    set(0, generic);
}

GenericRef::~GenericRef() {
#if 0
    auto& generic_refs = lambda()->generic_refs_;
    auto i = std::find(generic_refs.begin(), generic_refs.end(), this);
    assert(i != generic_refs.end() && "must be in use set");
    *i = generic_refs.back();
    generic_refs.pop_back();
#endif
}

//------------------------------------------------------------------------------

} // namespace anydsl2
