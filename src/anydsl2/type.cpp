#include "anydsl2/type.h"

#include <sstream>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/printer.h"
#include "anydsl2/world.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

const Type*& GenericMap::operator [] (const Generic* generic) const {
    size_t i = generic->index();
    if (i >= types_.size())
        types_.resize(i+1, 0);
    return types_[i];
}

bool GenericMap::is_empty() const {
    for (size_t i = 0, e = types_.size(); i != e; ++i)
        if (!types_[i])
            return false;

    return true;
}

const char* GenericMap::to_string() const {
    std::ostringstream o;
    bool first = true;
    for (size_t i = 0, e = types_.size(); i != e; ++i)
        if (const Type* type = types_[i]) {
            if (first)
                first = false;
            else
                o << ", ";
            o << '_' << i << " = " << type;
        }

    return o.str().c_str();
}

//------------------------------------------------------------------------------

int Type::order() const {
    if (kind() == Node_Ptr)
        return 0;

    int sub = 0;
    for_all (elem, elems())
        sub = std::max(sub, elem->order());

    if (kind() == Node_Pi)
        return sub + 1;

    return sub;
}

const Ptr* Type::to_ptr(size_t length) const { return world().ptr(this, length); }
size_t Type::length() const { return as<VectorType>()->length(); }

const Type* Type::elem_via_lit(const Def* def) const {
    return elem(def->primlit_value<size_t>());
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
    size_t num_elems = this->size();
    assert(num_elems == other->size());
    assert(this->isa<Generic>() || this->kind() == other->kind());

    if (this == other)
        return true;

    if (const Generic* generic = this->isa<Generic>()) {
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

const Type* Type::specialize(const GenericMap& generic_map) const {
    if (const Generic* generic = this->isa<Generic>()) {
        if (const Type* substitute = generic_map[generic])
            return substitute;
        else
            return this;
    } else if (empty())
        return this;

    Array<const Type*> new_elems(size());
    for_all2 (&new_elem, new_elems, elem, elems())
        new_elem = elem->specialize(generic_map);

    return world().rebuild(this, new_elems);
}

//------------------------------------------------------------------------------

CompoundType::CompoundType(World& world, int kind, size_t size)
    : Type(world, kind, size, false /*TODO named sigma*/)
{}

CompoundType::CompoundType(World& world, int kind, ArrayRef<const Type*> elems)
    : Type(world, kind, elems.size(), false)
{
    size_t x = 0;
    for_all (elem, elems) {
        if (elem->is_generic())
            is_generic_ = true;
        set(x++, elem);
    }
}

//------------------------------------------------------------------------------

size_t Sigma::hash() const {
    return named_ ? boost::hash_value(this) : CompoundType::hash();
}

bool Sigma::equal(const Node* other) const {
    return named_ ? this == other : CompoundType::equal(other);
}

//------------------------------------------------------------------------------

bool Pi::is_returning() const {
    bool ret = false;
    for_all (elem, elems()) {
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

size_t GenericBuilder::new_def() {
    size_t handle = index2generic_.size();
    index2generic_.push_back(0);
    return handle;
}

const Generic* GenericBuilder::use(size_t handle) {
    assert(handle < index2generic_.size());
    const Generic*& ref = index2generic_[handle];
    if (const Generic* generic = ref)
        return generic;

    return ref = world_.generic(index_++);
}

void GenericBuilder::pop() { 
    if (const Generic* generic = index2generic_.back()) {
        --index_;
        assert(generic->index() == index_);
    }
    index2generic_.pop_back(); 
}

//------------------------------------------------------------------------------

} // namespace anydsl2
