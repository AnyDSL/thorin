#include "anydsl2/type.h"

#include <sstream>
#include <typeinfo>

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

size_t Type::hash() const {
    size_t seed = 0;
    boost::hash_combine(seed, kind());
    boost::hash_combine(seed, size());
    for_all (elem, elems())
        boost::hash_combine(seed, elem);

    return seed;
}

bool Type::equal(const Type* other) const {
    if (typeid(*this) == typeid(*other) && this->size() == other->size()) {
        for_all2 (this_elem, this->elems(), other_elem, other->elems()) {
            if (this_elem != other_elem)
                return false;
        }
        return true;
    }
    return false;
}

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

const Ptr* Type::to_ptr() const { return world().ptr(this); }

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
    for (size_t i = 0, e = size(); i != e; ++i)
        new_elems[i] = elem(i)->specialize(generic_map);

    // TODO better OO here
    switch (kind()) {
        case Node_Pi:    return world().pi(new_elems);
        case Node_Sigma: return world().sigma(new_elems);
        case Node_Ptr:   assert(new_elems.size() == 1); return world().ptr(new_elems.front());
        default: ANYDSL2_UNREACHABLE;
    }
}

//------------------------------------------------------------------------------

CompoundType::CompoundType(World& world, int kind, size_t size)
    : Type(world, kind, size, false /*TODO named sigma*/)
{}

CompoundType::CompoundType(World& world, int kind, Elems elems)
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

bool Sigma::equal(const Type* other) const {
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

size_t Generic::hash() const { 
    size_t seed = Type::hash(); 
    boost::hash_combine(seed, index()); 
    return seed; 
}

bool Generic::equal(const Type* other) const {
    return Type::equal(other) ? index() == other->as<Generic>()->index() : false;
}

//------------------------------------------------------------------------------

size_t Opaque::hash() const { 
    size_t seed = Type::hash(); 
    for_all (flag, flags_)
        boost::hash_combine(seed, flag); 
    return seed; 
}

bool Opaque::equal(const Type* other) const {
    return Type::equal(other) ? flags() == other->as<Opaque>()->flags() : false;
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
