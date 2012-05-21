#ifndef ANYDSL_AIR_DEF_H
#define ANYDSL_AIR_DEF_H

#include <string>

#include <boost/cstdint.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>

#include "anydsl/air/airnode.h"

namespace anydsl {

//------------------------------------------------------------------------------

class Lambda;
class Type;
class World;
class Use;
typedef boost::unordered_set<Use*> Uses;
typedef std::vector<Use> Ops;

//------------------------------------------------------------------------------

class Def : public AIRNode {
protected:

    Def(IndexKind index, const Type* type)
        : AIRNode(index) 
        , type_(type)
    {}

public:

    //virtual ~Def() { anydsl_assert(uses_.empty(), "there are still uses pointing to this def"); }
    virtual ~Def() { /* TODO assertion above is useful */ }

    /**
     * Manually adds given \p Use object to the list of uses of this \p Def.
     * use->def() must already point to \p this .
     */
    void registerUse(Use* use);

    /**
     * Manually removes given \p Use object from the list of uses of this \p Def.
     * use->def() must point to \p this , but should be unset right after the call to this function
     */
    void unregisterUse(Use* use);

    const Uses& uses() const { return uses_; }
    const Type* type() const { return type_; }
    World& world() const;

protected:

    void setType(const Type* type) { type_ = type; }

private:

    const Type* type_;
    Uses uses_;
};

//------------------------------------------------------------------------------

class Param : public Def {
private:

    Param(Lambda* parent, const Type* type)
        : Def(Index_Param, type)
        , parent_(parent)
    {}

    const Lambda* parent() const { return parent_; }

private:

    Lambda* parent_;

    friend class Lambda;
};

//------------------------------------------------------------------------------

class Value : public Def {
protected:

    Value(IndexKind index, const Type* type)
        : Def(index, type)
    {}
};

struct ValueNumber {
public:
    IndexKind index;
    uintptr_t op1;


    union {
        uintptr_t op2;
        size_t size;
    };

    union {
        uintptr_t op3;
        uintptr_t* more;
    };

    ValueNumber() {}
    ValueNumber(IndexKind index)
        : index(index)
        , op1(0)
        , op2(0)
    {}
    ValueNumber(IndexKind index, uintptr_t p)
        : index(index)
        , op1(p)
        , op2(0)
        , op3(0)
    {}
    ValueNumber(IndexKind index, uintptr_t p1, uintptr_t p2) 
        : index(index)
        , op1(p1)
        , op2(p2)
        , op3(0)
    {}
    ValueNumber(IndexKind index, uintptr_t p1, uintptr_t p2, uintptr_t p3)
        : index(index)
        , op1(p1)
        , op2(p2)
        , op3(p3)
    {}
    ValueNumber(size_t size, uintptr_t* more)
        : size(size)
        , more(more)
    {}
    ~ValueNumber() {
        if (index == Index_Tuple)
            delete more;
    }
    /// Mimics a C++11 move constructor by const_cast.
    ValueNumber(const ValueNumber& vn) {
        if (index == Index_Tuple) {
            index = vn.index;
            size = vn.size;
            more = vn.more;
            const_cast<ValueNumber&>(vn).more = 0;
        } else
            std::memcpy(this, &vn, sizeof(ValueNumber));
    }
    ValueNumber& operator = (ValueNumber& vn) {
        if (index == Index_Tuple) {
            index = vn.index;
            size = vn.size;
            more = vn.more;
            vn.more = 0;
        } else
            std::memcpy(this, &vn, sizeof(ValueNumber));

        return *this;
    }


    bool operator == (const ValueNumber& vn) const {
        return index == vn.index && op1 == vn.op1 && op2 == vn.op2 && op3 == vn.op3;
    }
};

inline size_t hash_value(const ValueNumber& vn) {
    size_t seed = 0;
    boost::hash_combine(seed, vn.index);
    boost::hash_combine(seed, vn.op1);
    boost::hash_combine(seed, vn.op2);
    boost::hash_combine(seed, vn.op3);

    return seed;
}

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_DEF_H
