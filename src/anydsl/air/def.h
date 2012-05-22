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
private:

    ValueNumber& operator = (const ValueNumber& vn);

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
    explicit ValueNumber(IndexKind index)
        : index(index)
        , op1(0)
        , op2(0)
        , op3(0)
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
    ValueNumber(const ValueNumber& vn) {
        std::memcpy(this, &vn, sizeof(ValueNumber));
        if (hasMore(index)) {
            more = new uintptr_t[size];
            std::memcpy(more, vn.more, sizeof(uintptr_t) * size);
        }
    }
    ~ValueNumber() {
        if (hasMore(index))
            delete[] more;
    }

    /**
     * Creates a ValueNumber where the number of built-in fields do not suffice.
     * Memory allocation and deallocation is handled by this class. 
     * However, the caller is responsible to fill the allocated fields
     * (pointed to by \p more) with correct data.
     */
    static ValueNumber createMore(IndexKind index, size_t size) {
        ValueNumber res(index);
        res.size = size;
        res.more = new uintptr_t[size];
        return res;
    }

    bool operator == (const ValueNumber& vn) const;

    static bool hasMore(IndexKind kind) {
        switch (kind) {
            case Index_Tuple:
            case Index_Pi:
            case Index_Sigma: 
                return true;
            default:
                return false;
        }
    }
};

size_t hash_value(const ValueNumber& vn);

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_DEF_H
