#ifndef ANYDSL_DEF_H
#define ANYDSL_DEF_H

#include <cstring>
#include <string>

#include <boost/cstdint.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>

#include "anydsl/airnode.h"

namespace anydsl {

//------------------------------------------------------------------------------

class Def;
class Lambda;
class Type;
class Sigma;
class World;
typedef boost::unordered_multiset<const Def*> UseSet;
class Def;
class Jump;
class World;

//------------------------------------------------------------------------------

class Def : public AIRNode {
private:

    /// Do not copy-create a \p Def instance.
    Def(const Def&);
    /// Do not copy-assign a \p Def instance.
    Def& operator = (const Def&);

    void registerUse(const Def* use) const;
    void unregisterUse(const Def* use) const;

protected:

    Def(IndexKind index, const Type* type, size_t numOps)
        : AIRNode(index) 
        , type_(type)
        , numOps_(numOps)
        , ops_(new const Def*[numOps])
    {}

    virtual ~Def();

    void setOp(size_t i, const Def* def) { def->registerUse(this); ops_[i] = def; }

public:

    struct Ops {
        typedef const Def** const_iterator;
        typedef std::reverse_iterator<const Def**> const_reverse_iterator;

        Ops(const Def& def) : def(def) {}

        const_iterator begin() const { return def.ops_; }
        const_iterator end() const { return def.ops_ + size(); }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(def.ops_ + size()); }
        const_reverse_iterator rend() const { return const_reverse_iterator(def.ops_); }

        size_t size() const { return def.numOps(); }
        bool empty() const { return def.numOps() == 0; }

        const Def* const& operator [] (size_t i) const {
            anydsl_assert(i < size(), "index out of bounds");
            return def.ops_[i];
        }

        const Def* const& front() { return def.ops_[0]; }
        const Def* const& back() { return def.ops_[size()-1]; }

    private:

        const Def& def;
    };

    const UseSet& uses() const { return uses_; }
    const Type* type() const { return type_; }
    size_t numOps() const { return numOps_; }
    World& world() const;
    Ops ops() const { return Ops(*this); }

protected:

    void setType(const Type* type) { type_ = type; }

private:

    const Type* type_;
    mutable UseSet uses_;
    size_t numOps_;

protected:

    const Def** ops_;
};

//------------------------------------------------------------------------------

class Param : public Def {
private:

    Param(Lambda* parent);

public:

    const Lambda* parent() const { return parent_; }

private:

    Lambda* parent_;

    friend class Lambda;
};

//------------------------------------------------------------------------------

class Value : public Def {
protected:

    Value(IndexKind index, const Type* type, size_t numOps)
        : Def(index, type, numOps)
    {}

public:

    virtual bool equal(const Value* other) const;
    virtual size_t hash() const;
};

struct ValueHash : std::unary_function<const Value*, size_t> {
    size_t operator () (const Value* v) const { return v->hash(); }
};

struct ValueEqual : std::binary_function<const Value*, const Value*, bool> {
    bool operator () (const Value* v1, const Value* v2) const { return v1->equal(v2); }
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
