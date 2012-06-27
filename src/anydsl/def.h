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
class Def;
class Jump;
class World;

//------------------------------------------------------------------------------

class Use {
public:

    Use(size_t index, const Def* def)
        : index_(index)
        , def_(def)
    {}

    size_t index() const { return index_; }
    const Def* def() const { return def_; }

    bool operator == (const Use& use) const {
        return def() == use.def() && index() == use.index();
    }

    bool operator != (const Use& use) const {
        return def() != use.def() || index() != use.index();
    }

private:

    size_t index_;
    const Def* def_;
};

inline size_t hash_value(const Use& use) { 
    size_t seed = 0;
    boost::hash_combine(seed, use.def());
    boost::hash_combine(seed, use.index());

    return seed;
}

typedef boost::unordered_set<Use> UseSet;

//------------------------------------------------------------------------------

class Def : public AIRNode {
private:

    /// Do not copy-create a \p Def instance.
    Def(const Def&);
    /// Do not copy-assign a \p Def instance.
    Def& operator = (const Def&);

    void registerUse(size_t i, const Def* def) const;
    void unregisterUse(size_t i, const Def* use) const;

protected:

    Def(IndexKind index, const Type* type, size_t numOps)
        : AIRNode(index) 
        , type_(type)
        , numOps_(numOps)
        , ops_(new const Def*[numOps])
    {
        std::memset(ops_, 0, sizeof(const Def*) * numOps);
    }

    virtual ~Def();

    void setOp(size_t i, const Def* def) { def->registerUse(i, this); ops_[i] = def; }
    void delOp(size_t i) const { ops_[i] = 0; }

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

    // TODO iterator class which gets def directly instead of Use

    template<class T>
    struct FilteredUses {
        typedef UseSet::const_iterator const_iterator;

        FilteredUses(const UseSet& uses) : uses(uses) {}

        const_iterator begin() const { return skip(uses.begin()); }
        const_iterator end() const { return uses.end(); }

        size_t size() const {
            size_t n = 0;
            for (const_iterator i = begin(), e = end(); i != e; ++i)
                ++n;

            return n;
        }

        bool empty() const {
            return begin() == end();
        }

    private:

        const_iterator skip(const_iterator i) const {
            while (!(*i).def()->isa<T>() && i != uses.end())
                ++i;
            return i;
        }

        const UseSet& uses;
    };

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

    const UseSet& uses() const { return uses_; }
    const Type* type() const { return type_; }
    size_t numOps() const { return numOps_; }
    World& world() const;
    Ops ops() const { return Ops(*this); }

protected:

    void setType(const Type* type) { type_ = type; }

private:

    const Type* type_;
    size_t numOps_;
    mutable bool flag_;

protected:

    mutable UseSet uses_;
    const Def** ops_;

    friend class World;
};

//------------------------------------------------------------------------------

struct DefHash : std::unary_function<const Def*, size_t> {
    size_t operator () (const Def* v) const { return v->hash(); }
};

struct DefEqual : std::binary_function<const Def*, const Def*, bool> {
    bool operator () (const Def* v1, const Def* v2) const { return v1->equal(v2); }
};

//------------------------------------------------------------------------------

class Param : public Def {
private:

    Param(const Type* type, const Lambda* parent, size_t index);

    size_t index() const { return index_; }

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

public:

    const Lambda* lambda() const;

    virtual void dump(Printer& printer, LambdaPrinterMode mode) const;

private:

    size_t index_;

    friend class Lambda;
};

} // namespace anydsl

#endif
