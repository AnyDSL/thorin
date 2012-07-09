#ifndef ANYDSL_DEF_H
#define ANYDSL_DEF_H

#include <cstring>
#include <iterator>
#include <string>

#include <boost/cstdint.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>

#include "anydsl/airnode.h"
#include "anydsl/util/ptrascont.h"

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
    void unregisterUse(size_t i, const Def* def) const;

protected:

    Def(int kind, const Type* type, size_t numOps)
        : AIRNode(kind) 
        , type_(type)
        , numOps_(numOps)
        , ops_(new const Def*[numOps])
    {
        std::memset(ops_, 0, sizeof(const Def*) * numOps);
    }

    virtual ~Def();

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

    void setOp(size_t i, const Def* def) { def->registerUse(i, this); ops_[i] = def; }
    void delOp(size_t i) const { ops_[i] = 0; }
    void setType(const Type* type) { type_ = type; }

public:

    typedef PtrAsCont<const Def> Ops;
    typedef PtrAsCont<const Def> Args;

    template<class T>
    class FilteredUses {
    public:

        class const_iterator {
        public:
            typedef UseSet::const_iterator::iterator_category iterator_category;
            typedef const T* value_type;
            typedef ptrdiff_t difference_type;
            typedef const T** pointer;
            typedef const T*& reference;

            const_iterator(const const_iterator& i)
                : base_(i.base_)
                , end_(i.end_)
            {
                assert(base_ == end_ || base_->def()->isa<T>());
            }
            const_iterator(UseSet::const_iterator base, UseSet::const_iterator end)
                : base_(base)
                , end_(end)
            {
                skip();
            }

            const_iterator& operator ++ () { ++base_; skip(); return *this; }
            const_iterator operator ++ (int) { const_iterator i(*this); ++(*this); return i; }

            bool operator == (const const_iterator& i) { return base_ == i.base_; }
            bool operator != (const const_iterator& i) { return base_ != i.base_; }

            const T* operator *  () { return base_->def()->as<T>(); }
            const T* operator -> () { return base_->def()->as<T>(); }

        private:

            UseSet::const_iterator base_;
            UseSet::const_iterator end_;

            void skip() {
                while (base_ != end_ && !base_->def()->isa<T>())
                    ++base_;
            }
        };

        FilteredUses(const UseSet& uses) : uses_(uses) {}

        const_iterator begin() const { return const_iterator(uses_.begin(), uses_.end()); }
        const_iterator end() const   { return const_iterator(uses_.end(), uses_.end()); }

        /**
         * Be carfeull! This has O(n) worst case execution behavior!
         * Anyway, a node usually has less then 3 uses - so in most cases you can forget about this cost.
         */
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

        const UseSet& uses_;
    };

    const UseSet& uses() const { return uses_; }
    const Type* type() const { return type_; }
    size_t numOps() const { return numOps_; }
    World& world() const;
    Ops ops() const { return Ops(ops_, numOps_); }
    const Def* op(size_t i) const { anydsl_assert(i < numOps_, "index out of bounds"); return ops_[i]; }

private:

    const Type* type_;
    size_t numOps_;
    mutable UseSet uses_;
    mutable bool flag_;

protected:

    const Def** ops_;

    friend class World;
    friend class DefHash;
    friend class DefEqual;
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

    virtual bool equal(const Def* other) const;
    virtual size_t hash() const;

public:

    const Lambda* lambda() const;
    size_t index() const { return index_; }
    std::vector<const Def*> phiOps() const;

private:

    virtual void vdump(Printer& printer) const;

    size_t index_;

    friend class Lambda;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
