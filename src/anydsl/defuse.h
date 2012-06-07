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

class Lambda;
class Type;
class Sigma;
class World;
class Use;
typedef boost::unordered_set<Use*> UseSet;
class Def;
class Jump;
class World;

/**
 * Use encapsulates a use of an SSA value, i.e., a \p Def.
 *
 * This class is supposed to be embedded in other \p AIRNode%s.
 * \p Use already has enough encapsulation magic. 
 * No need to hammer further getters/setters around a Use aggregate within a class.
 * Just make it a public class member.
 */
class Use : public AIRNode {
private:

    /// Do not copy-create a \p Use instance.
    Use(const Use&);
    /// Do not copy-assign a \p Use instance.
    Use& operator = (const Use&);

public:

    /** 
     * @brief Construct a \p Use of the specified \p Def.
     * 
     * @param parent The class where \p Use is embedded in.
     * @param def 
     */
    Use(AIRNode* parent, const Def* def);
    virtual ~Use();

    /// Get the definition \p Def of this \p Use.
    const Def* def() const { return def_; }
    inline const Type* type() const;

    /// Get embedding ojbect.
    AIRNode* parent() { return parent_; }
    /// Get embedding ojbect.
    const AIRNode* parent() const { return parent_; }


    World& world();

private:

    const Def* def_;
    AIRNode* parent_;

    friend class Def;
};

//------------------------------------------------------------------------------

class Def : public AIRNode {
private:

    /// Do not copy-create a \p Def instance.
    Def(const Def&);
    /// Do not copy-assign a \p Def instance.
    Def& operator = (const Def&);

    void registerUse(Use* use) const;
    void unregisterUse(Use* use) const;

protected:

    Def(IndexKind index, const Type* type, size_t numOps)
        : AIRNode(index) 
        , type_(type)
        , numOps_(numOps)
        , ops_((Use*) ::operator new(sizeof(Use) * numOps))
    {}

    virtual ~Def();

    void setOp(size_t i, const Def* def) { new (&ops_[i]) Use(this, def); }

public:

    template<class AS>
    struct def_iter {
    public:

        /// typedefs are necessary for std::iterator_traits (needed by FOREACH)
        typedef std::bidirectional_iterator_tag iterator_category;
        typedef const Def* value_type;
        typedef ptrdiff_t difference_type;
        typedef const Def** pointer;
        typedef const Def*& reference;

        def_iter() {}
        explicit def_iter(const Use* base) : base_(base) {}
        template<class U> def_iter(def_iter<U> const& i) : base_(i.base_) {}

        const AS* operator * () const { return base_->def()->as<AS>(); }
        const AS* operator ->() const { return base_->def()->as<AS>(); }

        template<class U> bool operator == (const def_iter<U>& i) const { return base_ == i.base_; }
        template<class U> bool operator != (const def_iter<U>& i) const { return base_ != i.base_; }

        def_iter operator + (size_t i) { return def_iter(base_ + i); }
        def_iter operator - (size_t i) { return def_iter(base_ - i); }

        def_iter& operator ++() { ++base_; return *this; }
        def_iter& operator --() { --base_; return *this; }
        def_iter operator ++(int) { def_iter i(base_); ++base_; return i; }
        def_iter operator --(int) { def_iter i(base_); --base_; return i; }

    private:

        const Use* base_;
    };

    struct Ops {
        typedef const Use* const_iterator;
        typedef std::reverse_iterator<const Use*> const_reverse_iterator;

        Ops(const Def& def) : def(def) {}

        const_iterator begin() const { return def.ops_; }
        const_iterator end() const { return def.ops_ + size(); }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(def.ops_ + size()); }
        const_reverse_iterator rend() const { return const_reverse_iterator(def.ops_); }

        size_t size() const { return def.numOps(); }
        bool empty() const { return def.numOps() == 0; }

        const Use& operator [] (size_t i) const {
            anydsl_assert(i < size(), "index out of bounds");
            return def.ops_[i];
        }

        const Use& front() { return def.ops_[0]; }
        const Use& back() { return def.ops_[size()-1]; }

    private:

        const Def& def;
    };

    template<class AS>
    struct AsOps {
        typedef def_iter<AS> const_iterator;
        typedef std::reverse_iterator<def_iter<AS> > const_reverse_iterator;

        AsOps(const Def& def) : def(def) {}

        const_iterator begin() const { return const_iterator(def.ops_); }
        const_iterator end() const { return const_iterator(def.ops_ + size()); }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
        const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

        size_t size() const { return def.numOps(); }
        bool empty() const { return def.numOps() == 0; }

        const Def* operator [] (size_t i) const {
            anydsl_assert(i < size(), "index out of bounds");
            return def.ops_[i].def();
        }

        const Def* front() { return def.ops_[0].def(); }
        const Def* back() { return def.ops_[size()-1].def(); }

    private:

        const Def& def;
    };

    typedef AsOps<Def> DefOps;

    const UseSet& uses() const { return uses_; }
    const Type* type() const { return type_; }
    size_t numOps() const { return numOps_; }
    World& world() const;

    Ops ops() const { return Ops(*this); }
    DefOps defops() const { return DefOps(*this); }

protected:

    void setType(const Type* type) { type_ = type; }

private:

    const Type* type_;
    mutable UseSet uses_;
    size_t numOps_;

protected:

    Use* ops_;

public:

    friend Use::Use(AIRNode*, const Def*);
    friend Use::~Use();
};

//------------------------------------------------------------------------------

class Params : public Def {
private:

    Params(Lambda* parent, const Sigma* sigma);

    const Sigma* sigma() const;

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

const Type* Use::type() const { 
    return def_->type(); 
}

} // namespace anydsl

#endif
