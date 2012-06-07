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

    struct Ops {
        typedef Use* iterator;
        typedef Use* const_iterator;
        typedef std::reverse_iterator<Use*> reverse_iterator;
        typedef std::reverse_iterator<Use*> const_reverse_iterator;

        Ops(Def& def) : def(def) {}

        iterator begin() { return def.ops_; }
        iterator end() { return def.ops_ + size(); }
        const_iterator begin() const { return def.ops_; }
        const_iterator end() const { return def.ops_ + size(); }

        reverse_iterator rbegin() { return reverse_iterator(def.ops_ + size()); }
        reverse_iterator rend() { return reverse_iterator(def.ops_); }
        const_reverse_iterator rbegin() const { return reverse_iterator(def.ops_ + size()); }
        const_reverse_iterator rend() const { return reverse_iterator(def.ops_); }

        size_t size() const { return def.numOps(); }
        bool empty() const { return def.numOps() == 0; }

        Use* operator [] (size_t i) {
            anydsl_assert(i < size(), "index out of bounds");
            return def.ops_ + i;
        }
        const Use* operator [] (size_t i) const {
            anydsl_assert(i < size(), "index out of bounds");
            return def.ops_ + i;
        }

        Use& front() { return def.ops_[0]; }
        Use& back() { return def.ops_[size()-1]; }

        Def& def;
    };

    const UseSet& uses() const { return uses_; }
    const Type* type() const { return type_; }
    size_t numOps() const { return numOps_; }
    World& world() const;

    Ops ops() { return Ops(*this); }
    Ops ops() const { return Ops(*ccast<Def>(this)); }

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
