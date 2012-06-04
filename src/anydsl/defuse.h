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
    Use(AIRNode* parent, Def* def);
    virtual ~Use();

    /// Get the definition \p Def of this \p Use.
    Def* def() { return def_; }
    const Def* def() const { return def_; }
    inline const Type* type() const;

    /// Get embedding ojbect.
    AIRNode* parent() { return parent_; }
    /// Get embedding ojbect.
    const AIRNode* parent() const { return parent_; }


    World& world();

private:

    Def* def_;
    AIRNode* parent_;

    friend class Def;
};

//------------------------------------------------------------------------------
#if 0
    /// typedefs are necessary for std::iterator_traits (needed by FOREACH)
    typedef std::bidirectional_iterator_tag iterator_category;
    typedef Use value_type;
    typedef ptrdiff_t difference_type;
    typedef Use* pointer;
    typedef Use& reference;
#endif


//------------------------------------------------------------------------------

class Def : public AIRNode {
private:

    /// Do not copy-create a \p Def instance.
    Def(const Def&);
    /// Do not copy-assign a \p Def instance.
    Def& operator = (const Def&);

    void registerUse(Use* use);
    void unregisterUse(Use* use);

protected:

    Def(IndexKind index, const Type* type, size_t numOps)
        : AIRNode(index) 
        , type_(type)
        , numOps_(numOps)
        , ops_((Use*) ::operator new(sizeof(Use) * numOps))
    {}

    void setOp(size_t i, Def* def) { new (&ops_[i]) Use(this, def); }

public:

    virtual ~Def() { anydsl_assert(uses_.empty(), "there are still uses pointing to this def"); }

    const UseSet& uses() const { return uses_; }
    const Type* type() const { return type_; }
    size_t numOps() const { return numOps_; }
    World& world() const;

protected:

    void setType(const Type* type) { type_ = type; }

private:

    const Type* type_;
    UseSet uses_;
    size_t numOps_;

protected:

    Use* ops_;

public:

    struct Ops {
        typedef Use* iterator;
        typedef Use* const const_iterator;

        Ops(Def& def)
            : def(def)
        {}

        iterator begin() { return def.ops_; }
        iterator end() { return def.ops_ + size(); }
        const_iterator begin() const { return def.ops_; }
        const_iterator end() const { return def.ops_ + size(); }


        size_t size() const { return def.numOps(); }
        bool empty() const { return def.numOps() == 0; }

        Def& def;
    };

    friend Use::Use(AIRNode*, Def*);
    friend Use::~Use();
};

//------------------------------------------------------------------------------

class Param : public Def {
private:

    Param(Lambda* parent, const Type* type)
        : Def(Index_Param, type, 0)
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

    Value(IndexKind index, const Type* type, size_t numOps)
        : Def(index, type, numOps)
    {}

public:

    virtual bool equal(const Value* other) const;
    virtual size_t hash() const;
};

//------------------------------------------------------------------------------

const Type* Use::type() const { 
    return def_->type(); 
}

} // namespace anydsl

#endif
