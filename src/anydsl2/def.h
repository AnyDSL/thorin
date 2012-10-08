#ifndef ANYDSL_DEF_H
#define ANYDSL_DEF_H

#include <cstring>
#include <iterator>
#include <string>

#include <boost/cstdint.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>

#include "anydsl2/enums.h"
#include "anydsl2/node.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/cast.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

class Def;
class Lambda;
class Printer;
class PrimOp;
class Sigma;
class Type;
class World;

//------------------------------------------------------------------------------

class PhiOp {
public:

    PhiOp() {}
    PhiOp(const Def* def, Lambda* from)
        : def_(def)
        , from_(from)
    {}

    const Def* def() const { return def_; }
    Lambda* from() const { return from_; }

private:

    const Def* def_;
    Lambda* from_;
};

typedef Array<PhiOp> PhiOps;

inline const Def* const& op_as_def(const Node* const* ptr) { 
    assert( !(*ptr) || (*ptr)->as<Def>());
    return *((const Def* const*) ptr); 
}

typedef ArrayRef<const Node*, const Def*, op_as_def> Defs;

//------------------------------------------------------------------------------

class Use {
public:

    Use() {}
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

typedef boost::unordered_set<Use> Uses;

//------------------------------------------------------------------------------

class Def : public Node {
private:

    /// Do not copy-assign a \p Def instance.
    Def& operator = (const Def&);

protected:

    /// This variant leaves internal \p ops_ \p Array allocateble via ops_.alloc(size).
    Def(int kind, const Type* type);
    Def(int kind, size_t size, const Type* type);
    Def(const Def&);
    virtual ~Def();

    void set_type(const Type* type) { type_ = type; }
    void unregister_use(size_t i) const;

public:

    void set_op(size_t i, const Def* def);
    void unset_op(size_t i);

    Lambda* as_lambda() const;
    Lambda* isa_lambda() const;

    bool is_const() const;

    void dump() const;
    void dump(bool fancy) const;

    virtual void vdump(Printer &printer) const = 0;

    const Uses& uses() const { return uses_; }

    /**
     * Copies all use-info into an array.
     * Useful if you want to modfy users while iterating over all users.
     */
    Array<Use> copy_uses() const;
    const Type* type() const { return type_; }
    World& world() const;

    Defs ops() const { return ops_ref<const Node*, const Def*, op_as_def>(); }
    Defs ops(size_t begin, size_t end) const { return ops().slice(begin, end); }
    const Def* op(size_t i) const { return ops()[i]; }

    /// Updates operand \p i to point to \p def instead.
    void update(size_t i, const Def* def);
    void update(ArrayRef<const Def*> defs);

    /*
     * check for special literals
     */

    bool is_primlit(int val) const;
    bool is_zero() const { return is_primlit(0); }
    bool is_one() const { return is_primlit(1); }
    bool is_allset() const { return is_primlit(-1); }

private:

    const Type* type_;
    mutable Uses uses_;
};

//------------------------------------------------------------------------------

class Param : public Def {
private:

    Param(const Type* type, Lambda* parent, size_t index);
    virtual Param* clone() const { ANYDSL_UNREACHABLE; }

    virtual bool equal(const Node* other) const;
    virtual size_t hash() const;

public:

    Lambda* lambda() const { return lambda_; }
    size_t index() const { return index_; }
    PhiOps phi() const;
    virtual void vdump(Printer& printer) const;

private:

    mutable Lambda* lambda_;
    const size_t index_;

    friend class World;
    friend class Lambda;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
