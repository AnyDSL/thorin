#ifndef ANYDSL2_DEF_H
#define ANYDSL2_DEF_H

#include <cstring>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

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
class Tracker;
class Type;
class World;

//------------------------------------------------------------------------------

class Peek {
public:

    Peek() {}
    Peek(const Def* def, Lambda* from)
        : def_(def)
        , from_(from)
    {}

    const Def* def() const { return def_; }
    Lambda* from() const { return from_; }

private:

    const Def* def_;
    Lambda* from_;
};

typedef Array<Peek> Peeks;

//------------------------------------------------------------------------------

/// References a \p Def but updates its reference after a \p Def::replace with the replaced \p Def.
class Tracker {
private:

    /// Do not copy-construct a \p Tracker.
    Tracker(const Tracker&);

public:

    Tracker()
        : def_(0)
    {}
    Tracker(const Def* def) { set(def); }
    ~Tracker() { release(); }

    const Def* operator = (const Def* def) { set(def); return def_; }
    void set(const Def*);
    void release();
    const Def* def() const { return def_; }
    operator const Def*() const { return def_; }
    const Def* operator -> () const { return def_; }

private:

    const Def* def_;
};

//------------------------------------------------------------------------------

/// References a user, i.e., a \p Def using the referenced \p Def in question as \p index_'s operand.
class Use {
public:

    Use() {}
    Use(size_t index, const Def* def)
        : index_(index)
        , def_(def)
    {}

    size_t index() const { return index_; }
    const Def* def() const { return def_; }

    bool operator == (Use use) const { return def() == use.def() && index() == use.index(); }
    bool operator != (Use use) const { return def() != use.def() || index() != use.index(); }

    operator const Def*() const { return def_; }
    const Def* operator -> () const { return def_; }

private:

    size_t index_;
    const Def* def_;
};

//------------------------------------------------------------------------------

typedef std::vector<Use> Uses;
typedef std::vector<Tracker*> Trackers;

//------------------------------------------------------------------------------

/**
 * The base class for all three kinds of Definitions in AnyDSL.
 * These are:
 * - \p PrimOp%s
 * - \p Param%s and
 * - \p Lambda%s.
 */
class Def : public Node {
private:

    /// Do not copy-assign a \p Def instance.
    Def& operator = (const Def&);

protected:

    Def(size_t gid, int kind, const Type* type, bool is_const, const std::string& name)
        : Node(kind, name)
        , type_(type)
        , gid_(gid)
        , is_const_(is_const)
    {
        uses_.reserve(4);
    }
    Def(size_t gid, int kind, size_t size, const Type* type, bool is_const, const std::string& name)
        : Node(kind, size, name)
        , type_(type)
        , gid_(gid)
        , is_const_(is_const)
    {
        uses_.reserve(4);
    }

    void set_type(const Type* type) { type_ = type; }
    void unregister_use(size_t i) const;

public:

    virtual ~Def() {}
    void set_op(size_t i, const Def* def);
    void unset_op(size_t i);
    void unset_ops();
    Lambda* as_lambda() const;
    Lambda* isa_lambda() const;
    bool is_const() const { return is_const_; }
    void dump() const;
    virtual Printer& print(Printer&) const;
    Printer& print_name(Printer&) const;
    const PrimOp* is_non_const_primop() const;

    const Uses& uses() const { return uses_; }
    const Trackers& trackers() const { return trackers_; }
    size_t num_uses() const { return uses_.size(); }
    size_t gid() const { return gid_; }
    virtual const Def* representative() const { return this; }
    std::string unique_name() const;

    /**
     * Copies all use-info into an array.
     * Useful if you want to modfy users while iterating over all users.
     */
    Array<Use> copy_uses() const;
    const Type* type() const { return type_; }
    int order() const;
    bool is_generic() const;
    World& world() const;
    ArrayRef<const Def*> ops() const { return ops_ref<const Def*>(); }
    ArrayRef<const Def*> ops(size_t begin, size_t end) const { return ops().slice(begin, end); }
    const Def* op(size_t i) const { assert(i < ops().size()); return ops()[i]; }
    const Def* op_via_lit(const Def* def) const;
    void replace(const Def*) const;

    // check for special literals

    bool is_primlit(int val) const;
    bool is_zero() const { return is_primlit(0); }
    bool is_minus_zero() const;
    bool is_one() const { return is_primlit(1); }
    bool is_allset() const { return is_primlit(-1); }

    bool is_div()         const { return anydsl2::is_div  (kind()); }
    bool is_rem()         const { return anydsl2::is_rem  (kind()); }
    bool is_bitop()       const { return anydsl2::is_bitop(kind()); }
    bool is_shift()       const { return anydsl2::is_shift(kind()); }
    bool is_not()         const { return kind() == ArithOp_xor && op(0)->is_allset(); }
    bool is_minus()       const { return (kind() == ArithOp_sub || kind() == ArithOp_fsub) && op(0)->is_minus_zero(); }
    bool is_div_or_rem()  const { return anydsl2::is_div_or_rem(kind()); }
    bool is_commutative() const { return anydsl2::is_commutative(kind()); }
    bool is_associative() const { return anydsl2::is_associative(kind()); }

    // implementation in literal.h
    template<class T> inline T primlit_value() const;

private:

    const Type* type_;
    mutable Uses uses_;
    mutable Trackers trackers_;
    const size_t gid_;

protected:

    bool is_const_;

    friend class Tracker;
    friend class PrimOp;
    friend class World;
};

std::ostream& operator << (std::ostream& o, const Def* def);

//------------------------------------------------------------------------------

class Param : public Def {
private:

    Param(size_t gid, const Type* type, Lambda* lambda, size_t index, const std::string& name);

public:

    Lambda* lambda() const { return lambda_; }
    size_t index() const { return index_; }
    Peeks peek() const;
    virtual const Def* representative() const;

private:

    mutable Lambda* lambda_;
    const size_t index_;
    mutable const Def* representative_;

    friend class World;
    friend class Lambda;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
