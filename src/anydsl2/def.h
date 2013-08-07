#ifndef ANYDSL2_DEF_H
#define ANYDSL2_DEF_H

#include <cstring>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include "anydsl2/enums.h"
#include "anydsl2/node.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/autoptr.h"
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
    Tracker(const Def* def) 
        : def_(0)
    { 
        set(def); 
    }
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

/** 
 * References a user.
 * A \p Def u which uses \p Def d as i^th operand is a \p Use with \p index_ i of \p Def d.
 */
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
    bool operator < (Use) const;

    operator const Def*() const { return def_; }
    const Def* operator -> () const { return def_; }

private:

    size_t index_;
    const Def* def_;
};

//------------------------------------------------------------------------------

/** 
 * References a user which may use the \p Def in question multiple times.
 * For example, a \p Def u may use a \p Def d as i^th \em and is j'th operand.
 * Then a \p MultiUse of d references u with \p indices_ i and j.
 */
class MultiUse {
public:

    MultiUse() {}
    MultiUse(Use use)
        : indices_(1)
        , def_(use.def())
    {
        indices_[0] = use.index();
    }

    size_t index(size_t i) const { return indices_[i]; }
    size_t num_indices() const { return indices_.size(); }
    const std::vector<size_t>& indices() const { return indices_; }
    const Def* def() const { return def_; }
    void append_user(size_t index) { indices_.push_back(index); }

    operator const Def*() const { return def_; }
    const Def* operator -> () const { return def_; }

private:

    std::vector<size_t> indices_;
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
    /**
     * Returns the maximum depth of this \p Def%s depdency tree (induced by the \p ops).
     * \em const dependences are consideres leaves in this tree.
     * Thus, those dependences are not further propagted to determine the depth.
     */
    int non_const_depth() const;
    void dump() const;
    virtual Printer& print(Printer&) const;
    Printer& print_name(Printer&) const;
    const PrimOp* is_non_const_primop() const;

    const Uses& uses() const { return uses_; }
    const Trackers& trackers() const { return trackers_; }
    size_t num_uses() const { return uses_.size(); }
    size_t gid() const { return gid_; }
    std::string unique_name() const;

    /**
     * Copies all use-info into an array.
     * Useful if you want to modfy users while iterating over all users.
     */
    Array<Use> copy_uses() const;
    AutoVector<const Tracker*> tracked_uses() const;
    std::vector<MultiUse> multi_uses() const;
    const Type* type() const { return type_; }
    int order() const;
    bool is_generic() const;
    World& world() const;
    ArrayRef<const Def*> ops() const { return ops_ref<const Def*>(); }
    ArrayRef<const Def*> ops(size_t begin, size_t end) const { return ops().slice(begin, end); }
    const Def* op(size_t i) const { assert(i < ops().size()); return ops()[i]; }
    const Def* op_via_lit(const Def* def) const;
    void replace(const Def*) const;
    /**
     * Returns the vector length.
     * Raises an assertion if type of this is not a \p VectorType.
     */
    size_t length() const;

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
inline bool Use::operator < (Use use) const { return def()->gid() < use.def()->gid() && index() < use.index(); }

//------------------------------------------------------------------------------

class Param : public Def {
private:

    Param(size_t gid, const Type* type, Lambda* lambda, size_t index, const std::string& name);

public:

    Lambda* lambda() const { return lambda_; }
    size_t index() const { return index_; }
    Peeks peek() const;

private:

    mutable Lambda* lambda_;
    const size_t index_;

    friend class World;
    friend class Lambda;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
