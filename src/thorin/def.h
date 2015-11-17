#ifndef THORIN_DEF_H
#define THORIN_DEF_H

#include <set>
#include <string>
#include <vector>

#include "thorin/enums.h"
#include "thorin/type.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/cast.h"
#include "thorin/util/hash.h"
#include "thorin/util/location.h"
#include "thorin/util/stream.h"

namespace thorin {

//------------------------------------------------------------------------------

class DefNode;
class Lambda;
class PrimOp;
class Use;
class World;

//------------------------------------------------------------------------------

/**
 * This class acts as a proxy for \p DefNode pointers.
 * This proxy hides that a \p DefNode may have been replaced by another one.
 * It automatically forwards to the replaced node.
 * If in doubt use a \p Def instead of \p DefNode*.
 * You almost never have to use a \p DefNode* directly.
 */
class Def {
public:
    Def()
        : node_(nullptr)
    {}
    Def(const DefNode* node)
        : node_(node)
    {}

    bool empty() const { return node_ == nullptr; }
    const DefNode* node() const { return node_; }
    const DefNode* deref() const;
    const DefNode* operator *() const { return deref(); }
    bool operator == (const DefNode* other) const { return this->deref() == other; }
    operator const DefNode*() const { return deref(); }
    const DefNode* operator -> () const { return deref(); }

private:
    mutable const DefNode* node_;
};

/**
 * References a user.
 * A \p Def u which uses \p Def d as i^th operand is a \p Use with \p index_ i of \p Def d.
 */
class Use {
public:
    Use() {}
    Use(size_t index, Def def)
        : index_(index)
        , def_(def)
    {}

    size_t index() const { return index_; }
    const Def& def() const { return def_; }
    operator Def() const { return def_; }
    operator const DefNode*() const { return def_; }
    const Def& operator -> () const { return def_; }

private:
    size_t index_;
    Def def_;
};

struct UseLT {
    inline bool operator () (Use use1, Use use2) const;
};

std::ostream& operator << (std::ostream&, Def);
std::ostream& operator << (std::ostream&, Use);

//------------------------------------------------------------------------------

template<class To>
using DefMap  = HashMap<const DefNode*, To, GIDHash<const DefNode*>, GIDEq<const DefNode*>>;
using DefSet  = HashSet<const DefNode*, GIDHash<const DefNode*>, GIDEq<const DefNode*>>;
using Def2Def = DefMap<const DefNode*>;

//------------------------------------------------------------------------------

/**
 * The base class for all three kinds of Definitions in AnyDSL.
 * These are:
 * - \p PrimOp%s
 * - \p Param%s and
 * - \p Lambda%s.
 */
class DefNode : public HasLocation, public MagicCast<DefNode>, public Streamable {
private:
    DefNode& operator = (const DefNode&); ///< Do not copy-assign a \p DefNode instance.
    DefNode(const DefNode&);              ///< Do not copy-construct a \p DefNode.

protected:
    DefNode(size_t gid, NodeKind kind, Type type, size_t size, const Location& loc, const std::string& name)
        : HasLocation(loc)
        , kind_(kind)
        , ops_(size)
        , type_(type)
        , representative_(this)
        , gid_(gid)
        , name(name)
    {}
    virtual ~DefNode() {}

    void clear_type() { type_.clear(); }
    void set_type(Type type) { type_ = type; }
    void unregister_use(size_t i) const;
    void unregister_uses() const;
    void resize(size_t n) { ops_.resize(n, nullptr); }
    void unlink_representative() const;

public:
    NodeKind kind() const { return kind_; }
    bool is_corenode() const { return ::thorin::is_corenode(kind()); }
    size_t size() const { return ops_.size(); }
    bool empty() const { return ops_.empty(); }
    void set_op(size_t i, Def def);
    void unset_op(size_t i);
    void unset_ops();
    Def is_mem() const { return type().isa<MemType>() ? this : nullptr; }
    Lambda* as_lambda() const;
    Lambda* isa_lambda() const;
    bool is_const() const;
    void dump() const;
    std::vector<Use> uses() const;
    bool is_proxy() const { return representative_ != this; }
    /// WARNING: slow!
    size_t num_uses() const { return uses().size(); }
    size_t gid() const { return gid_; }
    std::string unique_name() const;
    Type type() const { return type_; }
    int order() const;
    World& world() const;
    ArrayRef<Def> ops() const { return ops_; }
    Def op(size_t i) const { assert(i < ops().size() && "index out of bounds"); return ops_[i]; }
    Def op(Def def) const;
    void replace(Def) const;
    size_t length() const; ///< Returns the vector length. Raises an assertion if type of this is not a \p VectorType.
    bool is_primlit(int val) const;
    bool is_zero() const { return is_primlit(0); }
    bool is_minus_zero() const;
    bool is_one() const { return is_primlit(1); }
    bool is_allset() const { return is_primlit(-1); }
    bool is_bitop()       const { return thorin::is_bitop(kind()); }
    bool is_shift()       const { return thorin::is_shift(kind()); }
    bool is_not()         const { return kind() == Node_xor && op(0)->is_allset(); }
    bool is_minus()       const { return kind() == Node_sub && op(0)->is_minus_zero(); }
    bool is_div_or_rem()  const { return thorin::is_div_or_rem(kind()); }
    bool is_commutative() const { return thorin::is_commutative(kind()); }
    bool is_associative() const { return thorin::is_associative(kind()); }
    template<class T> inline T primlit_value() const; // implementation in literal.h
    virtual Def rebuild() const { return this; }
    virtual std::ostream& stream(std::ostream&) const;

private:
    const NodeKind kind_;
    std::vector<Def> ops_;
    Type type_;
    mutable std::set<Use, UseLT> uses_;
    mutable const DefNode* representative_;
    mutable DefSet representatives_of_;
    const size_t gid_;
    mutable uint32_t candidate_ = 0;

public:
    mutable std::string name; ///< Just do what ever you want with this field.

    friend class Def;
    friend class PrimOp;
    friend class Scope;
    friend class World;
    friend class Cleaner;
};

namespace detail {
    inline std::ostream& stream(std::ostream& out, Def def) { return def->stream(out); }
}

//------------------------------------------------------------------------------

bool UseLT::operator () (Use use1, Use use2) const { // <- note that we switch the order here on purpose
    auto gid1 = use1.def().node()->gid();
    auto gid2 = use2.def().node()->gid();
    return (gid1 < gid2 || (gid1 == gid2 && use1.index() < use2.index()));
}

//------------------------------------------------------------------------------

template<>
struct Hash<ArrayRef<Def>> {
    uint64_t operator () (ArrayRef<Def> defs) const {
        uint64_t seed = hash_begin(defs.size());
        for (auto def : defs)
            seed = hash_combine(seed, def.empty() ? uint64_t(-1) : def->gid());
        return seed;
    }
};

template<>
struct Hash<Array<Def>> {
    uint64_t operator () (const Array<Def> defs) const { return Hash<ArrayRef<Def>>()(defs); }
};

//------------------------------------------------------------------------------

}

#endif
