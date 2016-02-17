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

class Def;
class Lambda;
class PrimOp;
class Use;
class World;

//------------------------------------------------------------------------------

/**
 * This class acts as a proxy for \p Def pointers.
 * This proxy hides that a \p Def may have been replaced by another one.
 * It automatically forwards to the replaced node.
 * If in doubt use a \p Def instead of \p Def*.
 * You almost never have to use a \p Def* directly.
 */
class Tracker {
public:
    Tracker()
        : def_(nullptr)
    {}
    Tracker(const Def* node)
        : def_(node)
    {}

    bool empty() const { return def_ == nullptr; }
    const Def* operator *() const { return def_; }
    bool operator == (const Def* other) const { return **this == other; }
    operator const Def*() const { return **this; }
    const Def* operator -> () const { return **this; }

private:
    mutable const Def* def_;
};

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
    operator const Def*() const { return def_; }
    const Def* operator -> () const { return def_; }
    bool operator == (Use other) { return this->def() == other.def() && this->index() == other.index(); }

private:
    size_t index_;
    const Def* def_;
};

struct UseHash {
    inline uint64_t operator () (Use use) const;
};

std::ostream& operator << (std::ostream&, const Def*);
std::ostream& operator << (std::ostream&, Use);

//------------------------------------------------------------------------------

template<class To>
using DefMap  = HashMap<const Def*, To, GIDHash<const Def*>, GIDEq<const Def*>>;
using DefSet  = HashSet<const Def*, GIDHash<const Def*>, GIDEq<const Def*>>;
using Def2Def = DefMap<const Def*>;

//------------------------------------------------------------------------------

/**
 * The base class for all three kinds of Definitions in AnyDSL.
 * These are:
 * - \p PrimOp%s
 * - \p Param%s and
 * - \p Lambda%s.
 */
class Def : public HasLocation, public MagicCast<Def>, public Streamable {
private:
    Def& operator = (const Def&); ///< Do not copy-assign a \p Def instance.
    Def(const Def&);              ///< Do not copy-construct a \p Def.

protected:
    Def(size_t gid, NodeKind kind, Type type, size_t size, const Location& loc, const std::string& name)
        : HasLocation(loc)
        , kind_(kind)
        , ops_(size)
        , type_(type)
        , gid_(gid)
        , name(name)
    {}
    virtual ~Def() {}

    void clear_type() { type_.clear(); }
    void set_type(Type type) { type_ = type; }
    void unregister_use(size_t i) const;
    void unregister_uses() const;
    void resize(size_t n) { ops_.resize(n, nullptr); }

public:
    NodeKind kind() const { return kind_; }
    bool is_corenode() const { return ::thorin::is_corenode(kind()); }
    size_t size() const { return ops_.size(); }
    bool empty() const { return ops_.empty(); }
    void set_op(size_t i, const Def* def);
    void unset_op(size_t i);
    void unset_ops();
    const Def* is_mem() const { return type().isa<MemType>() ? this : nullptr; }
    Lambda* as_lambda() const;
    Lambda* isa_lambda() const;
    bool is_const() const;
    void dump() const;
    const HashSet<Use, UseHash>& uses() const { return uses_; }
    size_t num_uses() const { return uses().size(); }
    size_t gid() const { return gid_; }
    std::string unique_name() const;
    Type type() const { return type_; }
    int order() const;
    World& world() const;
    ArrayRef<const Def*> ops() const { return ops_; }
    const Def* op(size_t i) const { assert(i < ops().size() && "index out of bounds"); return ops_[i]; }
    template<class T> const Def* op(const T* def) const;
    void replace(const Def*) const;
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
    virtual const Def* rebuild() const { return this; }
    virtual std::ostream& stream(std::ostream&) const;

private:
    const NodeKind kind_;
    std::vector<const Def*> ops_;
    Type type_;
    mutable HashSet<Use, UseHash> uses_;
    mutable HashSet<Tracker*> trackers_;
    const size_t gid_;
    mutable uint32_t candidate_ = 0;

public:
    mutable std::string name; ///< Just do what ever you want with this field.

    friend class Defx;
    friend class PrimOp;
    friend class Scope;
    friend class World;
    friend class Cleaner;
};

namespace detail {
    inline std::ostream& stream(std::ostream& out, const Def* def) { return def->stream(out); }
}

//------------------------------------------------------------------------------

uint64_t UseHash::operator () (Use use) const {
    return hash_combine(hash_begin(use->gid()), use.index());
}

//------------------------------------------------------------------------------

template<>
struct Hash<ArrayRef<const Def*>> {
    uint64_t operator () (ArrayRef<const Def*> defs) const {
        uint64_t seed = hash_begin(defs.size());
        for (auto def : defs)
            seed = hash_combine(seed, def ? def->gid() : uint64_t(-1));
        return seed;
    }
};

template<>
struct Hash<Array<const Def*>> {
    uint64_t operator () (const Array<const Def*> defs) const { return Hash<ArrayRef<const Def*>>()(defs); }
};

//------------------------------------------------------------------------------

}

#endif
