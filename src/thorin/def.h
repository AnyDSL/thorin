#ifndef THORIN_DEF_H
#define THORIN_DEF_H

#include <optional>
#include <vector>

#include "thorin/debug.h"
#include "thorin/tables.h"
#include "thorin/util/array.h"
#include "thorin/util/cast.h"
#include "thorin/util/hash.h"
#include "thorin/util/stream.h"

namespace thorin {

template<class T>
struct GIDLt {
    bool operator()(T a, T b) const { return a->gid() < b->gid(); }
};

template<class T>
struct GIDHash {
    static hash_t hash(T n) { return thorin::murmur3(n->gid()); }
    static bool eq(T a, T b) { return a == b; }
    static T sentinel() { return T(1); }
};

template<class Key, class Value>
using GIDMap = thorin::HashMap<Key, Value, GIDHash<Key>>;
template<class Key>
using GIDSet = thorin::HashSet<Key, GIDHash<Key>>;

//------------------------------------------------------------------------------

class App;
class Axiom;
class Param;
class Def;
class Stream;
class Tracker;
class World;

typedef ArrayRef<const Def*> Defs;

//------------------------------------------------------------------------------

/**
 * Similar to @p World::extract but also works on @p Sigma%s, and @p Arr%s.
 * If @p def is a value (see @p Def::is_value), proj resorts to @p World::extract.
 */
const Def* proj(const Def* def, u64 arity, u64 i, const Def* dbg = {});

template<class T = u64> std::optional<T> isa_lit(const Def*);
template<class T = u64> T as_lit(const Def* def);

//------------------------------------------------------------------------------

/**
 * References a user.
 * A @p Def @c u which uses @p Def @c d as @c i^th operand is a @p Use with @p index_ @c i of @p Def @c d.
 */
class Use {
public:
    Use() {}
    Use(const Def* def, size_t index)
        : tagged_ptr_(def, index)
    {}

    bool is_used_as_type() const { return index() == size_t(-1); }
    size_t index() const { return tagged_ptr_.index(); }
    const Def* def() const { return tagged_ptr_.ptr(); }
    operator const Def*() const { return tagged_ptr_; }
    const Def* operator->() const { return tagged_ptr_; }
    bool operator==(Use other) const { return this->tagged_ptr_ == other.tagged_ptr_; }

private:
    TaggedPtr<const Def, size_t> tagged_ptr_;
};

struct UseHash {
    inline static hash_t hash(Use use);
    static bool eq(Use u1, Use u2) { return u1 == u2; }
    static Use sentinel() { return Use((const Def*)(-1), u16(-1)); }
};

typedef HashSet<Use, UseHash> Uses;

enum class Sort { Term, Type, Kind, Space };

//------------------------------------------------------------------------------

/**
 * Base class for all @p Def%s.
 * The data layout (see @p World::alloc) looks like this:
@verbatim
|| Def debug type | op(0) ... op(num_ops-1) ||
      |-------------extended_ops-------------|
@endverbatim
 * This means that any subclass of @p Def must not introduce additional members.
 * See also @p Def::extended_ops.
 */
class Def : public RuntimeCast<Def>, public Streamable<Def> {
public:
    using NormalizeFn = const Def* (*)(const Def*, const Def*, const Def*, const Def*);

private:
    Def& operator=(const Def&) = delete;
    Def(const Def&) = delete;

protected:
    /// Constructor for a @em structural Def.
    Def(node_t, const Def* type, Defs ops, fields_t fields, const Def* dbg);
    /// Constructor for a @em nominal Def.
    Def(node_t, const Def* type, size_t num_ops, fields_t fields, const Def* dbg);
    virtual ~Def() {}

public:
    /// @name node-specific stuff
    //@{
    node_t node() const { return node_; }
    const char* node_name() const;
    //@}
    /// @name type
    //@{
    const Def* type() const { assert(node() != Node::Space); return type_; }
    Sort level() const;
    Sort sort() const;
    unsigned order() const { /*TODO assertion*/return order_; }
    const Def* arity() const;
    //@}
    /// @name ops
    //@{
    template<size_t N = size_t(-1)>
    auto ops() const {
        if constexpr (N == size_t(-1)) {
            return Defs(num_ops_, ops_ptr());
        } else {
            return ops().template to_array<N>();
        }
    }
    const Def* op(size_t i) const { return ops()[i]; }
    size_t num_ops() const { return num_ops_; }
    /// Includes @p debug (if not @c nullptr), @p type() (if not @p Space), and then the other @p ops() (if @p is_set) in this order.
    Defs extended_ops() const;
    const Def* extended_op(size_t i) const { return extended_ops()[i]; }
    size_t num_extended_ops() const { return extended_ops().size(); }
    Def* set(size_t i, const Def* def);
    Def* set(Defs ops) { for (size_t i = 0, e = num_ops(); i != e; ++i) set(i, ops[i]); return this; }
    void unset(size_t i);
    void unset() { for (size_t i = 0, e = num_ops(); i != e; ++i) unset(i); }
    /// @c true if all operands are set or @p num_ops == 0, @c false if all operands are @c nullptr, asserts otherwise.
    bool is_set() const;
    /// @p Param%s and @em nominals are @em not const; @p Axiom%s are always const; everything else const iff their @p extended_ops are const.
    bool is_const() const { return const_; }
    //@}
    /// @name uses
    //@{
    const Uses& uses() const { return uses_; }
    Array<Use> copy_uses() const { return Array<Use>(uses_.begin(), uses_.end()); }
    size_t num_uses() const { return uses().size(); }
    //@}
    /// @name split def via proj%s
    //@{
    /// Splits this @p Def into an array.
    /// Applies @p f to each @p proj%ected element.
    template<size_t A, class F>
    auto split(F f) const {
        using R = decltype(f(this));

        auto a = as_lit(arity());
        assert(a == A);
        std::array<R, A> array;
        for (size_t i = 0; i != A; ++i)
            array[i] = f(proj(this, a, i));
        return array;
    }
    template<class F>
    auto split(size_t a, F f) const {
        using R = decltype(f(this));
        return Array<R>(a, [&](size_t i) { return f(proj(this, a, i)); });
    }
    /// Splits this @p Def into an array.
    template<size_t A> auto split(        ) const { return split<A>(   [](const Def* def) { return def; }); }
                       auto split(size_t a) const { return split   (a, [](const Def* def) { return def; }); }
    const Def* out(size_t i, const Def* dbg = {}) const { return proj(this, num_outs(), i, dbg); }
    Array<const Def*> outs() const { return Array<const Def*>(num_outs(), [&](auto i) { return out(i); }); }
    size_t num_outs() const {
        if (auto a = isa_lit(arity())) return *a;
        return 1;
    }
    //@}
    /// @name external handling
    //@{
    bool is_external() const;
    void make_external();
    void make_internal();
    //@}
    /// @name Debug
    //@{
    const Def* dbg() const { return dbg_; }
    Debug debug() const { return dbg_; }
    void set_dbg(const Def* dbg) const { dbg_ = dbg; }
    void set_name(const std::string&) const;
    const Def* debug_history() const; ///< In Debug build if World::enable_history is true, this thing keeps the gid to track a history of gid%s.
    std::string unique_name() const;  ///< name + "_" + gid
    //@}
    /// @name casts
    //@{
    /// If @c this is @em nominal, it will cast constness away and perform a dynamic cast to @p T.
    template<class T = Def, bool invert = false> T* isa_nominal() const {
        if constexpr(std::is_same<T, Def>::value)
            return nominal_ ^ invert ? const_cast<Def*>(this) : nullptr;
        else
            return nominal_ ^ invert ? const_cast<Def*>(this)->template isa<T>() : nullptr;
    }
    template<class T = Def> const T* isa_structural() const { return isa_nominal<T, true>(); }
    /// Asserts that @c this is a @em nominal, casts constness away and performs a static cast to @p T (checked in Debug build).
    template<class T = Def, bool invert = false> T* as_nominal() const {
        assert(nominal_ ^ invert);
        if constexpr(std::is_same<T, Def>::value)
            return const_cast<Def*>(this);
        else
            return const_cast<Def*>(this)->template as<T>();
    }
    template<class T = Def> const T* as_structural() const { return as_nominal<T, true>(); }
    //@}
    /// @name retrieve @p Param for @em nominals.
    //@{
    /// Only returns a @p Param for this @em nominal if it has ever been created.
    const Param* has_param() { return param_ ? param() : nullptr; }
    const Param* param(const Def* dbg);
    const Def* param(size_t i, const Def* dbg) { return proj((const Def*) param(), num_params(), i, dbg); }
    const Param* param();       ///< Wrapper instead of default argument for easy access in @c gdb.
    const Def* param(size_t i); ///< Wrapper instead of default argument for easy access in @c gdb.
    Array<const Def*> params() { return Array<const Def*>(num_params(), [&](auto i) { return param(i); }); }
    size_t num_params();
    //@}
    /// @name rewrites last op by substituting @p param with @p arg.
    //@{
    Array<const Def*> apply(const Def* arg) const;
    Array<const Def*> apply(const Def* arg);
    //@}
    /// @name reduce/subst
    //@{
    const Def* reduce() const;
    /// @p rebuild%s this @p Def while using @p new_op as substitute for its @p i'th @p op
    const Def* refine(size_t i, const Def* new_op) const;
    //@}
    /// @name misc getters
    //@{
    fields_t fields() const { return fields_; }
    size_t gid() const { return gid_; }
    hash_t hash() const { return hash_; }
    World& world() const {
        if (node()                 == Node::Space) return *world_;
        if (type()->node()         == Node::Space) return *type()->world_;
        if (type()->type()->node() == Node::Space) return *type()->type()->world_;
        return *type()->type()->type()->world_;
    }
    //@}
    /// @name replace
    //@{
    void replace(Tracker) const;
    bool is_replaced() const { return substitute_ != nullptr; }
    //@}
    /// @name virtual methods
    //@{
    virtual const Def* rebuild(World&, const Def*, Defs, const Def*) const { THORIN_UNREACHABLE; }
    virtual Def* stub(World&, const Def*, const Def*) { THORIN_UNREACHABLE; }
    virtual const Def* restructure() { return nullptr; }
    //@}
    ///@{ @name stream
    Stream& stream(Stream& s) const;
    Stream& stream(Stream& s, size_t max) const;
    Stream& stream_assignment(Stream&) const;
    void dump() const;
    void dump(size_t) const;
    //@}
    bool equal(const Def* other) const;

protected:
    const Def** ops_ptr() const { return reinterpret_cast<const Def**>(reinterpret_cast<char*>(const_cast<Def*>(this + 1))); }
    void finalize();

    union {
        /// @p Axiom%s use this member to store their normalize function and the currying depth.
        TaggedPtr<std::remove_pointer_t<NormalizeFn>, u16> normalizer_depth_;
        /// Curried @p App%s of @p Axiom%s use this member to propagate the @p Axiom in question and the current currying depth.
        TaggedPtr<const Axiom, u16> axiom_depth_;
    };
    fields_t fields_;
    node_t node_;
    unsigned nominal_ :  1;
    unsigned param_   :  1;
    unsigned const_   :  1;
    unsigned order_   : 13;
    u32 gid_;
    u32 num_ops_;
    hash_t hash_;
    mutable Uses uses_;
    mutable const Def* substitute_ = nullptr; // TODO remove this
    mutable const Def* dbg_;
    union {
        const Def* type_;
        mutable World* world_;
    };

    friend class Cleaner;
    friend class Tracker;
    friend class World;
    friend void swap(World&, World&);
};

template<class T>
const T* isa(fields_t f, const Def* def) {
    if (auto d = def->template isa<T>(); d && d->fields() == f) return d;
    return nullptr;
}

template<class T>
const T* as([[maybe_unused]] fields_t f, const Def* def) { assert(isa<T>(f, def)); return def; }

//------------------------------------------------------------------------------

template<class To>
using DefMap  = GIDMap<const Def*, To>;
using DefSet  = GIDSet<const Def*>;
using Def2Def = DefMap<const Def*>;

using DefDef = std::tuple<const Def*, const Def*>;

struct DefDefHash {
    static hash_t hash(DefDef pair) {
        hash_t hash = std::get<0>(pair)->gid();
        hash = murmur3(hash, std::get<1>(pair)->gid());
        hash = murmur3_finalize(hash, 8);
        return hash;
    }
    static bool eq(DefDef p1, DefDef p2) { return p1 == p2; }
    static DefDef sentinel() { return {nullptr, nullptr}; }
};

struct DefsHash {
    static hash_t hash(Defs defs) {
        auto seed = hash_begin(defs.front()->gid());
        for (auto def : defs.skip_front())
            seed = hash_combine(seed, def->gid());
        return seed;
    }
    static bool eq(Defs d1, Defs d2) { return d1 == d2; }
    static Defs sentinel() { return Defs(); }
};

template<class To>
using DefDefMap  = HashMap<DefDef, To, DefDefHash>;
using DefDefSet  = HashSet<DefDef, DefDefHash>;
using DefDef2Def = DefDefMap<const Def*>;

template<class To>
using NomMap  = GIDMap<Def*, To>;
using NomSet  = GIDSet<Def*>;
using Nom2Nom = NomMap<Def*>;

//------------------------------------------------------------------------------

class Param : public Def {
private:
    Param(const Def* type, Def* nominal, const Def* dbg)
        : Def(Node, type, Defs{nominal}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    Def* nominal() const { return op(0)->as_nominal(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Param;
    friend class World;
};

template<class To>
using ParamMap    = GIDMap<const Param*, To>;
using ParamSet    = GIDSet<const Param*>;
using Param2Param = ParamMap<const Param*>;

class Space : public Def {
private:
    Space(World& world)
        : Def(Node, reinterpret_cast<const Def*>(&world), Defs{}, 0, nullptr)
    {}

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Space;
    friend class World;
};

class Kind : public Def {
private:
    Kind(World&);

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Kind;
    friend class World;
};

class Lit : public Def {
private:
    Lit(const Def* type, fields_t val, const Def* dbg)
        : Def(Node, type, Defs{}, val, dbg)
    {}

public:
    template<class T = fields_t>
    T get() const { static_assert(sizeof(T) <= 8); return bitcast<T>(fields_); }
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Lit;
    friend class World;
};

template<class T = u64> std::optional<T> isa_lit(const Def* def) {
    if (def == nullptr) return {};
    if (auto lit = def->isa<Lit>()) return lit->get<T>();
    return {};
}

template<class T = u64> T as_lit(const Def* def) { return def->as<Lit>()->get<T>(); }

class Tracker {
public:
    Tracker()
        : def_(nullptr)
    {}
    Tracker(const Def* def)
        : def_(def)
    {}

    operator const Def*() const { return def(); }
    const Def* operator->() const { return def(); }
    const Def* def() const {
        if (def_ != nullptr) {
            while (auto repr = def_->substitute_)
                def_ = repr;
        }
        return def_;
    }

    std::ostream& operator<<(std::ostream& os) const { return os << def(); }

private:
    mutable const Def* def_;
};

class Nat : public Def {
private:
    Nat(World& world);

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Nat;
    friend class World;
};

class Proxy : public Def {
private:
    Proxy(const Def* type, Defs ops, tag_t index, flags_t flags, const Def* dbg)
        : Def(Node, type, ops, (nat_t(index) << 32_u64) | nat_t(flags), dbg)
    {}

public:
    /// @name misc getters
    //@{
    tag_t id() const { return fields() >> 32_u64; }
    flags_t flags() const { return fields(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Proxy;
    friend class World;
};

/**
 * A global variable in the data segment.
 * A @p Global may be mutable or immutable.
 * @em deprecated. WILL BE REMOVED
 */
class Global : public Def {
private:
    Global(const Def* type, const Def* id, const Def* init, bool is_mutable, const Def* dbg)
        : Def(Node, type, {id, init}, is_mutable, dbg)
    {}

public:
    /// @name ops
    //@{
    /// This thing's sole purpose is to differentiate on global from another.
    const Def* id() const { return op(0); }
    const Def* init() const { return op(1); }
    //@}
    /// @name type
    //@{
    const App* type() const;
    const Def* alloced_type() const;
    //@}
    /// @name misc getters
    //@{
    bool is_mutable() const { return fields(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World& to, const Def* type, Defs ops, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Global;
    friend class World;
};

hash_t UseHash::hash(Use use) { return hash_combine(hash_begin(u16(use.index())), hash_t(use->gid())); }

Stream& operator<<(Stream&, const Def* def);
Stream& operator<<(Stream&, std::pair<const Def*, const Def*>);

//------------------------------------------------------------------------------

}

#endif
