#ifndef THORIN_DEF_H
#define THORIN_DEF_H

#include <optional>
#include <string>
#include <variant>
#include <vector>

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

class Axiom;
class Lam;
class Param;
class Pi;
class Def;
class Stream;
class Tracker;
class World;

typedef ArrayRef<const Def*> Defs;

//------------------------------------------------------------------------------

using Name = std::variant<const char*, std::string, const Def*>;

struct Debug {
    Debug(Name name,
          Name filename = "",
          nat_t front_line = nat_t(-1),
          nat_t front_col  = nat_t(-1),
          nat_t back_line  = nat_t(-1),
          nat_t back_col   = nat_t(-1),
          const Def* meta  = nullptr)
        : data(std::make_tuple(name, filename, front_line, front_col, back_line, back_col, meta))
    {}
    Debug(Name filename, nat_t front_line, nat_t front_col, nat_t back_line, nat_t back_col)
        : Debug("", filename, front_line, front_col, back_line, back_col)
    {}
    Debug(const Def* def = nullptr)
        : data(def)
    {}

    auto& operator*() { return data; }

    std::variant<std::tuple<Name, Name, nat_t, nat_t, nat_t, nat_t, const Def*>, const Def*> data;
};

namespace detail {
    const Def* world_extract(World&, const Def*, u64, Debug dbg = {});
}

/**
 * Similar to @p World::extract but also works on @p Sigma%s and @p Arr%s and considers @p Union%s as scalars.
 * If @p def is a value (see @p Def::is_value), proj resorts to @p World::extract.
 * You can disable this behavior via @p no_extract.
 * Useful within @p World::extract itself to prevent endless recursion.
 */
template<bool no_extract = false> const Def* proj(const Def* def, u64 arity, u64 i);

/**
 * Same as above but infers the arity from @p def.
 * @attention { Think twice whether this is sound due to 1-tuples being folded.
 * It's always a good idea to pass an appropriate arity along. }
 */
template<bool = false> const Def* proj(const Def* def, u64 i);
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
    const Def* type() const { assert(node() != Node::Universe); return type_; }
    int sort() const;
    unsigned order() const { /*TODO assertion*/return order_; }
    const Def* arity() const;
    const Def* tuple_arity() const;
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
    /// Includes @p debug (if not @c nullptr), @p type() (if not @p Universe), and then the other @p ops() (if @p is_set) in this order.
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
    /// @name split def via extracts
    //@{
    /// Splits this @p Def into an array by using @p arity many @p Extract%s.
    /// Applies @p f to each extracted element.
    template<size_t N = size_t(-1), class F>
    auto split(F f) const {
        using R = decltype(f(this));

        if constexpr (N == size_t(-1)) {
            auto a = isa_lit(tuple_arity());
            auto lit = a ? *a : 1;
            return Array<R>(lit, [&](size_t i) { return f(proj(this, lit, i)); });
        } else {
            auto a = as_lit(tuple_arity());
            assert(a == N);
            std::array<R, N> array;
            for (size_t i = 0; i != N; ++i)
                array[i] = f(proj(this, a, i));
            return array;
        }
    }
    /// Splits this @p Def into an array by using @p arity many @p Extract%s.
    template<size_t N = size_t(-1)> auto split() const { return split<N>([](const Def* def) { return def; }); }
    const Def* out(size_t i, Debug dbg = {}) const { return detail::world_extract(world(), this, i, dbg); }
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
    const Def* debug() const { return debug_; }
    void set_debug(Debug dbg) const;
    void set_name(const std::string&) const;
    const Def* debug_history() const; ///< In Debug build if World::enable_history is true, this thing keeps the gid to track a history of gid%s.
    std::string name() const;
    std::string unique_name() const;  ///< name + "_" + gid
    std::string filename() const;
    nat_t front_line() const;
    nat_t front_col() const;
    nat_t back_line() const;
    nat_t back_col() const;
    std::string loc() const;
    const Def* meta() const;
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
    const Param* param(Debug dbg);
    const Def* param(size_t i, Debug dbg) { return detail::world_extract(world(), (const Def*) param(), i, dbg); }
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
        if (node()                         == Node::Universe) return *world_;
        if (type()->node()                 == Node::Universe) return *type()->world_;
        if (type()->type()->node()         == Node::Universe) return *type()->type()->world_;
        if (type()->type()->type()->node() == Node::Universe) return *type()->type()->type()->world_;
        THORIN_UNREACHABLE;
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
    virtual bool is_value() const { return type()->is_type();                } ///< Anything that cannot appear as a type such as @c 23 or @c (int, bool).
    virtual bool is_type()  const { return type()->is_kind();                } ///< Anything that can be the @p type of a value (see @p is_value).
    virtual bool is_kind()  const { return type()->node() == Node::Universe; } ///< Anything that can be the @p type of a type (see @p is_type).
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
    unsigned const_   :  1;
    unsigned order_   : 14;
    u32 gid_;
    u32 num_ops_;
    hash_t hash_;
    mutable Uses uses_;
    mutable const Def* substitute_ = nullptr; // TODO remove this
    mutable const Def* debug_;
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
    static hash_t hash(DefDef pair) { return hash_combine(hash_begin(std::get<0>(pair)->gid()), std::get<1>(pair)->gid()); }
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

class Universe : public Def {
private:
    Universe(World& world)
        : Def(Node, reinterpret_cast<const Def*>(&world), Defs{}, 0, nullptr)
    {}

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    bool is_value() const override;
    bool is_type()  const override;
    bool is_kind()  const override;
    //@}

    static constexpr auto Node = Node::Universe;
    friend class World;
};

class Kind : public Def {
private:
    Kind(World&);

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    bool is_value() const override;
    bool is_type()  const override;
    bool is_kind()  const override;
    //@}

    static constexpr auto Node = Node::Kind;
    friend class World;
};

class Axiom : public Def {
private:
    Axiom(NormalizeFn normalizer, const Def* type, tag_t tag, flags_t flags, const Def* dbg);

public:
    /// @name misc getters
    //@{
    tag_t tag() const { return fields() >> 32_u64; }
    flags_t flags() const { return fields(); }
    NormalizeFn normalizer() const { return normalizer_depth_.ptr(); }
    u16 currying_depth() const { return normalizer_depth_.index(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Axiom;
    friend class World;
};

class Bot : public Def {
private:
    Bot(const Def* type, const Def* dbg)
        : Def(Node, type, Defs{}, 0, dbg)
    {}

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Bot;
    friend class World;
};

class Top : public Def {
private:
    Top(const Def* type, const Def* dbg)
        : Def(Node, type, Defs{}, 0, dbg)
    {}

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Top;
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

/// A function type AKA Pi type.
class Pi : public Def {
protected:
    /// Constructor for a @em structural Pi.
    Pi(const Def* type, const Def* domain, const Def* codomain, const Def* dbg)
        : Def(Node, type, {domain, codomain}, 0, dbg)
    {}
    /// Constructor for a @em nominal Pi.
    Pi(const Def* type, const Def* dbg)
        : Def(Node, type, 2, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* domain() const { return op(0); }
    const Def* domain(size_t i) const;
    Array<const Def*> domains() const { return Array<const Def*>(num_domains(), [&](size_t i) { return domain(i); }); }
    size_t num_domains() const;
    const Def* codomain() const { return op(1); }
    const Def* codomain(size_t i) const;
    bool is_cn() const { return codomain()->isa<Bot>(); }
    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;
    //@}
    /// @name setters for @em nominal @p Pi.
    //@{
    Pi* set_domain(const Def* domain) { return Def::set(0, domain)->as<Pi>(); }
    Pi* set_domain(Defs domains);
    Pi* set_codomain(const Def* codomain) { return Def::set(1, codomain)->as<Pi>(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Pi* stub(World&, const Def*, const Def*) override;
    const Pi* restructure();
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Pi;
    friend class World;
};

class Lam : public Def {
public:
    // TODO make these thigns axioms
    enum class Intrinsic : u8 {
        None,                       ///< Not an intrinsic.
        _Accelerator_Begin,
        CUDA = _Accelerator_Begin,  ///< Internal CUDA-Backend.
        NVVM,                       ///< Internal NNVM-Backend.
        OpenCL,                     ///< Internal OpenCL-Backend.
        AMDGPU,                     ///< Internal AMDGPU-Backend.
        HLS,                        ///< Internal HLS-Backend.
        Parallel,                   ///< Internal Parallel-CPU-Backend.
        Spawn,                      ///< Internal Parallel-CPU-Backend.
        Sync,                       ///< Internal Parallel-CPU-Backend.
        CreateGraph,                ///< Internal Flow-Graph-Backend.
        CreateTask,                 ///< Internal Flow-Graph-Backend.
        CreateEdge,                 ///< Internal Flow-Graph-Backend.
        ExecuteGraph,               ///< Internal Flow-Graph-Backend.
        Vectorize,                  ///< External vectorizer.
        _Accelerator_End,
        Reserve = _Accelerator_End, ///< Intrinsic memory reserve function
        Atomic,                     ///< Intrinsic atomic function
        CmpXchg,                    ///< Intrinsic cmpxchg function
        Undef,                      ///< Intrinsic undef function
        PeInfo,                     ///< Partial evaluation debug info.
    };

    /// calling convention
    enum class CC : u8 {
        C,          ///< C calling convention.
        Device,     ///< Device calling convention. These are special functions only available on a particular device.
    };

private:
    Lam(const Pi* pi, const Def* filter, const Def* body, const Def* dbg)
        : Def(Node, pi, {filter, body}, 0, dbg)
    {}
    Lam(const Pi* pi, CC cc, Intrinsic intrinsic, const Def* dbg)
        : Def(Node, pi, 2, u64(cc) << 8_u64 | u64(intrinsic), dbg)
    {}

public:
    /// @name type
    //@{
    const Pi* type() const { return Def::type()->as<Pi>(); }
    const Def* domain() const { return type()->domain(); }
    const Def* domain(size_t i) const { return type()->domain(i); }
    Array<const Def*> domains() const { return type()->domains(); }
    size_t num_domains() const { return type()->num_domains(); }
    const Def* codomain() const { return type()->codomain(); }
    //@}
    /// @name ops
    //@{
    const Def* filter() const { return op(0); }
    const Def* body() const { return op(1); }
    //@}
    /// @name params
    //@{
    const Def* mem_param(thorin::Debug dbg = {});
    const Def* ret_param(thorin::Debug dbg = {});
    //@}
    /// @name setters
    //@{
    Lam* set(size_t i, const Def* def) { return Def::set(i, def)->as<Lam>(); }
    Lam* set(Defs ops) { return Def::set(ops)->as<Lam>(); }
    Lam* set(const Def* filter, const Def* body) { return set({filter, body}); }
    Lam* set_filter(const Def* filter) { return set(0_s, filter); }
    Lam* set_body(const Def* body) { return set(1, body); }
    //@}
    /// @name setters: sets filter to @c false and sets the body by @p App -ing
    //@{
    void app(const Def* callee, const Def* arg, Debug dbg = {});
    void app(const Def* callee, Defs args, Debug dbg = {});
    void branch(const Def* cond, const Def* t, const Def* f, const Def* mem, Debug dbg = {});
    void match(const Def* val, Defs cases, const Def* mem, Debug dbg = {});
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Lam* stub(World&, const Def*, const Def*) override;
    bool is_value() const override;
    bool is_type()  const override;
    //@}
    /// @name get/set fields - Intrinsic and CC
    //@{
    Intrinsic intrinsic() const { return Intrinsic(fields() & 0x00ff_u64); }
    void set_intrinsic(Intrinsic intrin) { fields_ = (fields_ & 0xff00_u64) | u64(intrin); }
    void set_intrinsic(); ///< Sets Intrinsic derived on this @p Lam's @p name.
    CC cc() const { return CC(fields() >> 8_u64); }
    void set_cc(CC cc) { fields_ = (fields_ & 0x00ff_u64) | u64(cc) << 8_u64; }
    //@}

    bool is_basicblock() const;
    bool is_returning() const;
    bool is_intrinsic() const;
    bool is_accelerator() const;

    static constexpr auto Node = Node::Lam;
    friend class World;
};

template<class To>
using LamMap  = GIDMap<Lam*, To>;
using LamSet  = GIDSet<Lam*>;
using Lam2Lam = LamMap<Lam*>;

class App : public Def {
private:
    App(const Axiom* axiom, u16 currying_depth, const Def* type, const Def* callee, const Def* arg, const Def* dbg)
        : Def(Node, type, {callee, arg}, 0, dbg)
    {
        axiom_depth_.set(axiom, currying_depth);
    }

public:
    /// @name ops
    ///@{
    const Def* callee() const { return op(0); }
    const App* decurry() const { return callee()->as<App>(); } ///< Returns the @p callee again as @p App.
    const Pi* callee_type() const { return callee()->type()->as<Pi>(); }
    const Def* arg() const { return op(1); }
    const Def* arg(size_t i, Debug dbg = {}) const { return arg()->out(i, dbg); }
    Array<const Def*> args() const { return arg()->outs(); }
    size_t num_args() const { return arg()->num_outs(); }
    //@}
    /// @name split arg
    //@{
    template<size_t N = size_t(-1), class F> auto args(F f) const { return arg()->split<N, F>(f); }
    template<size_t N = size_t(-1)> auto args() const { return arg()->split<N>(); }
    //@}
    /// @name get axiom and current currying depth
    //@{
    const Axiom* axiom() const { return axiom_depth_.ptr(); }
    u16 currying_depth() const { return axiom_depth_.index(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::App;
    friend class World;
};

class CPS2DS : public Def {
private:
    CPS2DS(const Def* type, const Def* cps, const Def* dbg)
        : Def(Node, type, { cps }, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* cps() const { return op(0); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::CPS2DS;
    friend class World;
};

class DS2CPS : public Def {
private:
    DS2CPS(const Def* type, const Def* ds, const Def* dbg)
        : Def(Node, type, { ds }, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* ds() const { return op(0); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::DS2CPS;
    friend class World;
};

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

class Sigma : public Def {
private:
    /// Constructor for a @em structural Sigma.
    Sigma(const Def* type, Defs ops, const Def* dbg)
        : Def(Node, type, ops, 0, dbg)
    {}
    /// Constructor for a @em nominal Sigma.
    Sigma(const Def* type, size_t size, const Def* dbg)
        : Def(Node, type, size, 0, dbg)
    {}

public:
    /// @name setters
    //@{
    Sigma* set(size_t i, const Def* def) { return Def::set(i, def)->as<Sigma>(); }
    Sigma* set(Defs ops) { return Def::set(ops)->as<Sigma>(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Sigma* stub(World&, const Def*, const Def*) override;
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Sigma;
    friend class World;
};

/// Data constructor for a @p Sigma.
class Tuple : public Def {
private:
    Tuple(const Def* type, Defs args, const Def* dbg)
        : Def(Node, type, args, 0, dbg)
    {}

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Tuple;
    friend class World;
};

class Union : public Def {
private:
    /// Constructor for a @em structural Union.
    Union(const Def* type, Defs ops, const Def* dbg)
        : Def(Node, type, ops, 0, dbg)
    {}
    /// Constructor for a @em nominal Union.
    Union(const Def* type, size_t size, const Def* dbg)
        : Def(Node, type, size, 0, dbg)
    {}

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Union* stub(World&, const Def*, const Def*) override;
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Union;
    friend class World;
};

class Which : public Def {
private:
    Which(const Def* type, const Def* value, const Def* dbg)
        : Def(Node, type, {value}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* value() const { return op(0); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Which;
    friend class World;
};

class Arr : public Def {
private:
    /// Constructor for a @em structural Arr.
    Arr(const Def* type, const Def* shape, const Def* body, const Def* dbg)
        : Def(Node, type, {shape, body}, 0, dbg)
    {}
    /// Constructor for a @em nominal Arr.
    Arr(const Def* type, const Def* shape, const Def* dbg)
        : Def(Node, type, 2, 0, dbg)
    {
        Def::set(0, shape);
    }

public:
    /// @name ops
    //@{
    const Def* shape() const { return op(0); }
    const Def* body() const { return op(1); }
    //@}
    /// @name methods for nominals
    //@{
    Arr* set(const Def* body) { return Def::set(1, body)->as<Arr>(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    Arr* stub(World&, const Def*, const Def*) override;
    const Def* restructure();
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Arr;
    friend class World;
};

class Pack : public Def {
private:
    Pack(const Def* type, const Def* body, const Def* dbg)
        : Def(Node, type, {body}, 0, dbg)
    {}

public:
    /// @name getters
    //@{
    const Def* body() const { return op(0); }
    const Arr* type() const { return Def::type()->as<Arr>(); }
    const Def* shape() const { return type()->shape(); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Pack;
    friend class World;
};

inline bool is_sigma_or_arr (const Def* def) { return def->isa<Sigma>() || def->isa<Arr >(); }
inline bool is_tuple_or_pack(const Def* def) { return def->isa<Tuple>() || def->isa<Pack>(); }

/// Extracts from a @p Sigma or @p Variadic typed @p Def the element at position @p index.
class Extract : public Def {
private:
    Extract(const Def* type, const Def* tuple, const Def* index, const Def* dbg)
        : Def(Node, type, {tuple, index}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* tuple() const { return op(0); }
    const Def* index() const { return op(1); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Extract;
    friend class World;
};

/**
 * Creates a new @p Tuple/@p Pack by inserting @p value at position @p index into @p tuple.
 * @attention { This is a @em functional insert.
 *              The @p tuple itself remains untouched.
 *              The @p Insert itself is a @em new @p Tuple/@p Pack which contains the inserted @p value. }
 */
class Insert : public Def {
private:
    Insert(const Def* tuple, const Def* index, const Def* value, const Def* dbg)
        : Def(Node, tuple->type(), {tuple, index, value}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* tuple() const { return op(0); }
    const Def* index() const { return op(1); }
    const Def* value() const { return op(2); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Insert;
    friend class World;
};

/// Matches against a value, using @p ptrns.
class Match : public Def {
private:
    Match(const Def* type, Defs ops, const Def* dbg)
        : Def(Node, type, ops, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* arg() const { return op(0); }
    Defs ptrns() const { return ops().skip_front(); }
    const Def* ptrn(size_t i) const { return op(i + 1); }
    size_t num_ptrns() const { return num_ops() - 1; }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    //@}

    static constexpr auto Node = Node::Match;
    friend class World;
};

/// Pattern type.
class Case : public Def {
private:
    Case(const Def* type, const Def* domain, const Def* codomain, const Def* dbg)
        : Def(Node, type, {domain, codomain}, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    const Def* domain() const { return op(0); }
    const Def* codomain() const { return op(1); }
    //@}
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Case;
    friend class World;
};

/// Pattern value.
class Ptrn : public Def {
private:
    Ptrn(const Def* type, const Def* dbg)
        : Def(Node, type, 2, 0, dbg)
    {}

public:
    /// @name ops
    //@{
    Ptrn* set(const Def* matcher, const Def* body) { return Def::set({matcher, body})->as<Ptrn>(); }
    const Def* matcher() const { return op(0); }
    const Def* body() const { return op(1); }
    /// @name type
    //@{
    const Case* type() const { return Def::type()->as<Case>(); }
    const Def*  domain() const { return type()->domain(); }
    const Def*  codomain() const { return type()->codomain(); }
    //@}
    /// @name misc getters
    //@{
    bool is_trivial() const;
    bool matches(const Def*) const;
    //@}
    /// @name virtual methods
    //@{
    Ptrn* stub(World&, const Def*, const Def*) override;
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Ptrn;
    friend class World;
};

class Nat : public Def {
private:
    Nat(World& world);

public:
    /// @name virtual methods
    //@{
    const Def* rebuild(World&, const Def*, Defs, const Def*) const override;
    bool is_value() const override;
    bool is_type()  const override;
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
    tag_t index() const { return fields() >> 32_u64; }
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
    bool is_value() const override;
    bool is_type()  const override;
    //@}

    static constexpr auto Node = Node::Global;
    friend class World;
};

hash_t UseHash::hash(Use use) { return hash_combine(hash_begin(u16(use.index())), hash_t(use->gid())); }

template<tag_t tag> struct Tag2Def_ { using type = App; };
template<> struct Tag2Def_<Tag::Mem> { using type = Axiom; };
template<tag_t tag> using Tag2Def = typename Tag2Def_<tag>::type;

Stream& operator<<(Stream&, const Def* def);
Stream& operator<<(Stream&, std::pair<const Def*, const Def*>);
inline Stream& operator<<(Stream& s, std::pair<Lam*, Lam*> p) { return operator<<(s, std::pair<const Def*, const Def*>(p.first, p.second)); }

//------------------------------------------------------------------------------

}

#endif
