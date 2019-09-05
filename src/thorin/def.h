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

namespace thorin {

template<class T>
struct GIDLt {
    bool operator()(T a, T b) const { return a->gid() < b->gid(); }
};

template<class T>
struct GIDHash {
    static u32 hash(T n) { return thorin::murmur3(n->gid()); }
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
class Def;
class Tracker;
class World;

typedef ArrayRef<const Def*> Defs;
typedef std::vector<Lam*> Lams;

//------------------------------------------------------------------------------

using Name = std::variant<const char*, std::string, const Def*>;

struct Debug {
    Debug(Name name, Name filename, u64 front_line, u64 front_col, u64 back_line, u64 back_col)
        : data(std::make_tuple(name, filename, front_line, front_col, back_line, back_col))
    {}
    Debug(Name filename, u64 front_line, u64 front_col, u64 back_line, u64 back_col)
        : Debug("", filename, front_line, front_col, back_line, back_col)
    {}
    Debug(Name name)
        : Debug(name, "", u64(-1), u64(-1), u64(-1), u64(-1))
    {}
    Debug(const Def* def = nullptr)
        : data(def)
    {}

    auto& operator*() { return data; }

    std::variant<std::tuple<Name, Name, u64, u64, u64, u64>, const Def*> data;
};

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

    size_t index() const { return tagged_ptr_.index(); }
    const Def* def() const { return tagged_ptr_.ptr(); }
    operator const Def*() const { return tagged_ptr_; }
    const Def* operator->() const { return tagged_ptr_; }
    bool operator==(Use other) const { return this->tagged_ptr_ == other.tagged_ptr_; }

private:
    TaggedPtr<const Def, size_t> tagged_ptr_;
};

struct UseHash {
    inline static u32 hash(Use use);
    static bool eq(Use u1, Use u2) { return u1 == u2; }
    static Use sentinel() { return Use((const Def*)(-1), uint16_t(-1)); }
};

typedef HashSet<Use, UseHash> Uses;

//------------------------------------------------------------------------------

template<class To>
using DefMap  = GIDMap<const Def*, To>;
using DefSet  = GIDSet<const Def*>;
using Def2Def = DefMap<const Def*>;

std::ostream& operator<<(std::ostream&, const Def*);
std::ostream& operator<<(std::ostream&, Use);

//------------------------------------------------------------------------------

/**
 * Base class for all Def%s.
 * The data layout (see World::alloc) looks like this:
\verbatim
|| Def | op(0) ... op(num_ops-1) | Extra ||
\endverbatim
 * This means that any subclass of @p Def must not introduce additional members.
 * However, you can have this Extra field.
 * See App or Lit how this is done.
 */
class Def : public RuntimeCast<Def>, public Streamable {
public:
    using RebuildFn   = const Def* (*)(const Def*, World&, const Def*, Defs, const Def*);
    using StubFn      = Def* (*)(const Def*, World&, const Def*, const Def*);
    using NormalizeFn = const Def* (*)(const Def*, const Def*, const Def*, const Def*);

private:
    Def& operator=(const Def&) = delete;
    Def(const Def&) = delete;

protected:
    /// Constructor for a @em structural Def.
    Def(uint16_t tag, RebuildFn rebuild, const Def* type, Defs ops, uint64_t fields, const Def* dbg);
    /// Constructor for a @em nominal Def.
    Def(uint16_t tag, StubFn stub, const Def* type, size_t num_ops, uint64_t fields, const Def* dbg);
    virtual ~Def() {}

public:
    enum class Sort {
        Term, Type, Kind, Universe
    };

    /// @name type, Sort
    //@{
    const Def* type() const { assert(node() != Node::Universe); return type_; }
    unsigned order() const { assert(!is_term()); return order_; }
    Sort sort() const;
    bool is_term() const { return sort() == Sort::Term; }
    bool is_type() const { return sort() == Sort::Type; }
    bool is_kind() const { return sort() == Sort::Kind; }
    bool is_universe() const { return sort() == Sort::Universe; }
    virtual const Def* arity() const;
    u64 lit_arity() const;
    //@}
    /// @name ops
    //@{
    Defs ops() const { return Defs(num_ops_, ops_ptr()); }
    const Def* op(size_t i) const { return ops()[i]; }
    size_t num_ops() const { return num_ops_; }
    Def* set(size_t i, const Def* def);
    Def* set(Defs ops) { for (size_t i = 0, e = num_ops(); i != e; ++i) set(i, ops[i]); return this; }
    void unset(size_t i);
    void unset() { for (size_t i = 0, e = num_ops(); i != e; ++i) unset(i); }
    /// @c true if all operands are set or num_ops == 0, @c false if all operands are @c nullptr, asserts otherwise.
    bool is_set() const;
    //@}
    /// @name uses
    //@{
    const Uses& uses() const { return uses_; }
    Array<Use> copy_uses() const { return Array<Use>(uses_.begin(), uses_.end()); }
    size_t num_uses() const { return uses().size(); }
    //@}
    /// @name outs
    //@{
    const Def* out(size_t i, Debug dbg = {}) const;
    size_t num_outs() const;
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
    /// In Debug build if World::enable_history is true, this thing keeps the gid to track a history of gid%s.
    const Def* debug_history() const;
    std::string name() const;
    /// name + "_" + gid
    std::string unique_name() const;
    std::string filename() const;
    u64 front_line() const;
    u64 front_col() const;
    u64 back_line() const;
    u64 back_col() const;
    std::string loc() const;
    //@}
    /// @name casts
    //@{
    /// If @c this is @em nominal, it will cast constness away and perform a dynamic cast to @p T.
    template<class T = Def> T* isa_nominal() const {
        if constexpr(std::is_same<T, Def>::value)
            return nominal_ ? const_cast<Def*>(this) : nullptr;
        else
            return nominal_ ? const_cast<Def*>(this)->template isa<T>() : nullptr;
    }

    /// Asserts that @c this is a @em nominal, casts constness away and performs a static cast to @p T (checked in Debug build).
    template<class T = Def> T* as_nominal() const {
        assert(nominal_);
        if constexpr(std::is_same<T, Def>::value)
            return const_cast<Def*>(this);
        else
            return const_cast<Def*>(this)->template as<T>();
    }
    //@}
    /// @name misc getters
    //@{
    uint64_t fields() const { return fields_; }
    uint16_t node() const { return node_; }
    size_t gid() const { return gid_; }
    u32 hash() const { return hash_; }
    World& world() const {
        if (node()                 == Node::Universe) return *world_;
        if (type()->node()         == Node::Universe) return *type()->world_;
        if (type()->type()->node() == Node::Universe) return *type()->type()->world_;
        assert(type()->type()->type()->node() == Node::Universe);
        return *type()->type()->type()->world_;
    }
    //@}
    /// @name replace
    //@{
    void replace(Tracker) const;
    bool is_replaced() const { return substitute_ != nullptr; }
    //@}
    /// @name rebuild, stub, normalize, equal
    //@{
    const Def* rebuild(World& world, const Def* type, Defs ops, const Def* dbg) const {
        assert(!isa_nominal());
        return rebuild_(this, world, type, ops, dbg);
    }
    Def* stub(World& world, const Def* type, const Def* dbg) {
        assert(isa_nominal());
        return stub_(this, world, type, dbg);
    }
    bool equal(const Def* other) const;
    //@}
    /// @name stream
    //@{
    void dump() const;
    virtual const char* op_name() const;
    virtual std::ostream& stream(std::ostream&) const;
    virtual std::ostream& stream_assignment(std::ostream&) const;
    //@}

protected:
    char* extra_ptr() { return reinterpret_cast<char*>(this) + sizeof(Def) + sizeof(const Def*)*num_ops(); }
    const char* extra_ptr() const { return const_cast<Def*>(this)->extra_ptr(); }
    template<class T> T& extra() { return reinterpret_cast<T&>(*extra_ptr()); }
    template<class T> const T& extra() const { return reinterpret_cast<const T&>(*extra_ptr()); }
    const Def** ops_ptr() const { return reinterpret_cast<const Def**>(reinterpret_cast<char*>(const_cast<Def*>(this + 1))); }
    void finalize();

    union {
        const Def* type_;
        mutable World* world_;
    };
    union {
        RebuildFn rebuild_;
        StubFn    stub_;
    };
    union {
        /// @p Axiom%s use this member to store their normalize function and the currying depth.
        TaggedPtr<std::remove_pointer_t<NormalizeFn>, u16> normalizer_depth_;
        /// Curried @p App%s of @p Axiom%s use this member to propagate the @p Axiom in question and the current currying depth.
        TaggedPtr<const Axiom, u16> axiom_depth_;
    };
    const Def* debug_;
    uint64_t fields_;
    uint16_t node_;
    unsigned nominal_ :  1;
    unsigned order_   : 15;
    u32 gid_;
    u32 num_ops_;
    u32 hash_;
    mutable Uses uses_;
    mutable const Def* substitute_ = nullptr; // TODO remove this

    friend class Cleaner;
    friend class Tracker;
    friend class World;
    friend void swap(World&, World&);
};

class Universe : public Def {
private:
    Universe(World& world)
        : Def(Node, stub, reinterpret_cast<const Def*>(&world), 0_s, 0, nullptr)
    {}

public:
    static Def* stub(const Def*, World&, const Def*, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Universe;
    friend class World;
};

class KindArity : public Def {
private:
    KindArity(World&);

public:
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::KindArity;
    friend class World;
};

class KindMulti : public Def {
private:
    KindMulti(World&);

public:
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::KindMulti;
    friend class World;
};

class KindStar : public Def {
private:
    KindStar(World&);

public:
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::KindStar;
    friend class World;
};

class Axiom : public Def {
private:
    Axiom(NormalizeFn normalizer, const Def* type, u32 tag, u32 flags, const Def* dbg);

public:
    u32 tag() const { return fields() >> 32_u64; }
    u32 flags() const { return fields(); }
    NormalizeFn normalizer() const { return normalizer_depth_.ptr(); }
    u16 currying_depth() const { return normalizer_depth_.index(); }
    static Def* stub(const Def*, World&, const Def*, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Axiom;
    friend class World;
};

class Bot : public Def {
private:
    Bot(const Def* type, const Def* dbg)
        : Def(Node, rebuild, type, Defs{}, 0, dbg)
    {}

public:
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Bot;
    friend class World;
};

class Top : public Def {
private:
    Top(const Def* type, const Def* dbg)
        : Def(Node, rebuild, type, Defs{}, 0, dbg)
    {}

public:
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Top;
    friend class World;
};

class Lit : public Def {
private:
    Lit(const Def* type, uint64_t val, const Def* dbg)
        : Def(Node, rebuild, type, Defs{}, val, dbg)
    {}

public:
    template<class T = uint64_t>
    T get() const { static_assert(sizeof(T) <= 8); return bitcast<T>(fields_); }
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Lit;
    friend class World;
};

template<class T> std::optional<T> isa_lit(const Def* def) {
    if (auto lit = def->isa<Lit>()) return lit->get<T>();
    return {};
}

template<class T> T as_lit(const Def* def) { return def->as<Lit>()->get<T>(); }

/// A function type AKA Pi type.
class Pi : public Def {
protected:
    /// Constructor for a @em structural Pi.
    Pi(const Def* type, const Def* domain, const Def* codomain, const Def* dbg)
        : Def(Node, rebuild, type, {domain, codomain}, 0, dbg)
    {}
    /// Constructor for a @em nominal Pi.
    Pi(const Def* type, const Def* dbg)
        : Def(Node, stub, type, 2, 0, dbg)
    {}

public:
    /// @name getters
    //@{
    const Def* domain() const { return op(0); }
    const Def* domain(size_t i) const;
    Array<const Def*> domains() const;
    size_t num_domains() const;
    const Def* codomain() const { return op(1); }
    bool is_cn() const;
    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;
    //@}

    /// @name setters for @em nominal @p Pi%s
    //@{
    Pi* set_domain(const Def* domain) { return Def::set(0, domain)->as<Pi>(); }
    Pi* set_domain(Defs domains);
    Pi* set_codomain(const Def* codomain) { return Def::set(1, codomain)->as<Pi>(); }
    //@}

    /// @name retrieve @p Param for @em nominal @p Pi%s.
    //@{
    const Param* param(Debug dbg = {}) const;
    const Def* param(size_t i, Debug dbg = {}) const;
    Array<const Def*> params() const;
    size_t num_params() const { return domain()->lit_arity(); }
    /// Reduces the @p codomain by rewriting this @p Pi's @p Param with @p arg in order to retrieve the codomain of a dependent function @p App.
    const Def* apply(const Def* arg) const;
    //@}

    std::ostream& stream(std::ostream&) const override;
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    static Def* stub(const Def*, World&, const Def*, const Def*);

    static constexpr auto Node = Node::Pi;
    friend class World;
};

class App : public Def {
private:
    App(const Axiom* axiom, u16 currying_depth, const Def* type, const Def* callee, const Def* arg, const Def* dbg)
        : Def(Node, rebuild, type, {callee, arg}, 0, dbg)
    {
        axiom_depth_.set(axiom, currying_depth);
    }

public:
    const Def* callee() const { return op(0); }
    const Pi* callee_type() const { return callee()->type()->as<Pi>(); }
    const Def* arg() const { return op(1); }
    const Def* arg(size_t i) const;
    Array<const Def*> args() const;
    size_t num_args() const { return callee_type()->domain()->lit_arity(); }
    const Axiom* axiom() const { return axiom_depth_.ptr(); }
    u16 currying_depth() const { return axiom_depth_.index(); }

    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::App;
    friend class World;
};

class Lam : public Def {
public:
    enum class Intrinsic : uint8_t {
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
        Match,                      ///< match(val, otherwise, (case1, cont1), (case2, cont2), ...)
        PeInfo,                     ///< Partial evaluation debug info.
        EndScope,                   ///< Dummy function which marks the end of a @p Scope.
    };

    /// calling convention
    enum class CC : uint8_t {
        C,          ///< C calling convention.
        Device,     ///< Device calling convention. These are special functions only available on a particular device.
    };

private:
    Lam(const Pi* pi, const Def* filter, const Def* body, const Def* dbg)
        : Def(Node, rebuild, pi, {filter, body}, 0, dbg)
    {}
    Lam(const Pi* pi, CC cc, Intrinsic intrinsic, const Def* dbg)
        : Def(Node, stub, pi, 2, uint64_t(cc) << 8_u64 | uint64_t(intrinsic), dbg)
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
    const App* app() const { return body()->isa<App>(); }
    //@}
    /// @name params
    //@{
    const Param* param(Debug dbg = {}) const;
    const Def* param(size_t i, Debug dbg = {}) const;
    Array<const Def*> params() const;
    size_t num_params() const { return type()->num_params(); }
    const Def* mem_param() const;
    const Def* ret_param() const;
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
    void match(const Def* val, Lam* otherwise, Defs patterns, ArrayRef<Lam*> lams, Debug dbg = {});
    //@}
    /// @name rebuild, stub
    //@{
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    static Def* stub(const Def*, World&, const Def*, const Def*);
    //@}
    /// @name get/set fields - Intrinsic and CC
    //@{
    Intrinsic intrinsic() const { return Intrinsic(fields() & 0x00ff_u64); }
    void set_intrinsic(Intrinsic intrin) { fields_ = (fields_ & 0xff00_u64) | uint64_t(intrin); }
    void set_intrinsic(); ///< Sets Intrinsic derived on this @p Lam's @p name.
    CC cc() const { return CC(fields() >> 8_u64); }
    void set_cc(CC cc) { fields_ = (fields_ & 0x00ff_u64) | uint64_t(cc) << 8_u64; }
    //@}

    Lams preds() const;
    Lams succs() const;
    bool is_basicblock() const;
    bool is_returning() const;
    bool is_intrinsic() const;
    bool is_accelerator() const;

    /// @name stream
    //@{
    std::ostream& stream_head(std::ostream&) const;
    std::ostream& stream_body(std::ostream&) const;
    void dump_head() const;
    void dump_body() const;
    //@}

    static constexpr auto Node = Node::Lam;
    friend class World;
};

template<class To>
using LamMap  = GIDMap<Lam*, To>;
using LamSet  = GIDSet<Lam*>;
using Lam2Lam = LamMap<Lam*>;

class Param : public Def {
private:
    Param(const Def* type, Def* nominal, const Def* dbg)
        : Def(Node, rebuild, type, Defs{nominal}, 0, dbg)
    {}

public:
    Def* nominal() const { return op(0)->as_nominal(); }
    Lam* lam() const { return nominal()->as<Lam>(); }
    Pi* pi() const { return nominal()->as<Pi>(); }

    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);

    static constexpr auto Node = Node::Param;
    friend class World;
};

template<class To>
using ParamMap    = GIDMap<const Param*, To>;
using ParamSet    = GIDSet<const Param*>;
using Param2Param = ParamMap<const Param*>;

class Tracker {
public:
    Tracker()
        : def_(nullptr)
    {}
    Tracker(const Def* def)
        : def_(def)
    {}

    operator const Def*() { return def(); }
    const Def* operator->() { return def(); }
    const Def* def() {
        if (def_ != nullptr) {
            while (auto repr = def_->substitute_)
                def_ = repr;
        }
        return def_;
    }

private:
    const Def* def_;
};

class Sigma : public Def {
private:
    /// Constructor for a @em structural Pi.
    Sigma(const Def* type, Defs ops, const Def* dbg)
        : Def(Node, rebuild, type, ops, 0, dbg)
    {}
    /// Constructor for a @em nominal Pi.
    Sigma(const Def* type, size_t size, const Def* dbg)
        : Def(Node, stub, type, size, 0, dbg)
    {}

public:
    const Def* arity() const override;
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    static Def* stub(const Def*, World&, const Def*, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Sigma;
    friend class World;
};

/// Data constructor for a @p Sigma.
class Tuple : public Def {
private:
    Tuple(const Def* type, Defs args, const Def* dbg)
        : Def(Node, rebuild, type, args, 0, dbg)
    {}

public:
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Tuple;
    friend class World;
};

class Variadic : public Def {
private:
    Variadic(const Def* type, const Def* arity, const Def* body, const Def* dbg)
        : Def(Node, rebuild, type, {arity, body}, 0, dbg)
    {}

public:
    const Def* arity() const override { return op(0); }
    const Def* body() const { return op(1); }
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Variadic;
    friend class World;
};

class Pack : public Def {
private:
    Pack(const Def* type, const Def* body, const Def* dbg)
        : Def(Node, rebuild, type, {body}, 0, dbg)
    {}

public:
    const Def* body() const { return op(0); }
    std::ostream& stream(std::ostream&) const override;

    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);

    static constexpr auto Node = Node::Pack;
    friend class World;
};

/// Extracts from aggregate <tt>agg</tt> the element at position <tt>index</tt>.
class Extract : public Def {
private:
    Extract(const Def* type, const Def* agg, const Def* index, const Def* dbg)
        : Def(Node, rebuild, type, {agg, index}, 0, dbg)
    {}

public:
    const Def* agg() const { return op(0); }
    const Def* index() const { return op(1); }

    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);

    static constexpr auto Node = Node::Extract;
    friend class World;
};

/**
 * Creates a new aggregate by inserting <tt>val</tt> at position <tt>index</tt> into <tt>agg</tt>.
 * @attention { This is a @em functional insert.
 *              The val <tt>agg</tt> remains untouched.
 *              The \p Insert itself is a \em new aggregate which contains the newly created <tt>val</tt>. }
 */
class Insert : public Def {
private:
    Insert(const Def* agg, const Def* index, const Def* val, const Def* dbg)
        : Def(Node, rebuild, agg->type(), {agg, index, val}, 0, dbg)
    {}

    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);

public:
    const Def* agg() const { return op(0); }
    const Def* index() const { return op(1); }
    const Def* val() const { return op(2); }

    static constexpr auto Node = Node::Insert;
    friend class World;
};

/// The type of a variant (structurally typed).
class VariantType : public Def {
private:
    VariantType(const Def* type, Defs ops, const Def* dbg)
        : Def(Node, rebuild, type, ops, 0, dbg)
    {
        assert(std::adjacent_find(ops.begin(), ops.end()) == ops.end());
    }

public:
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::VariantType;
    friend class World;
};

/// The type of values that models side effects.
class Mem : public Def {
private:
    Mem(World& world);

public:
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Mem;
    friend class World;
};

class Nat : public Def {
private:
    Nat(World& world);

public:
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);
    std::ostream& stream(std::ostream&) const override;

    static constexpr auto Node = Node::Nat;
    friend class World;
};

struct AddrSpace {
    enum : u64 {
        Generic  = 0,
        Global   = 1,
        Texture  = 2,
        Shared   = 3,
        Constant = 4,
    };
};

/// Pointer type.
class Ptr : public Def {
private:
    Ptr(const Def* type, const Def* pointee, const Def* addr_space, const Def* dbg)
        : Def(Node, rebuild, type, {pointee, addr_space}, 0, dbg)
    {}

public:
    const Def* pointee() const { return op(0); }
    const Def* addr_space() const { return op(1); }
    u64 lit_addr_space() const { return as_lit<u64>(op(1)); }

    std::ostream& stream(std::ostream&) const override;
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);

    static constexpr auto Node = Node::Ptr;
    friend class World;
};

class Analyze : public Def {
private:
    Analyze(const Def* type, Defs ops, const Def* dbg)
        : Def(Node, rebuild, type, ops, 0, dbg)
    {}

public:
    u64 index() const { return as_lit<u64>(op(0)); }
    std::ostream& stream(std::ostream&) const override;
    static const Def* rebuild(const Def*, World&, const Def*, Defs, const Def*);

    static constexpr auto Node = Node::Analyze;
    friend class World;
};

u32 UseHash::hash(Use use) { return hash_combine(hash_begin(uint16_t(use.index())), u32(use->gid())); }

namespace detail {
    inline std::ostream& stream(std::ostream& os, const Def* def) { return def->stream(os); }
}

inline std::ostream& operator<<(std::ostream& os, const Def* def) { return def == nullptr ? os << "nullptr" : def->stream(os); }
inline std::ostream& operator<<(std::ostream& os, Use use) { return use->stream(os); }

//------------------------------------------------------------------------------

}

#endif
