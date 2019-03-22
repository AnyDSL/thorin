#ifndef THORIN_DEF_H
#define THORIN_DEF_H

#include <optional>
#include <string>
#include <vector>

#include "thorin/enums.h"
#include "thorin/util/array.h"
#include "thorin/util/hash.h"
#include "thorin/util/debug.h"

namespace thorin {

template<class T>
struct GIDLt {
    bool operator()(T a, T b) const { return a->gid() < b->gid(); }
};

template<class T>
struct GIDHash {
    static uint64_t hash(T n) { return thorin::murmur3(n->gid()); }
    static bool eq(T a, T b) { return a == b; }
    static T sentinel() { return T(1); }
};

template<class Key, class Value>
using GIDMap = thorin::HashMap<Key, Value, GIDHash<Key>>;
template<class Key>
using GIDSet = thorin::HashSet<Key, GIDHash<Key>>;

//------------------------------------------------------------------------------

class Lam;
class Param;
class Def;
class Tracker;
class World;

typedef ArrayRef<const Def*> Defs;
typedef std::vector<Lam*> Lams;

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
    inline static uint64_t hash(Use use);
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
private:
    Def& operator=(const Def&) = delete;
    Def(const Def&) = delete;

protected:
    /// Constructor for a @em structural Def.
    Def(NodeTag tag, const Def* type, Defs ops, Debug dbg);
    /// Constructor for a @em nominal Def.
    Def(NodeTag tag, const Def* type, size_t num_ops, Debug dbg);
    virtual ~Def() {}

public:
    enum class Sort {
        Term, Type, Kind, Universe
    };

    /// @name type, Sort
    //@{
    const Def* type() const { assert(tag() != Node_Universe); return type_; }
    unsigned order() const { assert(!is_term()); return order_; }
    Sort sort() const;
    bool is_term() const { return sort() == Sort::Term; }
    bool is_type() const { return sort() == Sort::Type; }
    bool is_kind() const { return sort() == Sort::Kind; }
    bool is_universe() const { return sort() == Sort::Universe; }
    bool is_value() const { return is_value_; }
    virtual const Def* arity() const;
    //@}
    /// @name ops
    //@{
    inline Defs ops() const { return Defs(num_ops_, ops_ptr()); }
    inline const Def* op(size_t i) const { return ops()[i]; }
    inline size_t num_ops() const { return num_ops_; }
    Def* set(size_t i, const Def* def);
    Def* set(Defs ops) { for (size_t i = 0, e = num_ops(); i != e; ++i) set(i, ops[i]); return this; }
    void unset(size_t i);
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
    /// @name Debug
    //@{
    Debug& debug() const { return debug_; }
    Loc loc() const { return debug_; }
    Symbol name() const { return debug().name(); }
    /// name + "_" + gid
    std::string unique_name() const;
    /// In Debug build if World::enable_history is true, this thing keeps the gid to track a history of gid%s.
    Debug debug_history() const;
    //@}
    /// @name cast for nominals
    //@{
    template<class T = Def> T* as_nominal() const { assert(nominal_ && as<T>()); return static_cast<T*>(const_cast<Def*>(this)); }
    template<class T = Def> T* isa_nominal() const { return dynamic_cast<T*>(nominal_ ? const_cast<Def*>(this) : nullptr); }
    //@}
    /// @name misc getters
    //@{
    NodeTag tag() const { return (NodeTag)tag_; }
    size_t gid() const { return gid_; }
    bool contains_lam() const { return contains_lam_; }
    uint64_t hash() const { return hash_; }
    World& world() const {
        if (tag()                 == Node_Universe) return *world_;
        if (type()->tag()         == Node_Universe) return *type()->world_;
        if (type()->type()->tag() == Node_Universe) return *type()->type()->world_;
        assert(type()->type()->type()->tag() == Node_Universe);
        return *type()->type()->type()->world_;
    }
    //@}
    /// @name replace
    //@{
    void replace(Tracker) const;
    bool is_replaced() const { return substitute_ != nullptr; }
    //@}
    /// @name rebuild, stub, equal
    //@{
    virtual const Def* rebuild(World&, const Def*, Defs) const = 0;
    virtual Def* stub(World&, const Def*) { THORIN_UNREACHABLE; }
    virtual bool equal(const Def* other) const;
    //@}
    /// @name stream
    //@{
    void dump() const;
    virtual const char* op_name() const;
    virtual std::ostream& stream(std::ostream&) const;
    virtual std::ostream& stream_assignment(std::ostream&) const;
    //@}

protected:
    inline char* extra_ptr() { return reinterpret_cast<char*>(this) + sizeof(Def) + sizeof(const Def*)*num_ops(); }
    inline const char* extra_ptr() const { return const_cast<Def*>(this)->extra_ptr(); }
    template<class T> inline T& extra() { return reinterpret_cast<T&>(*extra_ptr()); }
    template<class T> inline const T& extra() const { return reinterpret_cast<const T&>(*extra_ptr()); }
    inline const Def** ops_ptr() const { return reinterpret_cast<const Def**>(reinterpret_cast<char*>(const_cast<Def*>(this + 1))); }
    void finalize();

    struct Extra {};

    union {
        const Def* type_;
        mutable World* world_;
    };
    // TODO fine-tune bit fields
    unsigned tag_           : 10;
    unsigned is_value_      :  1;
    unsigned nominal_       :  1;
    unsigned dependent_     :  1;
    unsigned contains_lam_  :  1;
    unsigned order_         : 10;
    uint32_t gid_;
    uint32_t num_ops_;
    mutable const Def* substitute_ = nullptr;
    mutable Uses uses_;
    mutable Debug debug_;
    uint64_t hash_;

    friend class Cleaner;
    friend class Tracker;
    friend class World;
    friend void swap(World&, World&);
};

class Universe : public Def {
private:
    Universe(World& world)
        : Def(Node_Universe, reinterpret_cast<const Def*>(&world), 0_s, {"□"})
    {}

public:
    const Def* rebuild(World&, const Def*, Defs) const override;
    Universe* stub(World&, const Def*) override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

class Kind : public Def {
private:
    Kind(World&, NodeTag);

public:
    const Def* rebuild(World&, const Def*, Defs) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

typedef const Def* (*Normalizer)(const Def*, const Def*, Debug);

class Axiom : public Def {
private:
    struct Extra { Normalizer normalizer_; };

    Axiom(const Def* type, Normalizer normalizer, Debug dbg)
        : Def(Node_Axiom, type, 0, dbg)
    {
        extra<Extra>().normalizer_ = normalizer;
        //assert(type->free_vars().none());
    }

public:
    Normalizer normalizer() const { return extra<Extra>().normalizer_; }

    const Def* rebuild(World&, const Def*, Defs) const override;
    Axiom* stub(World&, const Def*) override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

class BotTop : public Def {
private:
    BotTop(bool is_top, const Def* type, Debug dbg)
        : Def(is_top ? Node_Top : Node_Bot, type, Defs{}, dbg)
    {}

public:
    const Def* rebuild(World& to, const Def*, Defs ops) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

class Lit : public Def {
private:
    struct Extra { Box box_; };

    Lit(const Def* type, Box box, Debug dbg)
        : Def(Node_Lit, type, Defs{}, dbg)
    {
        extra<Extra>().box_ = box;
        hash_ = hash_combine(hash_, box.get_u64());
    }

public:
    Box box() const { return extra<Extra>().box_; }
    bool equal(const Def*) const override;
    const Def* rebuild(World& to, const Def*, Defs ops) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

template<class T> std::optional<T> isa_lit(const Def* def) {
    if (auto lit = def->isa<Lit>())
        return lit->box().get<T>();
    return {};
}

template<class T> T as_lit(const Def* def) { return def->as<Lit>()->box().get<T>(); }

class Var : public Def {
private:
    struct Extra { u64 index_; };

    Var(const Def* type, u64 index, Debug dbg)
        : Def(Node_Var, type, Defs{}, dbg)
    {
        extra<Extra>().index_ = index;
        hash_ = hash_combine(hash_, index);
    }

public:
    u64 index() const { return extra<Extra>().index_; }

    bool equal(const Def*) const override;
    const Def* rebuild(World&, const Def*, Defs) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

class Pi : public Def {
protected:
    Pi(const Def* type, const Def* domain, const Def* codomain, Debug dbg)
        : Def(Node_Pi, type, {domain, codomain}, dbg)
    {}

public:
    const Def* domain() const { return op(0); }
    const Def* codomain() const { return op(1); }
    bool is_cn() const;

    const Def* domain(size_t i) const;
    Array<const Def*> domains() const;
    size_t num_domains() const;

    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;

    std::ostream& stream(std::ostream&) const override;
    const Def* rebuild(World& to, const Def*, Defs ops) const override;

    friend class World;
};

class App : public Def {
private:
    App(const Def* type, const Def* callee, const Def* arg, Debug dbg);

public:
    const Def* callee() const { return op(0); }
    const Pi* callee_type() const { return callee()->type()->as<Pi>(); }
    const Def* arg() const { return op(1); }
    const Def* arg(size_t i) const;
    Array<const Def*> args() const;
    size_t num_args() const { return as_lit<u64>(callee_type()->domain()->arity()); }

    const Def* rebuild(World&, const Def*, Defs) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

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

enum class CC : uint8_t {
    C,          ///< C calling convention.
    Device,     ///< Device calling convention. These are special functions only available on a particular device.
};

class Lam : public Def {
private:
    struct Extra {
        CC cc_;
        Intrinsic intrinsic_;
    };

    Lam(const Pi* pi, const Def* filter, const Def* body, Debug dbg)
        : Def(Node_Lam, pi, {filter, body}, dbg)
    {
        is_value_ = true;
        extra<Extra>().cc_ = CC::C;
        extra<Extra>().intrinsic_ = Intrinsic::None;
    }
    Lam(const Pi* pi, CC cc, Intrinsic intrinsic, Debug dbg)
        : Def(Node_Lam, pi, 2, dbg)
    {
        is_value_ = true;
        extra<Extra>().cc_ = cc;
        extra<Extra>().intrinsic_ = intrinsic;
    }

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
    const Def* filter(size_t i) const;
    Array<const Def*> filters() const;
    size_t num_filters() const { return num_params(); }
    const Def* body() const { return op(1); }
    const App* app() const { return body()->isa<App>(); }
    //@}
    /// @name params
    //@{
    const Param* param(Debug dbg = {}) const;
    const Def* param(size_t i, Debug dbg = {}) const;
    Array<const Def*> params() const;
    size_t num_params() const { return as_lit<u64>(type()->domain()->arity()); }
    const Def* mem_param() const;
    const Def* ret_param() const;
    //@}
    /// @name setters
    //@{
    void set_filter(const Def* filter) { set(0, filter); }
    void set_filter(Defs filter);
    void set_body(const Def* body) { set(1, body); }
    void destroy();
    //@}
    /// @name setters: sets filter to @c false and sets the body by App-ing
    //@{
    void app(const Def* callee, const Def* arg, Debug dbg = {});
    void app(const Def* callee, Defs args, Debug dbg = {});
    void branch(const Def* cond, const Def* t, const Def* f, const Def* mem, Debug dbg = {});
    void match(const Def* val, Lam* otherwise, Defs patterns, ArrayRef<Lam*> lams, Debug dbg = {});
    //@}
    /// @name rebuild, stub
    //@{
    const Def* rebuild(World&, const Def*, Defs) const override;
    Lam* stub(World&, const Def* type) override;
    //@}

    Lams preds() const;
    Lams succs() const;
    bool is_empty() const;
    Intrinsic& intrinsic() { return extra<Extra>().intrinsic_; }
    Intrinsic intrinsic() const { return extra<Extra>().intrinsic_; }
    CC& cc() { return extra<Extra>().cc_; }
    CC cc() const { return extra<Extra>().cc_; }
    void set_intrinsic(); ///< Sets @p intrinsic_ derived on this @p Lam's @p name.
    bool is_external() const;
    void make_external();
    void make_internal();
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

    friend class Cleaner;
    friend class World;
};

template<class To>
using LamMap  = GIDMap<Lam*, To>;
using LamSet  = GIDSet<Lam*>;
using Lam2Lam = LamMap<Lam*>;

class Param : public Def {
private:
    Param(const Def* type, const Lam* lam, Debug dbg)
        : Def(Node_Param, type, Defs{lam}, dbg)
    {
        assert(lam->isa_nominal<Lam>());
    }

public:
    Lam* lam() const { return op(0)->as_nominal<Lam>(); }

    const Def* rebuild(World&, const Def*, Defs) const override;

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
    Sigma(const Def* type, Defs ops, Debug dbg)
        : Def(Node_Sigma, type, ops, dbg)
    {}
    Sigma(const Def* type, size_t size, Debug dbg)
        : Def(Node_Sigma, type, size, dbg)
    {}

public:
    const Def* arity() const override;
    const Def* rebuild(World& to, const Def*, Defs ops) const override;
    Sigma* stub(World&, const Def*) override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

/// Data constructor for a @p Sigma.
class Tuple : public Def {
private:
    Tuple(const Def* type, Defs args, Debug dbg)
        : Def(Node_Tuple, type, args, dbg)
    {
        is_value_ = true;
    }

public:
    const Def* rebuild(World& to, const Def* type, Defs ops) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

class Variadic : public Def {
private:
    Variadic(const Def* type, const Def* arity, const Def* body, Debug dbg)
        : Def(Node_Variadic, type, {arity, body}, dbg)
    {}

public:
    const Def* arity() const override { return op(0); }
    const Def* body() const { return op(1); }
    const Def* rebuild(World&, const Def*, Defs) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

class Pack : public Def {
private:
    Pack(const Def* type, const Def* body, Debug dbg)
        : Def(Node_Pack, type, {body}, dbg)
    {
        is_value_ = true;
    }

public:
    const Def* body() const { return op(0); }
    const Def* rebuild(World&, const Def*, Defs) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

/// Base class for functional @p Insert and @p Extract.
class AggOp : public Def {
protected:
    AggOp(NodeTag tag, const Def* type, Defs args, Debug dbg)
        : Def(tag, type, args, dbg)
    {}

public:
    const Def* agg() const { return op(0); }
    const Def* index() const { return op(1); }

    friend class World;
};

/// Extracts from aggregate <tt>agg</tt> the element at position <tt>index</tt>.
class Extract : public AggOp {
private:
    Extract(const Def* type, const Def* agg, const Def* index, Debug dbg)
        : AggOp(Node_Extract, type, {agg, index}, dbg)
    {}

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

    friend class World;
};

/**
 * Creates a new aggregate by inserting <tt>val</tt> at position <tt>index</tt> into <tt>agg</tt>.
 * @attention { This is a @em functional insert.
 *              The val <tt>agg</tt> remains untouched.
 *              The \p Insert itself is a \em new aggregate which contains the newly created <tt>val</tt>. }
 */
class Insert : public AggOp {
private:
    Insert(const Def* agg, const Def* index, const Def* val, Debug dbg)
        : AggOp(Node_Insert, agg->type(), {agg, index, val}, dbg)
    {}

    const Def* rebuild(World& to, const Def* type, Defs ops) const override;

public:
    const Def* val() const { return op(2); }

    friend class World;
};

/// The type of a variant (structurally typed).
class VariantType : public Def {
private:
    VariantType(const Def* type, Defs ops, Debug dbg)
        : Def(Node_VariantType, type, ops, dbg)
    {
        assert(std::adjacent_find(ops.begin(), ops.end()) == ops.end());
    }

private:
    const Def* rebuild(World& to, const Def*, Defs ops) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

/// The type of the memory monad.
class MemType : public Def {
private:
    MemType(World& world);

public:
    const Def* rebuild(World& to, const Def* type, Defs ops) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

/// The type of a stack frame.
class FrameType : public Def {
private:
    FrameType(World& world);

public:
    const Def* rebuild(World& to, const Def*, Defs ops) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

/// Primitive type.
class PrimType : public Def {
private:
    PrimType(World& world, PrimTypeTag tag, Debug dbg);

public:
    PrimTypeTag primtype_tag() const { return (PrimTypeTag) tag(); }

    const Def* rebuild(World& to, const Def*, Defs ops) const override;
    std::ostream& stream(std::ostream&) const override;

    friend class World;
};

enum class AddrSpace : uint8_t {
    Generic  = 0,
    Global   = 1,
    Texture  = 2,
    Shared   = 3,
    Constant = 4,
};

/// Pointer type.
class PtrType : public Def {
private:
    struct Extra { AddrSpace addr_space_; }; // TODO make this a proper op

    PtrType(const Def* type, const Def* pointee, AddrSpace addr_space, Debug dbg)
        : Def(Node_PtrType, type, {pointee}, dbg)
    {
        extra<Extra>().addr_space_ = addr_space;
        hash_ = hash_combine(hash_, (uint8_t)addr_space);
    }

public:
    const Def* pointee() const { return op(0); }
    AddrSpace addr_space() const { return extra<Extra>().addr_space_; }

    bool equal(const Def* other) const override;
    std::ostream& stream(std::ostream&) const override;
    const Def* rebuild(World& to, const Def*, Defs ops) const override;

    friend class World;
};

class Analyze : public Def {
private:
    struct Extra { u64 index_; };

    Analyze(const Def* type, Defs ops, u64 index, Debug dbg)
        : Def(Node_Analyze, type, ops, dbg)
    {
        extra<Extra>().index_ = index;
        hash_ = hash_combine(hash_, index);
    }

public:
    u64 index() const { return extra<Extra>().index_; }

    bool equal(const Def* other) const override;
    std::ostream& stream(std::ostream&) const override;
    const Def* rebuild(World& to, const Def*, Defs ops) const override;

    friend class World;
};

uint64_t UseHash::hash(Use use) { return murmur3(uint64_t(use.index()) << 48_u64 | uint64_t(use->gid())); }

namespace detail {
    inline std::ostream& stream(std::ostream& os, const Def* def) { return def->stream(os); }
}

inline std::ostream& operator<<(std::ostream& os, const Def* def) { return def == nullptr ? os << "nullptr" : def->stream(os); }
inline std::ostream& operator<<(std::ostream& os, Use use) { return use->stream(os); }

//------------------------------------------------------------------------------

}

#endif
