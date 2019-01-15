#ifndef THORIN_DEF_H
#define THORIN_DEF_H

#include <string>
#include <vector>

#include "thorin/enums.h"
#include "thorin/util/array.h"
#include "thorin/util/hash.h"
#include "thorin/util/location.h"

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
class PrimOp;
class Tracker;
class Use;
class World;

typedef ArrayRef<const Def*> Defs;
typedef std::vector<Lam*> Lams;

//------------------------------------------------------------------------------

/**
 * References a user.
 * A \p Def \c u which uses \p Def \c d as \c i^th operand is a \p Use with \p index_ \c i of \p Def \c d.
 */
class Use {
public:
    Use() {}
#if defined(__x86_64__) || (_M_X64)
    Use(size_t index, const Def* def)
        : uptr_(reinterpret_cast<uintptr_t>(def) | (uintptr_t(index) << 48ull))
    {}

    size_t index() const { return uptr_ >> 48ull; }
    const Def* def() const {
        // sign extend to make pointer canonical
        return reinterpret_cast<const Def*>((iptr_  << 16) >> 16) ;
    }
#else
    Use(size_t index, const Def* def)
        : index_(index)
        , def_(def)
    {}

    size_t index() const { return index_; }
    const Def* def() const { return def_; }
#endif
    operator const Def*() const { return def(); }
    const Def* operator->() const { return def(); }
    bool operator==(Use other) const { return this->def() == other.def() && this->index() == other.index(); }

private:
#if defined(__x86_64__) || (_M_X64)
    /// A tagged pointer: First 16bit is index, remaining 48bit is the actual pointer.
    union {
        uintptr_t uptr_;
        intptr_t iptr_;
    };
#else
    size_t index_;
    const Def* def_;
#endif
};

//------------------------------------------------------------------------------

struct UseHash {
    inline static uint64_t hash(Use use);
    inline static bool eq(Use u1, Use u2) { return u1 == u2; }
    inline static Use sentinel() { return Use(size_t(-1), (const Def*)(-1)); }
};

// using a StackCapacity of 8 covers almost 99% of all real-world use-lists
typedef HashSet<Use, UseHash> Uses;

template<class To>
using DefMap  = GIDMap<const Def*, To>;
using DefSet  = GIDSet<const Def*>;
using Def2Def = DefMap<const Def*>;

std::ostream& operator<<(std::ostream&, const Def*);
std::ostream& operator<<(std::ostream&, Use);

//------------------------------------------------------------------------------

/**
 * The base class for all three tags of Definitions in AnyDSL.
 * These are:
 * - \p PrimOp%s
 * - \p Param%s and
 * - \p Lam%s.
 */
class Def : public RuntimeCast<Def>, public Streamable {
private:
    Def& operator=(const Def&) = delete;
    Def(const Def&) = delete;

protected:
    /// Constructor for a @em structural Def.
    Def(NodeTag tag, const Def* type, Defs ops, Debug dbg)
        : tag_(tag)
        , ops_(ops.size())
        , type_(type)
        , debug_(dbg)
        , gid_(gid_counter_++)
        , nominal_(false)
        , contains_lam_(false)
    {
        for (size_t i = 0, e = ops.size(); i != e; ++i)
            set_op(i, ops[i]);
    }
    /// Constructor for a @em nominal Def.
    Def(NodeTag tag, const Def* type, size_t size, Debug dbg)
        : tag_(tag)
        , ops_(size)
        , type_(type)
        , debug_(dbg)
        , gid_(gid_counter_++)
        , nominal_(true)
        , contains_lam_(false)
    {}
    virtual ~Def() {}

    void clear_type() { type_ = nullptr; }
    void set_type(const Def* type) { type_ = type; }
    void unregister_use(size_t i) const;
    void unregister_uses() const;

public:
    NodeTag tag() const { return tag_; }
    /// In Debug build if World::enable_history is true, this thing keeps the gid to track a history of gid%s.
    Debug debug_history() const;
    Debug& debug() const { return debug_; }
    Location location() const { return debug_; }
    Symbol name() const { return debug().name(); }
    size_t num_ops() const { return ops_.size(); }
    void set_op(size_t i, const Def* def);
    void unset_op(size_t i);
    void update_op(size_t i, const Def* def) { unset_op(i); set_op(i, def); }
    bool contains_lam() const { return contains_lam_; }
    bool is_nominal() const { return nominal_; }
    Lam* as_lam() const;
    Lam* isa_lam() const;
    void dump() const;
    const Uses& uses() const { return uses_; }
    Array<Use> copy_uses() const { return Array<Use>(uses_.begin(), uses_.end()); }
    size_t num_uses() const { return uses().size(); }
    size_t gid() const { return gid_; }
    std::string unique_name() const;
    World& world() const {
        if (tag()                 == Node_Universe) return *world_;
        if (type()->tag()         == Node_Universe) return *type()->world_;
        if (type()->type()->tag() == Node_Universe) return *type()->type()->world_;
        assert(type()->type()->type()->tag() == Node_Universe);
        return *type()->type()->type()->world_;
    }
    const Def* type() const { assert(tag() != Node_Universe); return type_; }
    int order() const { return type()->order(); }
    Defs ops() const { return ops_; }
    const Def* op(size_t i) const { assert(i < ops().size() && "index out of bounds"); return ops_[i]; }
    void replace(Tracker) const;
    bool is_replaced() const { return substitute_ != nullptr; }

    //@{ rebuild/stub
    virtual const Def* vrebuild(World&, const Def*, Defs) const = 0;
    const Def* rebuild(const Def* type, Defs ops) const { return vrebuild(world(), type, ops); }
    const Def* rebuild(Defs ops) const { return vrebuild(world(), type(), ops); }
    virtual Def* vstub(World&, const Def*) const { THORIN_UNREACHABLE; }
    Def* stub(const Def* type) const { return vstub(world(), type); }
    Def* stub() const { return vstub(world(), type()); }
    //@}

    virtual uint64_t vhash() const;
    virtual bool equal(const Def* other) const;
    virtual const char* op_name() const;
    virtual std::ostream& stream(std::ostream&) const;
    virtual std::ostream& stream_assignment(std::ostream&) const;
    static size_t gid_counter() { return gid_counter_; }

private:
    const NodeTag tag_;
    Array<const Def*> ops_;
    union {
        const Def* type_;
        mutable World* world_;
    };
    mutable const Def* substitute_ = nullptr;
    mutable Uses uses_;
    mutable Debug debug_;
    const size_t gid_ : 32;
protected:
    unsigned nominal_ : 1;
private:

    static size_t gid_counter_;

protected:
    unsigned contains_lam_ : 1;

private:
    uint64_t hash() const { return hash_ == 0 ? hash_ = vhash() : hash_; }

    mutable uint64_t hash_ = 0;

    friend struct DefHash;
    friend class Cleaner;
    friend class PrimOp;
    friend class Scope;
    friend class Tracker;
    friend class World;
};

class Bottom : public Def {
public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    Bottom(const Def* type, Debug dbg)
        : Def(Node_Bottom, type, {}, dbg)
    {}

    virtual const Def* vrebuild(World& to, const Def*, Defs ops) const override;

    friend class World;
};

class Top : public Def {
public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    Top(const Def* type, Debug dbg)
        : Def(Node_Top, type, {}, dbg)
    {}

    virtual const Def* vrebuild(World& to, const Def*, Defs ops) const override;

    friend class World;
};

class Kind : public Def {
private:
    Kind(World& world, NodeTag);

public:
    const Def* vrebuild(World&, const Def*, Defs) const override;

    friend class World;
};

class Var : public Def {
private:
    Var(const Def* type, u64 index, Debug dbg)
        : Def(Node_Var, type, Defs(), dbg)
        , index_(index)
    {}

public:
    u64 index() const { return index_; }
    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual uint64_t vhash() const override;
    virtual bool equal(const Def*) const override;
    virtual const Def* vrebuild(World&, const Def*, Defs) const;

    u64 index_;

    friend class World;
};

class Pi : public Def {
protected:
    Pi(const Def* type, const Def* domain, const Def* codomain, Debug dbg)
        : Def(Node_Pi, type, {domain, codomain}, dbg)
    {
        //++order_;
    }

public:
    const Def* domain() const { return op(0); }
    const Def* codomain() const { return op(1); }
    const Def* is_cn() const { return codomain()->isa<Bottom>(); }

    size_t num_domains() const;
    Array<const Def*> domains() const;
    const Def* domain(size_t i) const;

    bool is_basicblock() const { return order() == 1; }
    bool is_returning() const;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Def* vrebuild(World& to, const Def*, Defs ops) const override;

    friend class World;
};

class App : public Def {
private:
    App(const Def* type, const Def* callee, const Def* arg, Debug dbg)
        : Def(Node_App, type, {callee, arg}, dbg)
    {}

public:
    const Def* callee() const { return op(0); }
    const Def* arg() const { return op(1); }

    size_t num_args() const;
    const Def* arg(size_t i) const;
    Array<const Def*> args() const;

    const Def* vrebuild(World&, const Def*, Defs) const override;

    friend class World;
};

template<class To>
using AppMap  = GIDMap<const App*, To>;
using AppSet  = GIDSet<const App*>;
using App2App = AppMap<const App*>;

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
    Branch,                     ///< branch(cond, T, F).
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
    Lam(const Pi* pi, CC cc, Intrinsic intrinsic, Debug dbg);

public:
    //@{ operands
    const Def* filter() const { return op(0); }
    const Def* filter(size_t i) const;
    const Def* body() const { return op(1); }
    const App* app() const { return body()->isa<App>(); }
    //@}

    //@{ params
    const Param* param(Debug dbg = {}) const;
    size_t num_params() const;
    const Def* param(size_t i, Debug dbg = {}) const;
    Array<const Def*> params() const;
    const Def* mem_param() const;
    const Def* ret_param() const;
    //@}

    //@{ setters
    void set_filter(const Def* filter) { update_op(0, filter); }
    void set_filter(Defs filter);
    void set_all_true_filter();
    void set_body(const Def* body) { update_op(1, body); }
    void destroy_filter();
    //@}

    //@{ type
    const Pi* type() const { return Def::type()->as<Pi>(); }
    const Def* domain() const { return type()->domain(); }
    const Def* codomain() const { return type()->codomain(); }
    //@}

    Def* vstub(World&, const Def*) const override;
    const Def* vrebuild(World&, const Def*, Defs) const override;

    Lams preds() const;
    Lams succs() const;
    bool is_empty() const;
    Intrinsic& intrinsic() { return intrinsic_; }
    Intrinsic intrinsic() const { return intrinsic_; }
    CC& cc() { return cc_; }
    CC cc() const { return cc_; }
    void set_intrinsic(); ///< Sets @p intrinsic_ derived on this @p Lam's @p name.
    bool is_external() const;
    void make_external();
    void make_internal();
    bool is_basicblock() const;
    bool is_returning() const;
    bool is_intrinsic() const;
    bool is_accelerator() const;
    void destroy_body();

    std::ostream& stream_head(std::ostream&) const;
    std::ostream& stream_body(std::ostream&) const;
    void dump_head() const;
    void dump_body() const;

    // terminate

    void app(const Def* callee, const Def* arg, Debug dbg = {});
    void app(const Def* callee, Defs args, Debug dbg = {});
    void branch(const Def* cond, const Def* t, const Def* f, Debug dbg = {});
    void match(const Def* val, Lam* otherwise, Defs patterns, ArrayRef<Lam*> lams, Debug dbg = {});

private:
    CC cc_;
    Intrinsic intrinsic_;

    friend class Cleaner;
    friend class Scope;
    friend class CFA;
    friend class World;
};

template<class To>
using LamMap  = GIDMap<Lam*, To>;
using LamSet  = GIDSet<Lam*>;
using Lam2Lam = LamMap<Lam*>;

class Peek {
public:
    Peek() {}
    Peek(const Def* def, Lam* from)
        : def_(def)
        , from_(from)
    {}

    const Def* def() const { return def_; }
    Lam* from() const { return from_; }

private:
    const Def* def_;
    Lam* from_;
};

size_t get_param_index(const Def* def);
Lam* get_param_lam(const Def* def);
std::vector<Peek> peek(const Def*);

class Param : public Def {
private:
    Param(const Def* type, const Lam* lam, Debug dbg)
        : Def(Node_Param, type, Defs{lam}, dbg)
    {
        assert(lam->is_nominal());
    }

public:
    Lam* lam() const { return op(0)->as_lam(); }
    const Def* vrebuild(World&, const Def*, Defs) const override;

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
    {
        nominal_ = true;
    }

public:
    void set(size_t i, const Def* type) const { assert(is_nominal()); const_cast<Sigma*>(this)->Def::set_op(i, type); }

    virtual const Def* vrebuild(World& to, const Def*, Defs ops) const override;
    virtual std::ostream& stream(std::ostream&) const override;

    friend class World;
};

const Def* merge_sigma(const Def*, const Def*);


/// The type of a variant (structurally typed).
class VariantType : public Def {
private:
    VariantType(const Def* type, Defs ops, Debug dbg)
        : Def(Node_VariantType, type, ops, dbg)
    {
        assert(std::adjacent_find(ops.begin(), ops.end()) == ops.end());
    }

private:
    virtual const Def* vrebuild(World& to, const Def*, Defs ops) const override;
    virtual std::ostream& stream(std::ostream&) const override;

    friend class World;
};

/// The type of the memory monad.
class MemType : public Def {
public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    MemType(const Def* type)
        : Def(Node_MemType, type, Defs{}, {"mem"})
    {}

    virtual const Def* vrebuild(World& to, const Def* type, Defs ops) const override;

    friend class World;
};

/// The type of a stack frame.
class FrameType : public Def {
public:
    virtual std::ostream& stream(std::ostream&) const override;

private:
    FrameType(const Def* type)
        : Def(Node_FrameType, type, Defs{}, {"frame"})
    {}

    virtual const Def* vrebuild(World& to, const Def*, Defs ops) const override;

    friend class World;
};

/// Base class for all SIMD types.
class VectorType : public Def {
protected:
    VectorType(int tag, const Def* type, Defs ops, size_t length, Debug dbg)
        : Def((NodeTag)tag, type, ops, dbg)
        , length_(length)
    {}

    virtual uint64_t vhash() const override { return hash_combine(Def::vhash(), length()); }
    virtual bool equal(const Def* other) const override {
        return Def::equal(other) && this->length() == other->as<VectorType>()->length();
    }

public:
    /// The number of vector arguments - the vector length.
    size_t length() const { return length_; }
    bool is_vector() const { return length_ != 1; }
    /// Rebuilds the type with vector length 1.
    const VectorType* scalarize() const;

private:
    size_t length_;
};

/// Primitive type.
class PrimType : public VectorType {
private:
    PrimType(PrimTypeTag tag, const Def* type, size_t length, Debug dbg)
        : VectorType((int) tag, type, Defs{}, length, dbg)
    {}

public:
    PrimTypeTag primtype_tag() const { return (PrimTypeTag) tag(); }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Def* vrebuild(World& to, const Def*, Defs ops) const override;

    friend class World;
};

inline bool is_primtype (const Def* t) { return thorin::is_primtype(t->tag()); }
inline bool is_type_ps  (const Def* t) { return thorin::is_type_ps (t->tag()); }
inline bool is_type_pu  (const Def* t) { return thorin::is_type_pu (t->tag()); }
inline bool is_type_qs  (const Def* t) { return thorin::is_type_qs (t->tag()); }
inline bool is_type_qu  (const Def* t) { return thorin::is_type_qu (t->tag()); }
inline bool is_type_pf  (const Def* t) { return thorin::is_type_pf (t->tag()); }
inline bool is_type_qf  (const Def* t) { return thorin::is_type_qf (t->tag()); }
inline bool is_type_p   (const Def* t) { return thorin::is_type_p  (t->tag()); }
inline bool is_type_q   (const Def* t) { return thorin::is_type_q  (t->tag()); }
inline bool is_type_s   (const Def* t) { return thorin::is_type_s  (t->tag()); }
inline bool is_type_u   (const Def* t) { return thorin::is_type_u  (t->tag()); }
inline bool is_type_i   (const Def* t) { return thorin::is_type_i  (t->tag()); }
inline bool is_type_f   (const Def* t) { return thorin::is_type_f  (t->tag()); }
inline bool is_type_bool(const Def* t) { return t->tag() == Node_PrimType_bool; }

enum class AddrSpace : uint32_t {
    Generic  = 0,
    Global   = 1,
    Texture  = 2,
    Shared   = 3,
    Constant = 4,
};

/// Pointer type.
class PtrType : public VectorType {
private:
    PtrType(const Def* type, const Def* pointee, size_t length, int32_t device, AddrSpace addr_space, Debug dbg)
        : VectorType(Node_PtrType, type, {pointee}, length, dbg)
        , addr_space_(addr_space)
        , device_(device)
    {}

public:
    const Def* pointee() const { return op(0); }
    AddrSpace addr_space() const { return addr_space_; }
    int32_t device() const { return device_; }
    bool is_host_device() const { return device_ == -1; }

    virtual uint64_t vhash() const override;
    virtual bool equal(const Def* other) const override;

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Def* vrebuild(World& to, const Def*, Defs ops) const override;

    AddrSpace addr_space_;
    int32_t device_;

    friend class World;
};

class ArrayType : public Def {
protected:
    ArrayType(int tag, const Def* type, const Def* elem_type, Debug dbg)
        : Def((NodeTag)tag, type, {elem_type}, dbg)
    {}

public:
    const Def* elem_type() const { return op(0); }
};

class IndefiniteArrayType : public ArrayType {
public:
    IndefiniteArrayType(const Def* type, const Def* elem_type, Debug dbg)
        : ArrayType(Node_IndefiniteArrayType, type, elem_type, dbg)
    {}

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Def* vrebuild(World& to, const Def*, Defs ops) const override;

    friend class World;
};

class DefiniteArrayType : public ArrayType {
public:
    DefiniteArrayType(const Def* type, const Def* elem_type, u64 dim, Debug dbg)
        : ArrayType(Node_DefiniteArrayType, type, elem_type, dbg)
        , dim_(dim)
    {}

    u64 dim() const { return dim_; }
    virtual uint64_t vhash() const override { return hash_combine(Def::vhash(), dim()); }
    virtual bool equal(const Def* other) const override {
        return Def::equal(other) && this->dim() == other->as<DefiniteArrayType>()->dim();
    }

    virtual std::ostream& stream(std::ostream&) const override;

private:
    virtual const Def* vrebuild(World& to, const Def*, Defs ops) const override;

    u64 dim_;

    friend class World;
};

uint64_t UseHash::hash(Use use) { return murmur3(uint64_t(use.index()) << 48_u64 | uint64_t(use->gid())); }

/// Returns the vector length. Raises an assertion if type of this is not a \p VectorType.
size_t vector_length(const Def*);
bool is_unit(const Def*);
bool is_const(const Def*);
bool is_primlit(const Def*, int64_t);
bool is_minus_zero(const Def*);
inline bool is_mem        (const Def* def) { return def->type()->isa<MemType>(); }
inline bool is_zero       (const Def* def) { return is_primlit(def, 0); }
inline bool is_one        (const Def* def) { return is_primlit(def, 1); }
inline bool is_allset     (const Def* def) { return is_primlit(def, -1); }
inline bool is_bitop      (const Def* def) { return thorin::is_bitop(def->tag()); }
inline bool is_shift      (const Def* def) { return thorin::is_shift(def->tag()); }
inline bool is_not        (const Def* def) { return def->tag() == Node_xor && is_allset(def->op(0)); }
inline bool is_minus      (const Def* def) { return def->tag() == Node_sub && is_minus_zero(def->op(0)); }
inline bool is_div_or_rem (const Def* def) { return thorin::is_div_or_rem(def->tag()); }
inline bool is_commutative(const Def* def) { return thorin::is_commutative(def->tag()); }
inline bool is_associative(const Def* def) { return thorin::is_associative(def->tag()); }

namespace detail {
    inline std::ostream& stream(std::ostream& os, const Def* def) { return def->stream(os); }
}

inline std::ostream& operator<<(std::ostream& os, const Def* def) { return def == nullptr ? os << "nullptr" : def->stream(os); }
inline std::ostream& operator<<(std::ostream& os, Use use) { return use->stream(os); }

//------------------------------------------------------------------------------

}

#endif
