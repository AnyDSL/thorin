#ifndef THORIN_DEF_H
#define THORIN_DEF_H

#include <string>
#include <vector>

#include "thorin/enums.h"
#include "thorin/type.h"
#include "thorin/debug.h"

namespace thorin {

//------------------------------------------------------------------------------

class Continuation;
class Def;
class Tracker;
class Use;
class World;

typedef ArrayRef<const Def*> Defs;

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

//------------------------------------------------------------------------------

namespace Dep {
enum : unsigned {
    Bot,
    Cont,
    Param,
    Top = Cont | Param
};
}

/**
 * The base class for all three tags of Definitions in AnyDSL.
 * These are:
 * - \p PrimOp%s
 * - \p Param%s and
 * - \p Continuation%s.
 */
class Def : public RuntimeCast<Def>, public Streamable<Def> {
private:
    Def& operator=(const Def&) = delete;
    Def(const Def&) = delete;

protected:
    /// Constructor for a @em structural Def.
    Def(NodeTag tag, const Type* type, Defs args, Debug dbg);
    /// Constructor for a @em nom Def.
    Def(NodeTag tag, const Type* type, size_t size, Debug);
    virtual ~Def() {}

    void clear_type() { type_ = nullptr; }
    void set_type(const Type* type) { type_ = type; }
    void unregister_use(size_t i) const;
    void unregister_uses() const;
    void resize(size_t n) { ops_.resize(n, nullptr); }

public:
    /// @name getters
    //@{
    NodeTag tag() const { return tag_; }
    size_t gid() const { return gid_; }
    World& world() const;
    //@}

    /// @name ops
    //@{
    Defs ops() const { return ops_; }
    Array<const Def*> copy_ops() const { return Array<const Def*>(ops_.begin(), ops_.end()); }
    const Def* op(size_t i) const { assert(i < ops().size() && "index out of bounds"); return ops_[i]; }
    size_t num_ops() const { return ops_.size(); }
    /// Is @p def the @p i^th result of a @p T @p PrimOp?
    template<int i, class T> inline static const T* is_out(const Def* def);
    //@}

    /// @name out
    //@{
    const Def* out(size_t i) const;
    bool empty() const { return ops_.empty(); }
    void set_op(size_t i, const Def* def);
    void unset_op(size_t i);
    void unset_ops();
    virtual bool has_multiple_outs() const { return false; }
    //@}

    /// @name uses
    //@{
    const Uses& uses() const { return uses_; }
    Array<Use> copy_uses() const { return Array<Use>(uses_.begin(), uses_.end()); }
    size_t num_uses() const { return uses().size(); }
    //@}

    /// @name type
    //@{
    const Type* type() const { return type_; }
    int order() const { return type()->order(); }
    //@}

    /// @name dependence checks
    //@{
    unsigned dep() const { return dep_; }
    bool no_dep() const { return dep() == Dep::Bot; }
    bool has_dep(unsigned dep) const { return (dep_ & dep) != 0; }
    //@}

    /// @name Debug
    //@{
    Debug debug() const { return debug_; }
    /// In Debug build if @c World::enable_history is @c true, this thing keeps the @p gid to track a history of @p gid%s.
    Debug debug_history() const;
    std::string name() const { return debug().name; }
    Loc loc() const { return debug().loc; }
    void set_name(const std::string&) const;
    std::string unique_name() const;
    //@}

    /// @name casts
    //@{
    /// If @c this is @em nom, it will cast constness away and perform a dynamic cast to @p T.
    template<class T = Def, bool invert = false> T* isa_nom() const {
        if constexpr(std::is_same<T, Def>::value)
            return nom_ ^ invert ? const_cast<Def*>(this) : nullptr;
        else
            return nom_ ^ invert ? const_cast<Def*>(this)->template isa<T>() : nullptr;
    }

    /// Asserts that @c this is a @em nom, casts constness away and performs a static cast to @p T (checked in Debug build).
    template<class T = Def, bool invert = false> T* as_nom() const {
        assert(nom_ ^ invert);
        if constexpr(std::is_same<T, Def>::value)
            return const_cast<Def*>(this);
        else
            return const_cast<Def*>(this)->template as<T>();
    }

    template<class T = Def> const T* isa_structural() const { return isa_nom<T, true>(); }
    template<class T = Def> const T* as_structural() const { return as_nom<T, true>(); }
    //@}

    /// @name rebuild/stub
    //@{
    virtual const Def* rebuild(World&, const Type*, Defs) const { THORIN_UNREACHABLE; }
    // TODO stub
    void replace_uses(Tracker) const;
    void replace(Tracker) const;                                ///< @deprecated
    bool is_replaced() const { return substitute_ != nullptr; } ///< @deprecated
    //@}

    /// @name hash/equal
    //@{
    virtual hash_t vhash() const;
    virtual bool equal(const Def*) const;
    hash_t hash() const { return hash_ == 0 ? hash_ = vhash() : hash_; }
    //@}

    /// @name stream
    //@{
    Stream& stream(Stream&) const;
    Stream& stream1(Stream&) const;
    Stream& stream_let(Stream&) const;
    Stream& stream(Stream&, size_t max) const;
    virtual const char* op_name() const;
    void dump() const;
    void dump(size_t max) const;
    //@}

    static size_t gid_counter() { return gid_counter_; } // TODO move to World

private:
    const NodeTag tag_;
    std::vector<const Def*> ops_;
    const Type* type_;
    mutable const Def* substitute_ = nullptr;
    mutable Uses uses_;
    mutable Debug debug_;
    mutable hash_t hash_ = 0; // TODO init in ctor
    const uint32_t gid_;
    unsigned nom_ : 1;
    unsigned dep_ : 2;

    static size_t gid_counter_;

    friend class Cleaner;
    friend class Scope;
    friend class Tracker;
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

private:
    mutable const Def* def_;
};

uint64_t UseHash::hash(Use use) {
    assert(use->gid() != uint32_t(-1));
    return murmur3(uint64_t(use.index()) << 48_u64 | uint64_t(use->gid()));
}

/// Returns the vector length. Raises an assertion if type of this is not a \p VectorType.
size_t vector_length(const Def*);
bool is_unit(const Def*);
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

//------------------------------------------------------------------------------

}

#endif
