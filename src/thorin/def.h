#ifndef THORIN_DEF_H
#define THORIN_DEF_H

#include <string>
#include <vector>

#include "thorin/enums.h"
#include "thorin/type.h"
#include "thorin/util/location.h"

namespace thorin {

//------------------------------------------------------------------------------

class Continuation;
class Def;
class PrimOp;
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

std::ostream& operator<<(std::ostream&, const Def*);
std::ostream& operator<<(std::ostream&, Use);

//------------------------------------------------------------------------------

/**
 * The base class for all three tags of Definitions in AnyDSL.
 * These are:
 * - \p PrimOp%s
 * - \p Param%s and
 * - \p Continuation%s.
 */
class Def : public RuntimeCast<Def>, public Streamable {
private:
    Def& operator=(const Def&) = delete;
    Def(const Def&) = delete;

protected:
    Def(NodeTag tag, const Type* type, size_t size, Debug);
    virtual ~Def() {}

    void clear_type() { type_ = nullptr; }
    void set_type(const Type* type) { type_ = type; }
    void unregister_use(size_t i) const;
    void unregister_uses() const;
    void resize(size_t n) { ops_.resize(n, nullptr); }

public:
    NodeTag tag() const { return tag_; }
    /// In Debug build if World::enable_history is true, this thing keeps the gid to track a history of gid%s.
    Debug debug_history() const;
    Debug& debug() const { return debug_; }
    Location location() const { return debug_; }
    const std::string& name() const { return debug().name(); }
    size_t num_ops() const { return ops_.size(); }
    bool empty() const { return ops_.empty(); }
    void set_op(size_t i, const Def* def);
    void unset_op(size_t i);
    void unset_ops();
    bool contains_continuation() const { return contains_continuation_; }
    Continuation* as_continuation() const;
    Continuation* isa_continuation() const;
    void dump() const;
    const Uses& uses() const { return uses_; }
    Array<Use> copy_uses() const { return Array<Use>(uses_.begin(), uses_.end()); }
    size_t num_uses() const { return uses().size(); }
    size_t gid() const { return gid_; }
    std::string unique_name() const;
    const Type* type() const { return type_; }
    int order() const { return type()->order(); }
    World& world() const;
    Defs ops() const { return ops_; }
    const Def* op(size_t i) const { assert(i < ops().size() && "index out of bounds"); return ops_[i]; }
    void replace(Tracker) const;
    bool is_replaced() const { return substitute_ != nullptr; }

    virtual std::ostream& stream(std::ostream&) const;
    static size_t gid_counter() { return gid_counter_; }

private:
    const NodeTag tag_;
    std::vector<const Def*> ops_;
    const Type* type_;
    mutable const Def* substitute_ = nullptr;
    mutable Uses uses_;
    mutable Debug debug_;
    const size_t gid_ : sizeof(size_t) * 8 - 1;

    static size_t gid_counter_;
    
protected:
    bool contains_continuation_;

    friend class Cleaner;
    friend class PrimOp;
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
    inline std::ostream& stream(std::ostream& os, const Type* type) { return type->stream(os); }
}

inline std::ostream& operator<<(std::ostream& os, const Def* def) { return def->stream(os); }
inline std::ostream& operator<<(std::ostream& os, const Type* type) { return type->stream(os); }
inline std::ostream& operator<<(std::ostream& os, Use use) { return use->stream(os); }

//------------------------------------------------------------------------------

}

#endif
