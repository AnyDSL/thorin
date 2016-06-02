#ifndef THORIN_DEF_H
#define THORIN_DEF_H

#include <set>
#include <string>
#include <vector>

#include "thorin/enums.h"
#include "thorin/type.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/location.h"

namespace thorin {

//------------------------------------------------------------------------------

class Def;
class Tracker;
class Continuation;
class PrimOp;
class Use;
class World;

typedef ArrayRef<const Def*> Defs;

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
    operator const Def*() const { return def_; }
    const Def* operator->() const { return def_; }
    bool operator==(Use other) const { return this->def() == other.def() && this->index() == other.index(); }

private:
    size_t index_;
    const Def* def_;
};

//------------------------------------------------------------------------------

struct UseHash {
    inline uint64_t operator()(Use use) const;
};

typedef HashSet<Use, UseHash> Uses;

template<class To>
using DefMap  = GIDMap<Def, To>;
using DefSet  = GIDSet<Def>;
using Def2Def = DefMap<const Def*>;

std::ostream& operator<<(std::ostream&, const Def*);
std::ostream& operator<<(std::ostream&, Use);

//------------------------------------------------------------------------------

/**
 * The base class for all three kinds of Definitions in AnyDSL.
 * These are:
 * - \p PrimOp%s
 * - \p Param%s and
 * - \p Continuation%s.
 */
class Def : public HasLocation, public MagicCast<Def>, public Streamable {
private:
    Def& operator=(const Def&); ///< Do not copy-assign a \p Def instance.
    Def(const Def&);              ///< Do not copy-construct a \p Def.

protected:
    Def(NodeKind kind, const Type* type, size_t size, const Location& loc, const std::string& name);
    virtual ~Def() {}

    void clear_type() { type_ = nullptr; }
    void set_type(const Type* type) { assert(type->is_closed()); type_ = type; }
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
    const Def* is_mem() const { return type()->isa<MemType>() ? this : nullptr; }
    Continuation* as_continuation() const;
    Continuation* isa_continuation() const;
    bool is_const() const;
    void dump() const;
    const Uses& uses() const { return uses_; }
    size_t num_uses() const { return uses().size(); }
    size_t gid() const { return gid_; }
    std::string unique_name() const;
    const Type* type() const { return type_; }
    int order() const;
    World& world() const;
    Defs ops() const { return ops_; }
    const Def* op(size_t i) const { assert(i < ops().size() && "index out of bounds"); return ops_[i]; }
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

    virtual bool is_outdated() const { return false; }
    virtual const Def* rebuild(Def2Def&) const { return this; }
    virtual std::ostream& stream(std::ostream&) const;
    static size_t gid_counter() { return gid_counter_; }

private:
    const NodeKind kind_;
    std::vector<const Def*> ops_;
    const Type* type_;
    mutable Uses uses_;
    const size_t gid_;
    mutable uint32_t candidate_ = 0;
    static size_t gid_counter_;

public:
    mutable std::string name; ///< Just do what ever you want with this field.

    friend class Defx;
    friend class PrimOp;
    friend class Scope;
    friend class World;
    friend class Cleaner;
    friend class Tracker;
};

namespace detail {
    inline std::ostream& stream(std::ostream& os, const Def* def) { return def->stream(os); }
    inline std::ostream& stream(std::ostream& os, const Type* type) { return type->stream(os); }
}

inline std::ostream& operator<<(std::ostream& os, const Def* def) { return def->stream(os); }
inline std::ostream& operator<<(std::ostream& os, const Type* type) { return type->stream(os); }
inline std::ostream& operator<<(std::ostream& os, Use use) { return use->stream(os); }

//------------------------------------------------------------------------------

class Tracker {
public:
    Tracker()
        : def_(nullptr)
    {}
    Tracker(const Def* def)
        : def_(def)
    {
        if (def) {
            put(*this);
            verify();
        }
    }
    Tracker(const Tracker& other)
        : def_(other)
    {
        if (other) {
            put(*this);
            verify();
        }
    }
    Tracker(Tracker&& other)
        : def_(*other)
    {
        if (other) {
            other.unregister();
            other.def_ = nullptr;
            put(*this);
            verify();
        }
    }
    ~Tracker() { if (*this) unregister(); }

    const Def* operator*() const { return def_; }
    bool operator==(const Tracker& other) const { return this->def_ == other.def_; }
    bool operator!=(const Tracker& other) const { return this->def_ != other.def_; }
    bool operator==(const Def* def) const { return this->def_ == def; }
    bool operator!=(const Def* def) const { return this->def_ != def; }
    const Def* operator->() const { return **this; }
    operator const Def*() const { return **this; }
    explicit operator bool() { return def_; }
    Tracker& operator=(Tracker other) { swap(*this, other); return *this; }

    friend void swap(Tracker& t1, Tracker& t2) {
        using std::swap;

        if (t1 != t2) {
            if (t1) {
                if (t2) {
                    t1.update(t2);
                    t2.update(t1);
                } else {
                    t1.update(t2);
                }
            } else {
                assert(!t1 && t2);
                t2.update(t1);
            }

            std::swap(t1.def_, t2.def_);
        } else {
            t1.verify();
            t2.verify();
        }
    }

private:
    HashSet<Tracker*>& trackers(const Def* def);
    void verify() { assert(!def_ || trackers(def_).contains(this)); }
    void put(Tracker& other) {
        auto p = trackers(def_).insert(&other);
        assert_unused(p.second && "couldn't insert tracker");
    }

    void unregister() {
        assert(trackers(def_).contains(this) && "tracker not found");
        trackers(def_).erase(this);
    }

    void update(Tracker& other) {
        unregister();
        put(other);
    }

    mutable const Def* def_;
    friend void Def::replace(const Def*) const;
};

//------------------------------------------------------------------------------

uint64_t UseHash::operator()(Use use) const {
    return hash_combine(hash_begin(use->gid()), use.index());
}

template<>
struct Hash<Defs> {
    uint64_t operator()(Defs defs) const {
        uint64_t seed = hash_begin(defs.size());
        for (auto def : defs)
            seed = hash_combine(seed, def ? def->gid() : uint64_t(-1));
        return seed;
    }
};

template<>
struct Hash<Array<const Def*>> {
    uint64_t operator()(const Array<const Def*> defs) const { return Hash<Defs>()(defs); }
};

//------------------------------------------------------------------------------

}

#endif
