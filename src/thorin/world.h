#ifndef THORIN_WORLD_H
#define THORIN_WORLD_H

#include <cassert>
#include <iostream>
#include <functional>
#include <initializer_list>
#include <string>

#include "thorin/enums.h"
#include "thorin/primop.h"
#include "thorin/util/hash.h"
#include "thorin/util/stream.h"
#include "thorin/config.h"

namespace thorin {

/**
 * The World represents the whole program and manages creation and destruction of Thorin nodes.
 * In particular, the following things are done by this class:
 *
 *  - @p Type unification: \n
 *      There exists only one unique @p Type.
 *      These @p Type%s are hashed into an internal map for fast access.
 *      The getters just calculate a hash and lookup the @p Type, if it is already present, or create a new one otherwise.
 *  - Value unification: \n
 *      This is a built-in mechanism for the following things:
 *      - constant pooling
 *      - constant folding
 *      - common subexpression elimination
 *      - canonicalization of expressions
 *      - several local optimizations
 *
 *  @p PrimOp%s do not explicitly belong to a Lam.
 *  Instead they either implicitly belong to a Lam--when
 *  they (possibly via multiple levels of indirection) depend on a Lam's Param--or they are dead.
 *  Use @p cleanup to remove dead code and unreachable code.
 *
 *  You can create several worlds.
 *  All worlds are completely independent from each other.
 *  This is particular useful for multi-threading.
 */
class World : public Streamable {
public:
    struct DefHash {
        static uint64_t hash(const Def* def) { return def->hash(); }
        static bool eq(const Def* def1, const Def* def2) { return def1->equal(def2); }
        static const Def* sentinel() { return (const Def*)(1); }
    };

    typedef HashSet<const Def*, DefHash> DefSet;

    struct BreakHash {
        static uint64_t hash(size_t i) { return i; }
        static bool eq(size_t i1, size_t i2) { return i1 == i2; }
        static size_t sentinel() { return size_t(-1); }
    };

    typedef HashSet<size_t, BreakHash> Breakpoints;

    explicit World(Debug = {});
    ~World();

    // getters
    bool empty() { return externals().empty(); }
    Debug debug() const { return debug_; }
    const DefSet& defs() const { return defs_; }
    std::vector<Lam*> copy_lams() const;
    const LamSet& externals() const { return externals_; }

    /// @defgroup core @p Def%s
    //@{
    /// @defgroup @p Universe and @p Kind%s
    //@{
    const Universe* universe() { return universe_; }
    const Kind* kind(NodeTag tag) { return unify(new Kind(*this, tag)); }
    const Kind* kind_arity() { return kind_arity_; }
    const Kind* kind_multi() { return kind_multi_; }
    const Kind* kind_star()  { return kind_star_; }
    const Var* var(const Def* type, u64 index, Debug dbg = {}) { return unify(new Var(type, index, dbg)); }
    const VariantType* variant_type(Defs ops, Debug dbg = {}) { return unify(new VariantType(kind_star(), ops, dbg)); }
    //@}
    /// @defgroup @p Pi%s
    //@{
    const Pi* pi(Defs domain, const Def* codomain, Debug dbg = {}) { return pi(sigma(domain), codomain, dbg); }
    const Pi* pi(const Def* domain, const Def* codomain, Debug dbg = {});
    ///@defgroup continuation types - Pi types with codomain bottom
    //@{
    const Pi* cn() { return cn(sigma()); }
    const Pi* cn(Defs domains) { return cn(sigma(domains)); }
    const Pi* cn(const Def* domain) { return pi(domain, bot()); }
    //@}
    //@}
    /// @defgroup @p Sigma%s
    //@{
    /// @defgroup @em structural @p Sigma%s
    //@{
    const Def* sigma(const Def* type, Defs ops, Debug dbg = {});
    /// a @em structural @p Sigma of type @p star
    const Def* sigma(Defs ops, Debug dbg = {}) { return sigma(kind_star(), ops, dbg); }
    const Sigma* sigma() { return sigma_; } ///< Returns an empty @p Sigma - AKA unit - of type @p star
    //@}
    /// @defgroup @em nominal @p Sigma%s
    //@{
    Sigma* sigma(const Def* type, size_t size, Debug dbg = {}) { return insert(new Sigma(type, size, dbg)); }
    /// a @em nominal @p Sigma of type @p star
    Sigma* sigma(size_t size, Debug dbg = {}) { return sigma(kind_star(), size, dbg); }
    //@}
    //@}
    /// @defgroup Variadic%s
    //@{
    const Def* variadic(const Def* arities, const Def* body, Debug dbg = {});
    const Def* variadic(Defs arities, const Def* body, Debug dbg = {});
    const Def* variadic(u64 a, const Def* body, Debug dbg = {}) { return variadic(lit_arity(a, dbg), body, dbg); }
    const Def* variadic(ArrayRef<u64> a, const Def* body, Debug dbg = {}) {
        return variadic(Array<const Def*>(a.size(), [&](size_t i) { return lit_arity(a[i], dbg); }), body, dbg);
    }
    //@}
    /// @defgroup Pack%s
    //@{
    const Def* pack(const Def* arities, const Def* body, Debug dbg = {});
    const Def* pack(Defs arities, const Def* body, Debug dbg = {});
    const Def* pack(u64 a, const Def* body, Debug dbg = {}) { return pack(lit_arity(a, dbg), body, dbg); }
    const Def* pack(ArrayRef<u64> a, const Def* body, Debug dbg = {}) {
        return pack(Array<const Def*>(a.size(), [&](auto i) { return lit_arity(a[i], dbg); }), body, dbg);
    }
    //@}
    /// @defgroup create @p Lit%erals
    //@{
    const Lit* lit(const Def* type, Box box, Debug dbg) { return unify(new Lit(type, box, dbg)); }
    const Lit* lit(PrimTypeTag tag, Box box, Debug dbg) { return lit(type(tag), box, dbg); }
    template<class T>
    const Lit* lit(T value, Debug dbg = {}) { return lit(type2tag<T>::tag, Box(value), dbg); }
#define THORIN_ALL_TYPE(T, M) \
    const Def* lit_##T(T val, Debug dbg = {}) { return lit(PrimType_##T, Box(val), dbg); }
#include "thorin/tables/primtypetable.h"
    const Lit* zero(PrimTypeTag tag, Debug dbg = {}) { return lit(tag, 0, dbg); }
    const Lit* zero(const Def* type, Debug dbg = {}) { return zero(type->as<PrimType>()->primtype_tag(), dbg); }
    const Lit* one(PrimTypeTag tag, Debug dbg = {}) { return lit(tag, 1, dbg); }
    const Lit* one(const Def* type, Debug dbg = {}) { return one(type->as<PrimType>()->primtype_tag(), dbg); }
    const Lit* allset(PrimTypeTag tag, Debug dbg = {});
    const Lit* allset(const Def* type, Debug dbg = {}) { return allset(type->as<PrimType>()->primtype_tag(), dbg); }
    const Lit* lit_arity(u64 a, Loc loc = {});
    const Lit* lit_index(u64 arity, u64 idx, Loc loc = {}) { return lit_index(lit_arity(arity), idx, loc); }
    const Lit* lit_index(const Lit* arity, u64 index, Loc loc = {});
    //@}
    /// @defgroup top/bottom
    //@{
    const Def* bot_top(bool is_top, const Def* type, Debug dbg = {}) { return unify(new BotTop(is_top, type, dbg)); }
    const Def* bot(const Def* type, Debug dbg = {}) { return bot_top(false, type, dbg); }
    const Def* top(const Def* type, Debug dbg = {}) { return bot_top(true,  type, dbg); }
    const Def* bot(PrimTypeTag tag, Debug dbg = {}) { return bot_top(false, type(tag), dbg); }
    const Def* top(PrimTypeTag tag, Debug dbg = {}) { return bot_top( true, type(tag), dbg); }
    const Def* bot(Debug dbg = {}) { return bot_top(false, kind_star(), dbg); }
    const Def* top(Debug dbg = {}) { return bot_top(true,  kind_star(), dbg); }
    //@}
    //@}

#define THORIN_ALL_TYPE(T, M) \
    const PrimType* type_##T() { return type(PrimType_##T); }
#include "thorin/tables/primtypetable.h"
    const PrimType* type(PrimTypeTag tag) {
        size_t i = tag - Begin_PrimType;
        assert(i < (size_t) Num_PrimTypes);
        return primtypes_[i];
    }
    const MemType* mem_type() const { return mem_; }
    const FrameType* frame_type() const { return frame_; }
    const PtrType* ptr_type(const Def* pointee, AddrSpace addr_space = AddrSpace::Generic, Debug dbg = {}) { return unify(new PtrType(kind_star(), pointee, addr_space, dbg)); }
    const DefiniteArrayType*   definite_array_type(const Def* elem, u64 dim, Debug dbg = {}) { return unify(new DefiniteArrayType(kind_star(), elem, dim, dbg)); }
    const IndefiniteArrayType* indefinite_array_type(const Def* elem, Debug dbg = {}) { return unify(new IndefiniteArrayType(kind_star(), elem, dbg)); }

    // arithops

    /// Creates an \p ArithOp or a \p Cmp.
    const Def* binop(int tag, const Def* lhs, const Def* rhs, Debug dbg = {});
    const Def* arithop_not(const Def* def, Debug dbg = {});
    const Def* arithop_minus(const Def* def, Debug dbg = {});
    const Def* arithop(ArithOpTag tag, const Def* lhs, const Def* rhs, Debug dbg = {});
#define THORIN_ARITHOP(OP) \
    const Def* arithop_##OP(const Def* lhs, const Def* rhs, Debug dbg = {}) { \
        return arithop(ArithOp_##OP, lhs, rhs, dbg); \
    }
#include "thorin/tables/arithoptable.h"

    // compares

    const Def* cmp(CmpTag tag, const Def* lhs, const Def* rhs, Debug dbg = {});
#define THORIN_CMP(OP) \
    const Def* cmp_##OP(const Def* lhs, const Def* rhs, Debug dbg = {}) { \
        return cmp(Cmp_##OP, lhs, rhs, dbg);  \
    }
#include "thorin/tables/cmptable.h"

    // casts

    const Def* convert(const Def* to, const Def* from, Debug dbg = {});
    const Def* cast(const Def* to, const Def* from, Debug dbg = {});
    const Def* bitcast(const Def* to, const Def* from, Debug dbg = {});

    // aggregate operations

    const Def* definite_array(const Def* elem, Defs ops, Debug dbg = {}) {
        return try_fold_aggregate(unify(new DefiniteArray(*this, elem, ops, dbg)));
    }
    /// Create definite_array with at least one element. The type of that element is the element type of the definite array.
    const Def* definite_array(Defs ops, Debug dbg = {}) {
        assert(!ops.empty());
        return definite_array(ops.front()->type(), ops, dbg);
    }
    const Def* indefinite_array(const Def* elem, const Def* dim, Debug dbg = {}) {
        return unify(new IndefiniteArray(*this, elem, dim, dbg));
    }
    const Def* tuple(const Def* type, Defs ops, Debug dbg = {});
    const Def* tuple(Defs ops, Debug dbg = {});
    const Def* variant(const VariantType* variant_type, const Def* value, Debug dbg = {}) { return unify(new Variant(variant_type, value, dbg)); }
    const Def* extract(const Def* tuple, const Def* index, Debug dbg = {});
    const Def* extract(const Def* tuple, u32 index, Debug dbg = {}) {
        return extract(tuple, lit_qu32(index, dbg), dbg);
    }
    const Def* insert(const Def* tuple, const Def* index, const Def* value, Debug dbg = {});
    const Def* insert(const Def* tuple, u32 index, const Def* value, Debug dbg = {}) {
        return insert(tuple, lit_qu32(index, dbg), value, dbg);
    }

    const Def* select(const Def* cond, const Def* t, const Def* f, Debug dbg = {});
    const Def* size_of(const Def* type, Debug dbg = {});

    // memory stuff

    const Def* load(const Def* mem, const Def* ptr, Debug dbg = {});
    const Def* store(const Def* mem, const Def* ptr, const Def* val, Debug dbg = {});
    const Def* enter(const Def* mem, Debug dbg = {});
    const Def* slot(const Def* type, const Def* frame, Debug dbg = {}) { return unify(new Slot(type, frame, dbg)); }
    const Def* alloc(const Def* type, const Def* mem, const Def* extra, Debug dbg = {});
    const Def* alloc(const Def* type, const Def* mem, Debug dbg = {}) { return alloc(type, mem, lit_qu64(0, dbg), dbg); }
    const Def* global(const Def* init, bool is_mutable = true, Debug dbg = {});
    const Def* global_immutable_string(const std::string& str, Debug dbg = {});
    const Def* lea(const Def* ptr, const Def* index, Debug dbg);
    const Assembly* assembly(const Def* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});
    const Assembly* assembly(Defs types, const Def* mem, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});

    // partial evaluation related stuff

    const Def* hlt(const Def* def, Debug dbg = {});
    const Def* known(const Def* def, Debug dbg = {});
    const Def* run(const Def* def, Debug dbg = {});

    // lams

    const Param* param(Lam* lam, Debug dbg = {}) { return unify(new Param(lam->domain(), lam, dbg)); }
    /// @defgroup nominal @p Lam%bdas
    //@{
    Lam* lam(const Pi* cn, CC cc = CC::C, Intrinsic intrinsic = Intrinsic::None, Debug dbg = {}) { return insert(new Lam(cn, cc, intrinsic, dbg)); }
    Lam* lam(const Pi* cn, Debug dbg = {}) { return lam(cn, CC::C, Intrinsic::None, dbg); }
    //@}
    /// @defgroup structural @p Lam%bdas
    const Lam* lam(const Def* domain, const Def* filter, const Def* body, Debug dbg);
    const Lam* lam(const Def* domain, const Def* body, Debug dbg) { return lam(domain, lit_bool(true, Debug()), body, dbg); }
    //@{
    Lam* branch() const { return branch_; }
    Lam* match(const Def* type, size_t num_patterns);
    Lam* end_scope() const { return end_scope_; }
    const Def* app(const Def* callee, const Def* op, Debug dbg = {});
    const Def* app(const Def* callee, Defs ops, Debug dbg = {}) { return app(callee, tuple(ops), dbg); }
    const Def* branch(const Def* cond, const Def* t, const Def* f, Debug dbg = {}) { return app(branch(), {cond, t, f}, dbg); }

    /// Performs dead code, unreachable code and unused type elimination.
    void cleanup();
    void opt();

    // other stuff

    void mark_pe_done(bool flag = true) { pe_done_ = flag; }
    bool is_pe_done() const { return pe_done_; }
    void add_external(Lam* lam) { externals_.insert(lam); }
    void remove_external(Lam* lam) { externals_.erase(lam); }
    bool is_external(const Lam* lam) { return externals().contains(const_cast<Lam*>(lam)); }
#if THORIN_ENABLE_CHECKS
    void breakpoint(size_t number) { breakpoints_.insert(number); }
    const Breakpoints& breakpoints() const { return breakpoints_; }
    void swap_breakpoints(World& other) { swap(this->breakpoints_, other.breakpoints_); }
    bool track_history() const { return track_history_; }
    void enable_history(bool flag = true) { track_history_ = flag; }
#endif

    // Note that we don't use overloading for the following methods in order to have them accessible from gdb.
    virtual std::ostream& stream(std::ostream&) const override; ///< Streams thorin to file @p out.
    void write_thorin(const char* filename) const;              ///< Dumps thorin to file with name @p filename.
    void thorin() const;                                        ///< Dumps thorin to a file with an auto-generated file name.

    friend void swap(World& w1, World& w2) {
        using std::swap;
        swap(w1.debug_,      w2.debug_);
        swap(w1.externals_,  w2.externals_);
        swap(w1.defs_,       w2.defs_);
        swap(w1.universe_,   w2.universe_);
        swap(w1.kind_arity_, w2.kind_arity_);
        swap(w1.kind_multi_, w2.kind_multi_);
        swap(w1.kind_star_,  w2.kind_star_);
        swap(w1.sigma_,      w2.sigma_);
        swap(w1.bot_,        w2.bot_);
        swap(w1.mem_,        w2.mem_);
        swap(w1.frame_,      w2.frame_);
        swap(w1.branch_,     w2.branch_);
        swap(w1.end_scope_,  w2.end_scope_);
        swap(w1.pe_done_,    w2.pe_done_);

#define THORIN_ALL_TYPE(T, M) \
        swap(w1.T##_,       w2.T##_);
#include "thorin/tables/primtypetable.h"

#if THORIN_ENABLE_CHECKS
        swap(w1.breakpoints_,   w2.breakpoints_);
        swap(w1.track_history_, w2.track_history_);
#endif
        swap(w1.universe_->world_, w2.universe_->world_);
        assert(&w1.universe()->world() == &w1);
        assert(&w2.universe()->world() == &w2);
    }

private:
    const Def* try_fold_aggregate(const Def*);

    template<class T>
    T* insert(T* def) {
#ifndef NDEBUG
        if (breakpoints_.contains(def->gid())) THORIN_BREAK;
#endif
        auto p = defs_.emplace(def);
        assert_unused(p.second);
        return def;
    }

    template<class T>
    const T* unify(T* def) {
#ifndef NDEBUG
        if (breakpoints_.contains(def->gid())) THORIN_BREAK;
#endif
        assert(!def->is_nominal());
        auto [i, success] = defs_.emplace(def);
        if (success) {
            def->finalize();
            return static_cast<const T*>(def);
        }

        delete def;
        return static_cast<const T*>(*i);
    }

    Debug debug_;
    LamSet externals_;
    DefSet defs_;
    bool pe_done_ = false;
#if THORIN_ENABLE_CHECKS
    Breakpoints breakpoints_;
    bool track_history_ = false;
#endif
    Universe* universe_;
    const Kind* kind_arity_;
    const Kind* kind_multi_;
    const Kind* kind_star_;
    const Sigma* sigma_;
    const BotTop* bot_;
    const MemType* mem_;
    const FrameType* frame_;
    union {
        struct {
#define THORIN_ALL_TYPE(T, M) const PrimType* T##_;
#include "thorin/tables/primtypetable.h"
        };

        const PrimType* primtypes_[Num_PrimTypes];
    };
    Lam* branch_;
    Lam* end_scope_;

    friend class Cleaner;
    friend class Lam;
    friend void Def::replace(Tracker) const;
};

}

#endif
