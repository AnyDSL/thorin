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
 * The World represents the whole program and manages creation of Thorin nodes (Def%s).
 * In particular, the following things are done by this class:
 *
 *  - @p Def unification: \n
 *      There exists only one unique @p Def.
 *      These @p Def%s are hashed into an internal map for fast access.
 *      The getters just calculate a hash and lookup the @p Def, if it is already present, or create a new one otherwise.
 *      This is corresponds to value numbering.
 *  - constant folding
 *  - canonicalization of expressions
 *  - several local optimizations like <tt>x + 0 -> x</tt>
 *
 *  Use @p cleanup to remove dead and unreachable code.
 *
 *  You can create several worlds.
 *  All worlds are completely independent from each other.
 *
 *  Note that types are also just @p Def%s and will be hashed as well.
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

    World(const World&) = delete;
    World(World&&) = delete;
    World& operator=(const World&) = delete;

    explicit World(uint32_t cur_gid, Debug = {});
    World(World& other)
        : World(other.cur_gid(), other.debug())
    {
        pe_done_ = other.pe_done_;
#if THORIN_ENABLE_CHECKS
        track_history_ = track_history_;
        breakpoints_   = breakpoints_;
#endif
    }
    ~World();

    // getters
    Debug debug() const { return debug_; }
    const DefSet& defs() const { return defs_; }
    std::vector<Lam*> copy_lams() const;

    /// @name manage global identifier - a unique number for each Def
    //@{
    uint32_t cur_gid() const { return cur_gid_; }
    uint32_t next_gid() { return ++cur_gid_; }
    //@}

    /// @name Universe and Kind
    //@{
    const Universe* universe() { return universe_; }
    const Kind* kind(NodeTag tag) { return unify<Kind>(0, *this, tag); }
    const Kind* kind_arity() { return kind_arity_; }
    const Kind* kind_multi() { return kind_multi_; }
    const Kind* kind_star()  { return kind_star_; }
    /// @name Param and Var
    //@{
    const Param* param(Lam* lam, Debug dbg = {}) { return unify<Param>(1, lam->domain(), lam, dbg); }
    const Var* var(const Def* type, u64 index, Debug dbg = {}) { return unify<Var>(0, type, index, dbg); }
    //@}
    //@}
    /// @name Pi
    //@{
    const Pi* pi(Defs domain, const Def* codomain, Debug dbg = {}) { return pi(sigma(domain), codomain, dbg); }
    const Pi* pi(const Def* domain, const Def* codomain, Debug dbg = {});
    //@}
    /// @name Pi: continuation type, i.e., Pi type with codomain Bottom
    //@{
    const Pi* cn() { return cn(sigma()); }
    const Pi* cn(Defs domains) { return cn(sigma(domains)); }
    const Pi* cn(const Def* domain) { return pi(domain, bot_star()); }
    //@}
    /// @name Lambda: nominal
    //@{
    Lam* lam(const Pi* cn, CC cc = CC::C, Intrinsic intrinsic = Intrinsic::None, Debug dbg = {}) {
        auto lam = insert<Lam>(2, cn, cc, intrinsic, dbg);
        lam->destroy(); // set filter to false and body to top
        return lam;
    }
    Lam* lam(const Pi* cn, Debug dbg = {}) { return lam(cn, CC::C, Intrinsic::None, dbg); }
    //@}
    /// @name Lambda: structural
    const Lam* lam(const Def* domain, const Def* filter, const Def* body, Debug dbg);
    const Lam* lam(const Def* domain, const Def* body, Debug dbg) { return lam(domain, lit_bool(true, Debug()), body, dbg); }
    //@}
    /// @name App
    //@{
    const Def* app(const Def* callee, const Def* op, Debug dbg = {});
    const Def* app(const Def* callee, Defs ops, Debug dbg = {}) { return app(callee, tuple(ops), dbg); }
    //@}
    /// @name Sigma: structural
    //@{
    const Def* sigma(const Def* type, Defs ops, Debug dbg = {});
    /// a @em structural @p Sigma of type @p star
    const Def* sigma(Defs ops, Debug dbg = {}) { return sigma(kind_star(), ops, dbg); }
    const Sigma* sigma() { return sigma_; } ///< the unit type within @p kind_star()
    //@}
    /// @name Sigma: nominal
    //@{
    Sigma* sigma(const Def* type, size_t size, Debug dbg = {}) { return insert<Sigma>(size, type, size, dbg); }
    Sigma* sigma(size_t size, Debug dbg = {}) { return sigma(kind_star(), size, dbg); } ///< a @em nominal @p Sigma of type @p star
    //@}
    /// @name Variadic
    //@{
    const Def* variadic(const Def* arity, const Def* body, Debug dbg = {});
    const Def* variadic(Defs arities, const Def* body, Debug dbg = {});
    const Def* variadic(u64 a, const Def* body, Debug dbg = {}) { return variadic(lit_arity(a, dbg), body, dbg); }
    const Def* variadic(ArrayRef<u64> a, const Def* body, Debug dbg = {}) {
        return variadic(Array<const Def*>(a.size(), [&](size_t i) { return lit_arity(a[i], dbg); }), body, dbg);
    }
    const Def* unsafe_variadic(const Def* body, Debug dbg = {}) { return variadic(top_arity(), body, dbg); }
    //@}
    /// @name Tuple
    //@{
    /// ascribes @p type to this tuple - needed for dependetly typed and structural @p Sigma%s
    const Def* tuple(const Def* type, Defs ops, Debug dbg = {});
    const Def* tuple(Defs ops, Debug dbg = {});
    const Tuple* tuple() { return tuple_; } ///< the unit value of type <tt>[]</tt>
    //@}
    /// @name Pack
    //@{
    const Def* pack(const Def* arity, const Def* body, Debug dbg = {});
    const Def* pack(Defs arities, const Def* body, Debug dbg = {});
    const Def* pack(u64 a, const Def* body, Debug dbg = {}) { return pack(lit_arity(a, dbg), body, dbg); }
    const Def* pack(ArrayRef<u64> a, const Def* body, Debug dbg = {}) {
        return pack(Array<const Def*>(a.size(), [&](auto i) { return lit_arity(a[i], dbg); }), body, dbg);
    }
    //@}
    /// @name Extract
    //@{
    const Def* extract(const Def* agg, const Def* i, Debug dbg = {});
    const Def* extract(const Def* agg, u32 i, Debug dbg = {}) { return extract(agg, lit_index(agg->arity(), i, dbg), dbg); }
    const Def* extract(const Def* agg, u32 a, u32 i, Debug dbg = {}) { return extract(agg, lit_index(a, i, dbg), dbg); }
    const Def* unsafe_extract(const Def* agg, const Def* i, Debug dbg = {}) { return extract(agg, cast(agg->arity(), i, dbg), dbg); }
    const Def* unsafe_extract(const Def* agg, u64 i, Debug dbg = {}) { return unsafe_extract(agg, lit_qu64(i, dbg), dbg); }
    //@}
    /// @name Insert
    //@{
    const Def* insert(const Def* agg, const Def* i, const Def* value, Debug dbg = {});
    const Def* insert(const Def* agg, u32 i, const Def* value, Debug dbg = {}) { return insert(agg, lit_index(agg->arity(), i, dbg), value, dbg); }
    const Def* unsafe_insert(const Def* agg, const Def* i, const Def* value, Debug dbg = {}) { return insert(agg, cast(agg->arity(), i, dbg), value, dbg); }
    const Def* unsafe_insert(const Def* agg, u32 i, const Def* value, Debug dbg = {}) { return unsafe_insert(agg, lit_qu64(i, dbg), value, dbg); }
    //@}
    /// @name LEA - load effective address
    //@{
    const Def* lea(const Def* ptr, const Def* index, Debug dbg);
    const Def* unsafe_lea(const Def* ptr, const Def* index, Debug dbg) { return lea(ptr, cast(ptr->type()->as<PtrType>()->pointee()->arity(), index, dbg), dbg); }
    //@}
    /// @name Literal
    //@{
    const Lit* lit(const Def* type, Box box, Debug dbg = {}) { return unify<Lit>(0, type, box, dbg); }
    const Lit* lit(PrimTypeTag tag, Box box, Loc loc = {}) { return lit(type(tag), box, loc); }
    //@}
    /// @name Literal: Arity - note that this is a type
    //@{
    const Lit* lit_arity(u64 a, Loc loc = {});
    const Lit* lit_arity_1() { return lit_arity_1_; } ///< unit arity 1ₐ
    //@}
    /// @name Literal: Index - the inhabitants of an Arity
    //@{
    const Lit* lit_index(u64 arity, u64 idx, Loc loc = {}) { return lit_index(lit_arity(arity), idx, loc); }
    const Lit* lit_index(const Def* arity, u64 index, Loc loc = {});
    const Lit* lit_index_0_1() { return lit_index_0_1_; } ///< unit index 0₁ of type unit arity 1ₐ
    //@}
    /// @name Literal: Nat
    //@{
    const Lit* lit_nat(int64_t val, Loc loc = {}) { return lit(type_nat(), {val}, {loc}); }
    const Lit* lit_nat_0 () { return lit_nat_0_; }
    const Lit* lit_nat_1 () { return lit_nat_[0]; }
    const Lit* lit_nat_2 () { return lit_nat_[1]; }
    const Lit* lit_nat_4 () { return lit_nat_[2]; }
    const Lit* lit_nat_8 () { return lit_nat_[3]; }
    const Lit* lit_nat_16() { return lit_nat_[4]; }
    const Lit* lit_nat_32() { return lit_nat_[5]; }
    const Lit* lit_nat_64() { return lit_nat_[6]; }
    //@}
    /// @name Literal: Bool
    //@{
    const Lit* lit(bool val) { return lit_bool_[size_t(val)]; }
    const Lit* lit_false() { return lit_bool_[0]; }
    const Lit* lit_true()  { return lit_bool_[1]; }
    //@}
    /// @name Literal: PrimTypes
    //@{
#define THORIN_ALL_TYPE(T, M) \
    const Def* lit_##T(T val, Loc loc = {}) { return lit(PrimType_##T, Box(val), loc); }
#include "thorin/tables/primtypetable.h"
    const Lit* lit_zero(PrimTypeTag tag, Loc loc = {}) { return lit(tag, 0, loc); }
    const Lit* lit_zero(const Def* type, Loc loc = {}) { return lit_zero(type->as<PrimType>()->primtype_tag(), loc); }
    const Lit* lit_one(PrimTypeTag tag, Loc loc = {}) { return lit(tag, 1, loc); }
    const Lit* lit_one(const Def* type, Loc loc = {}) { return lit_one(type->as<PrimType>()->primtype_tag(), loc); }
    const Lit* lit_allset(PrimTypeTag tag, Loc loc = {});
    const Lit* lit_allset(const Def* type, Loc loc = {}) { return lit_allset(type->as<PrimType>()->primtype_tag(), loc); }
    //@}
    /// @name Top/Bottom
    //@{
    const Def* bot_top(bool is_top, const Def* type, Debug dbg = {});
    const Def* bot(const Def* type, Loc dbg = {}) { return bot_top(false, type, dbg); }
    const Def* top(const Def* type, Loc dbg = {}) { return bot_top( true, type, dbg); }
    const Def* bot(PrimTypeTag tag, Loc dbg = {}) { return bot_top(false, type(tag), dbg); }
    const Def* top(PrimTypeTag tag, Loc dbg = {}) { return bot_top( true, type(tag), dbg); }
    const Def* bot_star () { return bot_star_; }
    const Def* top_arity() { return top_arity_; } ///< use this guy to encode an unknown arity, e.g., for unsafe arrays
    //@}
    /// @name Variant
    //@{
    const VariantType* variant_type(Defs ops, Debug dbg = {}) { return unify<VariantType>(ops.size(), kind_star(), ops, dbg); }
    const Def* variant(const VariantType* variant_type, const Def* value, Debug dbg = {}) { return unify<Variant>(1, variant_type, value, dbg); }
    //@}
    /// @name misc types
    //@{
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
    const PtrType* ptr_type(const Def* pointee, AddrSpace addr_space = AddrSpace::Generic, Debug dbg = {}) { return unify<PtrType>(1, kind_star(), pointee, addr_space, dbg); }
    //@}
    /// @name ArithOps
    //@{
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
    //@}
    /// @name Cmps
    //@{
    const Def* cmp(CmpTag tag, const Def* lhs, const Def* rhs, Debug dbg = {});
#define THORIN_CMP(OP) \
    const Def* cmp_##OP(const Def* lhs, const Def* rhs, Debug dbg = {}) { \
        return cmp(Cmp_##OP, lhs, rhs, dbg);  \
    }
#include "thorin/tables/cmptable.h"
    //@}
    /// @name Casts
    //@{
    const Def* convert(const Def* to, const Def* from, Debug dbg = {});
    const Def* cast(const Def* to, const Def* from, Debug dbg = {});
    const Def* bitcast(const Def* to, const Def* from, Debug dbg = {});
    //@}
    /// @name memory-related operations
    //@{
    const Def* load(const Def* mem, const Def* ptr, Debug dbg = {});
    const Def* store(const Def* mem, const Def* ptr, const Def* val, Debug dbg = {});
    const Def* enter(const Def* mem, Debug dbg = {});
    const Def* slot(const Def* type, const Def* frame, Debug dbg = {});
    const Def* alloc(const Def* type, const Def* mem, Debug dbg = {});
    const Def* global(const Def* init, bool is_mutable = true, Debug dbg = {});
    const Def* global_immutable_string(const std::string& str, Debug dbg = {});
    const Assembly* assembly(const Def* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});
    const Assembly* assembly(Defs types, const Def* mem, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});
    //@}
    /// @name partial evaluation related operations
    //@{
    const Def* hlt(const Def* def, Debug dbg = {});
    const Def* known(const Def* def, Debug dbg = {});
    const Def* run(const Def* def, Debug dbg = {});
    //@}
    /// @name misc operations
    //@{
    const Def* select(const Def* cond, const Def* t, const Def* f, Debug dbg = {});
    const Def* size_of(const Def* type, Debug dbg = {});
    //@}
    /// @name Rewrite
    //@{
    const Def* rewrite(const Def* type, const Def* def, const Def* from, const Def* to, u32 depth, Debug dbg = {}) { return unify<Rewrite>(3, type, def, from, to, depth, dbg = {}); }
    /// Inherits the type of the @p Rewrite from @p def.
    const Def* rewrite(const Def* def, const Def* from, const Def* to, u32 depth, Debug dbg = {}) { return rewrite(def->type(), from, to, depth, dbg); }
    //@}
    // TODO not all of them are axioms right now
    /// @name Axioms
    //@{
    Axiom* axiom(const Def* type, Normalizer, Debug dbg = {});
    Axiom* axiom(const Def* type, Debug dbg = {}) { return axiom(type, nullptr, dbg); }
    std::optional<Axiom*> lookup_axiom(Symbol name) { return axioms_.lookup(name); }
    Axiom* type_nat() { return type_nat_; }
    Lam* branch() const { return branch_; }
    Lam* match(const Def* type, size_t num_patterns);
    Lam* end_scope() const { return end_scope_; }
    const Def* branch(const Def* cond, const Def* t, const Def* f, Debug dbg = {}) { return app(branch(), {cond, t, f}, dbg); }
    //@}

    /// Performs dead code, unreachable code and unused type elimination.
    void cleanup();
    void opt();

    // other stuff

    void mark_pe_done(bool flag = true) { pe_done_ = flag; }
    bool is_pe_done() const { return pe_done_; }

    /// @name manage externals
    //@{
    bool empty() { return externals().empty(); }
    const LamSet& externals() const { return externals_; }
    void add_external(Lam* lam) { externals_.insert(lam); }
    void remove_external(Lam* lam) { externals_.erase(lam); }
    bool is_external(const Lam* lam) { return externals().contains(const_cast<Lam*>(lam)); }
    //@}

#if THORIN_ENABLE_CHECKS
    /// @name debugging features
    //@{
    void breakpoint(size_t number) { breakpoints_.insert(number); }
    const Breakpoints& breakpoints() const { return breakpoints_; }
    void swap_breakpoints(World& other) { swap(this->breakpoints_, other.breakpoints_); }
    bool track_history() const { return track_history_; }
    void enable_history(bool flag = true) { track_history_ = flag; }
    //@}
#endif

    /// @name stream
    //@{
    // Note that we don't use overloading for the following methods in order to have them accessible from gdb.
    virtual std::ostream& stream(std::ostream&) const override; ///< Streams thorin to file @p out.
    void write_thorin(const char* filename) const;              ///< Dumps thorin to file with name @p filename.
    void thorin() const;                                        ///< Dumps thorin to a file with an auto-generated file name.
    //@}

    friend void swap(World& w1, World& w2) {
        using std::swap;
        swap(w1.root_page_,     w2.root_page_);
        swap(w1.cur_page_,      w2.cur_page_);
        swap(w1.cur_gid_,       w2.cur_gid_);
        swap(w1.buffer_index_,  w2.buffer_index_);
        swap(w1.debug_,         w2.debug_);
        swap(w1.axioms_,        w2.axioms_);
        swap(w1.externals_,     w2.externals_);
        swap(w1.defs_,          w2.defs_);
        swap(w1.universe_,      w2.universe_);
        swap(w1.kind_arity_,    w2.kind_arity_);
        swap(w1.kind_multi_,    w2.kind_multi_);
        swap(w1.kind_star_,     w2.kind_star_);
        swap(w1.bot_star_,      w2.bot_star_);
        swap(w1.top_arity_,     w2.top_arity_);
        swap(w1.mem_,           w2.mem_);
        swap(w1.frame_,         w2.frame_);
        swap(w1.type_nat_,      w2.type_nat_);
        swap(w1.lit_nat_0_,     w2.lit_nat_0_);
        swap(w1.lit_nat_,       w2.lit_nat_);
        swap(w1.lit_arity_1_,   w2.lit_arity_1_);
        swap(w1.lit_index_0_1_, w2.lit_index_0_1_);
        swap(w1.lit_bool_,      w2.lit_bool_);
        swap(w1.sigma_,         w2.sigma_);
        swap(w1.tuple_,         w2.tuple_);
        swap(w1.branch_,        w2.branch_);
        swap(w1.end_scope_,     w2.end_scope_);
        swap(w1.pe_done_,       w2.pe_done_);

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

    template<class T, class... Args>
    const T* unify(size_t num_ops, Args&&... args) {
        auto def = alloc<T>(num_ops, args...);
#ifndef NDEBUG
        if (breakpoints_.contains(def->gid())) THORIN_BREAK;
#endif
        assert(!def->isa_nominal());
        auto [i, success] = defs_.emplace(def);
        if (success) {
            def->finalize();
            return def;
        }

        dealloc<T>(def);
        return static_cast<const T*>(*i);
    }

    template<class T, class... Args>
    T* insert(size_t num_ops, Args&&... args) {
        auto def = alloc<T>(num_ops, args...);
#ifndef NDEBUG
        if (breakpoints_.contains(def->gid())) THORIN_BREAK;
#endif
        auto p = defs_.emplace(def);
        assert_unused(p.second);
        return def;
    }

    struct Zone {
        static const size_t Size = 1024 * 1024 - sizeof(std::unique_ptr<int>); // 1MB - sizeof(next)
        char buffer[Size];
        std::unique_ptr<Zone> next;
    };

#ifndef NDEBUG
    struct Lock {
        Lock() { assert((alloc_guard_ = !alloc_guard_) && "you are not allowed to recursively invoke alloc"); }
        ~Lock() { alloc_guard_ = !alloc_guard_; }
        static bool alloc_guard_;
    };
#else
    struct Lock { ~Lock() {} };
#endif

    static inline size_t align(size_t n) { return (n + (sizeof(void*) - 1)) & ~(sizeof(void*)-1); }

    template<class T> static inline size_t num_bytes_of(size_t num_ops) {
        size_t result = std::is_empty<typename T::Extra>() ? 0 : sizeof(typename T::Extra);
        result += sizeof(Def) + sizeof(const Def*)*num_ops;
        return align(result);
    }

    template<class T, class... Args>
    T* alloc(size_t num_ops, Args&&... args) {
        static_assert(sizeof(Def) == sizeof(T), "you are not allowed to introduce any additional data in subclasses of Def - use 'Extra' struct");
        Lock lock;
        size_t num_bytes = num_bytes_of<T>(num_ops);
        num_bytes = align(num_bytes);
        assert(num_bytes < Zone::Size);

        if (buffer_index_ + num_bytes >= Zone::Size) {
            auto page = new Zone;
            cur_page_->next.reset(page);
            cur_page_ = page;
            buffer_index_ = 0;
        }

        auto result = new (cur_page_->buffer + buffer_index_) T(args...);
        assert(result->num_ops() == num_ops);
        buffer_index_ += num_bytes;
        assert(buffer_index_ % alignof(T) == 0);

        return result;
    }

    template<class T>
    void dealloc(const T* def) {
        size_t num_bytes = num_bytes_of<T>(def->num_ops());
        num_bytes = align(num_bytes);
        def->~T();
        if (ptrdiff_t(buffer_index_ - num_bytes) > 0) // don't care otherwise
            buffer_index_-= num_bytes;
        assert(buffer_index_ % alignof(T) == 0);
    }

    std::unique_ptr<Zone> root_page_;
    Zone* cur_page_;
    size_t buffer_index_ = 0;
    uint32_t cur_gid_;
    Debug debug_;
    SymbolMap<Axiom*> axioms_;
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
    const BotTop* bot_star_;
    const BotTop* top_arity_;
    const Sigma* sigma_;
    const Tuple* tuple_;
    const MemType* mem_;
    const FrameType* frame_;
    Axiom* type_nat_;
    const Lit* lit_nat_0_;
    std::array<const Lit*, 2> lit_bool_;
    std::array<const Lit*, 7> lit_nat_;
    const Lit* lit_arity_1_;
    const Lit* lit_index_0_1_;
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
