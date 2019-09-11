#ifndef THORIN_WORLD_H
#define THORIN_WORLD_H

#include <cassert>
#include <iostream>
#include <functional>
#include <initializer_list>
#include <string>

#include "thorin/def.h"
#include "thorin/primop.h"
#include "thorin/util/hash.h"
#include "thorin/util/stream.h"
#include "thorin/config.h"

namespace thorin {

const Def* infer_width(const Def*);

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
    struct SeaHash {
        static uint32_t hash(const Def* def) { return def->hash(); }
        static bool eq(const Def* def1, const Def* def2) { return def1->equal(def2); }
        static const Def* sentinel() { return (const Def*)(1); }
    };

    struct BreakHash {
        static uint32_t hash(size_t i) { return i; }
        static bool eq(size_t i1, size_t i2) { return i1 == i2; }
        static size_t sentinel() { return size_t(-1); }
    };

    struct ExternalsHash {
        static uint32_t hash(const std::string& s) { return thorin::hash(s.c_str()); }
        static bool eq(const std::string& s1, const std::string& s2) { return s1 == s2; }
        static std::string sentinel() { return std::string(); }
    };

    using Sea         = HashSet<const Def*, SeaHash>;///< This @p HashSet contains Thorin's "sea of nodes".
    using Breakpoints = HashSet<size_t, BreakHash>;
    using Externals   = HashMap<std::string, Def*, ExternalsHash>;

    World(World&&) = delete;
    World& operator=(const World&) = delete;

    explicit World(uint32_t cur_gid, const std::string& name = {});
    ///  Inherits properties of the @p other @p World but does @em not perform a copy.
    explicit World(const World& other)
        : World(other.cur_gid(), other.name())
    {
        pe_done_ = other.pe_done_;
#if THORIN_ENABLE_CHECKS
        track_history_ = other.track_history_;
        breakpoints_   = other.breakpoints_;
#endif
    }
    ~World();

    // getters
    const std::string& name() const { return name_; }
    const Sea& defs() const { return defs_; }
    std::vector<Lam*> copy_lams() const;

    /// @name manage global identifier - a unique number for each Def
    //@{
    uint32_t cur_gid() const { return cur_gid_; }
    uint32_t next_gid() { return ++cur_gid_; }
    //@}
    /// @name Universe and Kind
    //@{
    const Universe* universe() { return cache_.universe_; }
    const KindArity* kind_arity() { return cache_.kind_arity_; }
    const KindMulti* kind_multi() { return cache_.kind_multi_; }
    const KindStar*  kind_star()  { return cache_.kind_star_; }
    //@}
    /// @name Param
    //@{
    const Param* param(const Def* type, Def* nominal, Debug dbg) { return unify<Param>(1, type, nominal, debug(dbg)); }
    //@}
    /// @name Axiom
    //@{
    Axiom* axiom(Def::NormalizeFn normalize, const Def* type, tag_t tag, flags_t flags, Debug dbg);
    Axiom* axiom(const Def* type, tag_t tag, flags_t flags, Debug dbg) { return axiom(nullptr, type, tag, flags, dbg); }
    //@}
    /// @name Pi
    //@{
    const Pi* pi(const Def* domain, const Def* codomain, Debug dbg = {});
    const Pi* pi(Defs domain, const Def* codomain, Debug dbg = {}) { return pi(sigma(domain), codomain, dbg); }
    Pi* pi(const Def* type, Debug dbg = {}) { return insert<Pi>(2, type, debug(dbg)); } ///< @em nominal Pi.
    //@}
    /// @name Pi: continuation type, i.e., Pi type with codomain Bottom
    //@{
    const Pi* cn() { return cn(sigma()); }
    const Pi* cn(Defs domains) { return cn(sigma(domains)); }
    const Pi* cn(const Def* domain) { return pi(domain, bot_star()); }
    //@}
    /// @name Lambda: nominal
    //@{
    Lam* lam(const Pi* cn, Lam::CC cc = Lam::CC::C, Lam::Intrinsic intrinsic = Lam::Intrinsic::None, Debug dbg = {}) {
        auto lam = insert<Lam>(2, cn, cc, intrinsic, debug(dbg));
        return lam;
    }
    Lam* lam(const Pi* cn, Debug dbg = {}) { return lam(cn, Lam::CC::C, Lam::Intrinsic::None, dbg); }
    //@}
    /// @name Lambda: structural
    //@{
    const Lam* lam(const Def* domain, const Def* filter, const Def* body, Debug dbg);
    const Lam* lam(const Def* domain, const Def* body, Debug dbg) { return lam(domain, lit_true(), body, dbg); }
    //@}
    /// @name App
    //@{
    const Def* app(const Def* callee, const Def* arg, Debug dbg = {});
    const Def* app(const Def* callee, Defs args, Debug dbg = {}) { return app(callee, tuple(args), dbg); }
    /// Does not invoke the @p App's normalizer.
    const Def* raw_app(const Def* callee, const Def* arg, Debug dbg = {}) {
        auto type = callee->type()->as<Pi>()->apply(arg);
        auto [axiom, currying_depth] = get_axiom(callee);
        return unify<App>(2, axiom, currying_depth, type, callee, arg, debug(dbg));
    }
    /// Does not invoke the @p App's normalizer.
    const Def* raw_app(const Def* callee, Defs args, Debug dbg = {}) { return raw_app(callee, tuple(args, dbg), dbg); }
    //@}
    /// @name Sigma: structural
    //@{
    const Def* sigma(const Def* type, Defs ops, Debug dbg = {});
    /// a @em structural @p Sigma of type @p star
    const Def* sigma(Defs ops, Debug dbg = {}) { return sigma(kind_star(), ops, dbg); }
    const Sigma* sigma() { return cache_.sigma_; } ///< the unit type within @p kind_star()
    //@}
    /// @name Sigma: nominal
    //@{
    Sigma* sigma(const Def* type, size_t size, Debug dbg = {}) { return insert<Sigma>(size, type, size, debug(dbg)); }
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
    const Def* tuple_str(const char* s, Debug = {});
    const Def* tuple_str(const std::string& s, Debug dbg = {}) { return tuple_str(s.c_str(), dbg); }
    const Tuple* tuple() { return cache_.tuple_; } ///< the unit value of type <tt>[]</tt>
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
    const Def* extract(const Def* agg, u64 i, Debug dbg = {}) { return extract(agg, lit_index(agg->arity(), i, dbg), dbg); }
    const Def* extract(const Def* agg, u64 a, u64 i, Debug dbg = {}) { return extract(agg, lit_index(a, i, dbg), dbg); }
    const Def* unsafe_extract(const Def* agg, const Def* i, Debug dbg = {}) { return extract(agg, bitcast(agg->arity(), i, dbg), dbg); }
    const Def* unsafe_extract(const Def* agg, u64 i, Debug dbg = {}) { return unsafe_extract(agg, lit_nat(i, dbg), dbg); }
    //@}
    /// @name Insert
    //@{
    const Def* insert(const Def* agg, const Def* i, const Def* value, Debug dbg = {});
    const Def* insert(const Def* agg, u64 i, const Def* value, Debug dbg = {}) { return insert(agg, lit_index(agg->arity(), i, dbg), value, dbg); }
    const Def* unsafe_insert(const Def* agg, const Def* i, const Def* value, Debug dbg = {}) { return insert(agg, bitcast(agg->arity(), i, dbg), value, dbg); }
    const Def* unsafe_insert(const Def* agg, u64 i, const Def* value, Debug dbg = {}) { return unsafe_insert(agg, lit_nat(i, dbg), value, dbg); }
    //@}
    /// @name LEA - load effective address
    //@{
    const Def* lea(const Def* ptr, const Def* index, Debug dbg);
    const Def* unsafe_lea(const Def* ptr, const Def* index, Debug dbg) { return lea(ptr, bitcast(ptr->type()->as<Ptr>()->pointee()->arity(), index, dbg), dbg); }
    //@}
    /// @name Literal
    //@{
    const Lit* lit(const Def* type, u64 val, Debug dbg = {}) { return unify<Lit>(0, type, val, debug(dbg)); }
    template<class T>
    const Lit* lit(const Def* type, T val, Debug dbg = {}) { return lit(type, thorin::bitcast<u64>(val), dbg); }
    //@}
    /// @name Literal: Arity - note that this is a type
    //@{
    const Lit* lit_arity(u64 a, Debug dbg = {}) { return lit(kind_arity(), a, dbg); }
    const Lit* lit_arity_1() { return cache_.lit_arity_1_; } ///< unit arity 1ₐ
    //@}
    /// @name Literal: Index - the inhabitants of an Arity
    //@{
    const Lit* lit_index(u64 arity, u64 idx, Debug dbg = {}) { return lit_index(lit_arity(arity), idx, dbg); }
    const Lit* lit_index(const Def* arity, u64 index, Debug dbg = {});
    const Lit* lit_index_0_1() { return cache_.lit_index_0_1_; } ///< unit index 0₁ of type unit arity 1ₐ
    //@}
    /// @name Literal: Nat
    //@{
    const Lit* lit_nat(u64 a, Debug dbg = {}) { return lit(type_nat(), a, dbg); }
    //@}
    /// @name Literal: Int, Real
    //@{
    template<class I> const Lit* lit_int(I val, Debug dbg = {}) {
        static_assert(std::is_integral<I>());
        return lit(type_int(sizeof(I)*8), val, dbg);
    }
    template<class R> const Lit* lit_real(R val, Debug dbg = {}) {
        static_assert(std::is_floating_point<R>() || std::is_same<R, r16>());
        return lit(type_real(sizeof(R)*8), val, dbg);
    }
    const Lit* lit_bool(bool val) { return cache_.lit_bool_[size_t(val)]; }
    const Lit* lit_false() { return cache_.lit_bool_[0]; }
    const Lit* lit_true()  { return cache_.lit_bool_[1]; }
    //@}
    /// @name Top/Bottom
    //@{
    const Def* bot_top(bool is_top, const Def* type, Debug dbg = {});
    const Def* bot(const Def* type, Debug dbg = {}) { return bot_top(false, type, dbg); }
    const Def* top(const Def* type, Debug dbg = {}) { return bot_top(true,  type, dbg); }
    const Def* bot_star () { return cache_.bot_star_; }
    const Def* top_arity() { return cache_.top_arity_; } ///< use this guy to encode an unknown arity, e.g., for unsafe arrays
    //@}
    /// @name Variant
    //@{
    const VariantType* variant_type(Defs ops, Debug dbg = {}) { return unify<VariantType>(ops.size(), kind_star(), ops, debug(dbg)); }
    const Def* variant(const VariantType* variant_type, const Def* value, Debug dbg = {}) { return unify<Variant>(1, variant_type, value, debug(dbg)); }
    //@}
    /// @name misc types
    //@{
    const Nat* type_nat() { return cache_.type_nat_; }
    const Mem* type_mem() { return cache_.type_mem_; }
    const Axiom* type_int()  { return cache_.type_int_; }
    const Axiom* type_real() { return cache_.type_real_; }
    const App* type_bool() { return type_int(1); }
    const App* type_int (nat_t num_bits) { return type_int (lit_nat(num_bits)); }
    const App* type_real(nat_t num_bits) { return type_real(lit_nat(num_bits)); }
    const App* type_int (const Def* num_bits) { return app(type_int(),  num_bits)->as<App>(); }
    const App* type_real(const Def* num_bits) { return app(type_real(), num_bits)->as<App>(); }
    const Ptr* type_ptr(const Def* pointee, const Def* addr_space, Debug dbg = {}) { return unify<Ptr>(2, kind_star(), pointee, addr_space, debug(dbg)); }
    const Ptr* type_ptr(const Def* pointee, nat_t addr_space = AddrSpace::Generic, Debug dbg = {}) { return type_ptr(pointee, lit_nat(addr_space), dbg); }
    //@}
    /// @name IOp
    //@{
    template<IOp o> const Axiom* op() { return cache_.IOp_[size_t(o)]; }
    template<IOp o> const Def* op(const Def* a, const Def* b, Debug dbg = {}) { return op<o>(infer_width(a), a, b, dbg); }
    template<IOp o> const Def* op(const Def* w, const Def* a, const Def* b, Debug dbg = {}) { return app(app(op<o>(), w), {a, b}, dbg); }
    //@}
    /// @name WOp
    //@{
    template<WOp o> const Axiom* op() { return cache_.WOp_[size_t(o)]; }
    template<WOp o> const Def* op(flags_t wmode, const Def* a, const Def* b, Debug dbg = {}) { return op<o>(wmode, infer_width(a), a, b, dbg); }
    template<WOp o> const Def* op(flags_t wmode, const Def* w, const Def* a, const Def* b, Debug dbg = {}) { return app(app(op<o>(), {lit_nat(wmode), w}), {a, b}, dbg);
    }
    //@}
    /// @name ZOp
    //@{
    template<ZOp o> const Axiom* op() { return cache_.ZOp_[size_t(o)]; }
    template<ZOp o> const Def* op(const Def* z, const Def* a, const Def* b, Debug dbg = {}) { return op<o>(infer_width(a), z, a, b, dbg); }
    template<ZOp o> const Def* op(const Def* w, const Def* z, const Def* a, const Def* b, Debug dbg = {}) { return app(app(op<o>(), w), {z, a, b}, dbg); }
    //@}
    /// @name ROp
    //@{
    template<ROp o> const Axiom* op() { return cache_.ROp_[size_t(o)]; }
    template<ROp o> const Def* op(const Def* a, const Def* b, Debug dbg = {}) { return op<o>(RMode::none, a, b, dbg); }
    template<ROp o> const Def* op(flags_t rmode, const Def* a, const Def* b, Debug dbg = {}) { return op<o>(rmode, infer_width(a), a, b, dbg); }
    template<ROp o> const Def* op(flags_t rmode, const Def* w, const Def* a, const Def* b, Debug dbg = {}) { return app(app(op<o>(), {lit_nat(rmode), w}), {a, b}, dbg); }
    //@}
    /// @name ICmp
    //@{
    template<ICmp o> const Axiom* op() { return cache_.ICmp_[size_t(o)]; }
    template<ICmp o> const Def* op(const Def* a, const Def* b, Debug dbg = {}) { return op<o>(infer_width(a), a, b, dbg); }
    template<ICmp o> const Def* op(const Def* w, const Def* a, const Def* b, Debug dbg = {}) { return app(app(op<o>(), w), {a, b}, dbg); }
    //@}
    /// @name RCmp
    //@{
    template<RCmp o> const Axiom* op() { return cache_.RCmp_[size_t(o)]; }
    template<RCmp o> const Def* op(const Def* a, const Def* b, Debug dbg = {}) { return op<o>(infer_width(a), a, b, dbg); }
    template<RCmp o> const Def* op(const Def* w, const Def* a, const Def* b, Debug dbg = {}) { return app(app(op<o>(), w), {a, b}, dbg); }
    //@}
    /// @name Conv
    //@{
    template<Conv o> const Axiom* op() { return cache_.Conv_[size_t(o)]; }
    template<Conv o> const Def* op(const Def* dst_type, const Def* src, Debug dbg = {}) {
        auto sw = src->type()->decurry()->arg();
        auto dw =  dst_type  ->decurry()->arg();
        return app(app(op<o>(), {sw, dw}), src, dbg);
    }
    //@}
    /// @name Other casts.
    //@{
    //const App* op_bitcast(const Def* to, const Def* from, Debug dbg = {});
    const Def* bitcast(const Def* to, const Def* from, Debug dbg = {}); ///< deprecated
    //@}
    /// @name memory-related operations
    //@{
    const Def* load(const Def* mem, const Def* ptr, Debug dbg = {});
    const Def* store(const Def* mem, const Def* ptr, const Def* val, Debug dbg = {});
    const Slot* slot(const Def* type, const Def* mem, Debug dbg = {});
    const Alloc* alloc(const Def* type, const Def* mem, Debug dbg = {});
    const Def* global(const Def* id, const Def* init, bool is_mutable = true, Debug dbg = {});
    const Def* global(const Def* init, bool is_mutable = true, Debug dbg = {}) { return global(lit_nat(cur_gid_), init, is_mutable, debug(dbg)); }
    const Def* global_immutable_string(const std::string& str, Debug dbg = {});
    //const Assembly* assembly(const Def* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             //ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});
    //const Assembly* assembly(Defs types, const Def* mem, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             //ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});
    //@}
    /// @name partial evaluation related operations
    //@{
    const Def* hlt(const Def* def, Debug dbg = {});
    const Def* known(const Def* def, Debug dbg = {});
    const Def* run(const Def* def, Debug dbg = {});
    //@}
    /// @name Analyze - used internally for Pass%es
    //@{
    const Analyze* analyze(const Def* type, Defs ops, Debug dbg = {}) { return unify<Analyze>(ops.size(), type, ops, debug(dbg)); }
    const Analyze* analyze(const Def* type, u64 index, const Def* op, Debug dbg = {}) { return analyze(type, {lit_nat(index), op}, dbg); }
    const Analyze* analyze(const Def* type, u64 index, Defs ops, Debug dbg = {}) { return analyze(type, concat((const Def*) lit_nat(index), ops), dbg); }
    //@}
    /// @name misc operations
    //@{
    const Axiom* op_select() const { return cache_.op_select_; }
    const Axiom* op_sizeof() const { return cache_.op_sizeof_; }
    const Def* op_select(const Def* cond, const Def* t, const Def* f, Debug dbg = {}) { return app(app(cache_.op_select_, t->type()), {cond, t, f}, dbg); }
    const Def* op_sizeof(const Def* type, Debug dbg = {}) { return app(op_sizeof(), type, dbg); }
    Lam* match(const Def* type, size_t num_patterns);
    Lam* end_scope() const { return cache_.end_scope_; }
    //@}
    /// @name partial evaluation done?
    //@{
    void mark_pe_done(bool flag = true) { pe_done_ = flag; }
    bool is_pe_done() const { return pe_done_; }
    //@}
    /// @name manage externals
    //@{
    bool empty() { return externals_.empty(); }
    const Externals& externals() const { return externals_; }
    void make_external(Def* def) { externals_.emplace(def->name(), def); }
    void make_internal(Def* def) { externals_.erase(def->name()); }
    bool is_external(const Def* def) { return externals_.contains(def->name()); }
    Def* lookup(const std::string& name) { return externals_.lookup(name).value_or(nullptr); }
    //@}

#if THORIN_ENABLE_CHECKS
    /// @name debugging features
    //@{
    void breakpoint(size_t number) { breakpoints_.insert(number); }
    const Breakpoints& breakpoints() const { return breakpoints_; }
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
        swap(w1.name_,          w2.name_);
        swap(w1.externals_,     w2.externals_);
        swap(w1.defs_,          w2.defs_);
        swap(w1.pe_done_,       w2.pe_done_);
        swap(w1.cache_,         w2.cache_);
#if THORIN_ENABLE_CHECKS
        swap(w1.breakpoints_,   w2.breakpoints_);
        swap(w1.track_history_, w2.track_history_);
#endif
        swap(w1.cache_.universe_->world_, w2.cache_.universe_->world_);
        assert(&w1.universe()->world() == &w1);
        assert(&w2.universe()->world() == &w2);
    }

private:
    /// @name helpers for optional/variant arguments
    //@{
    const Def* name2def(Name n) {
        if (auto s = std::get_if<const char*>(&n)) return tuple_str(*s);
        if (auto s = std::get_if<std::string>(&n)) return tuple_str(s->c_str());
        return std::get<const Def*>(n);
    }
    const Def* debug(Debug dbg) {
        if (auto d  = std::get_if<0>(&*dbg)) {
            auto n  = name2def(std::get<0>(*d));
            auto f  = name2def(std::get<1>(*d));
            auto fl = lit_nat(std::get<2>(*d));
            auto fc = lit_nat(std::get<3>(*d));
            auto bl = lit_nat(std::get<4>(*d));
            auto bc = lit_nat(std::get<5>(*d));
            return tuple({n, f, fl, fc, bl, bc});
        }
        return std::get<const Def*>(*dbg);
    }
    //const Def* filter(Filter f) { return f ? *f : lit_false(); }
    //@}

    /// @name memory management and hashing
    //@{
    template<class T, class... Args>
    const T* unify(size_t num_ops, Args&&... args) {
        auto def = allocate<T>(num_ops, args...);
#ifndef NDEBUG
        if (breakpoints_.contains(def->gid())) THORIN_BREAK;
#endif
        assert(!def->isa_nominal());
        auto [i, success] = defs_.emplace(def);
        if (success) {
            def->finalize();
            return def;
        }

        deallocate<T>(def);
        return static_cast<const T*>(*i);
    }

    template<class T, class... Args>
    T* insert(size_t num_ops, Args&&... args) {
        auto def = allocate<T>(num_ops, args...);
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
        Lock() { assert((allocate_guard_ = !allocate_guard_) && "you are not allowed to recursively invoke allocate"); }
        ~Lock() { allocate_guard_ = !allocate_guard_; }
        static bool allocate_guard_;
    };
#else
    struct Lock { ~Lock() {} };
#endif

    static inline size_t align(size_t n) { return (n + (sizeof(void*) - 1)) & ~(sizeof(void*)-1); }

    template<class T> static inline size_t num_bytes_of(size_t num_ops) {
        size_t result = sizeof(Def) + sizeof(const Def*)*num_ops;
        return align(result);
    }

    template<class T, class... Args>
    T* allocate(size_t num_ops, Args&&... args) {
        static_assert(sizeof(Def) == sizeof(T), "you are not allowed to introduce any additional data in subclasses of Def");
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
    void deallocate(const T* def) {
        size_t num_bytes = num_bytes_of<T>(def->num_ops());
        num_bytes = align(num_bytes);
        def->~T();
        if (ptrdiff_t(buffer_index_ - num_bytes) > 0) // don't care otherwise
            buffer_index_-= num_bytes;
        assert(buffer_index_ % alignof(T) == 0);
    }
    //@}

    std::unique_ptr<Zone> root_page_;
    Zone* cur_page_;
    size_t buffer_index_ = 0;
    std::string name_;
    Externals externals_;
    Sea defs_;
    uint32_t cur_gid_;
    bool pe_done_ = false;
#if THORIN_ENABLE_CHECKS
    bool track_history_ = false;
    Breakpoints breakpoints_;
#endif
    struct Cache {
        Universe* universe_;
        const KindArity* kind_arity_;
        const KindMulti* kind_multi_;
        const KindStar*  kind_star_;
        const Bot* bot_star_;
        const Top* top_arity_;
        const Sigma* sigma_;
        const Tuple* tuple_;
        const Nat* type_nat_;
        const Mem* type_mem_;
        std::array<const Lit*, 2> lit_bool_;
        const Lit* lit_arity_1_;
        const Lit* lit_index_0_1_;
        Lam* end_scope_;
        Axiom* type_int_;
        Axiom* type_real_;
        std::array<Axiom*, Num<IOp>>  IOp_;
        std::array<Axiom*, Num<WOp>>  WOp_;
        std::array<Axiom*, Num<ZOp>>  ZOp_;
        std::array<Axiom*, Num<ROp>>  ROp_;
        std::array<Axiom*, Num<ICmp>> ICmp_;
        std::array<Axiom*, Num<RCmp>> RCmp_;
        std::array<Axiom*, Num<Conv>> Conv_;
        Axiom* op_select_;
        Axiom* op_sizeof_;
    } cache_;

    friend class Cleaner;
    friend void Def::replace(Tracker) const;
};

}

#endif
