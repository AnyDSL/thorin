#ifndef THORIN_WORLD_H
#define THORIN_WORLD_H

#include <cassert>
#include <iostream>
#include <functional>
#include <initializer_list>
#include <string>

#include "thorin/axiom.h"
#include "thorin/tuple.h"
#include "thorin/union.h"
#include "thorin/util/hash.h"
#include "thorin/config.h"

namespace thorin {

enum class LogLevel { Debug, Verbose, Info, Warn, Error };

class Checker;
class DepNode;
class ErrorHandler;
class RecStreamer;
class Scope;

inline const Def* infer_size(const Def* def) { return isa_sized_type(def->type()); }

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
 *  - several local optimizations like <code>x + 0 -> x</code>
 *
 *  Use @p cleanup to remove dead and unreachable code.
 *
 *  You can create several worlds.
 *  All worlds are completely independent from each other.
 *
 *  Note that types are also just @p Def%s and will be hashed as well.
 */
class World : public Streamable<World> {
public:
    struct SeaHash {
        static hash_t hash(const Def* def) { return def->hash(); }
        static bool eq(const Def* def1, const Def* def2) { return def1->equal(def2); }
        static const Def* sentinel() { return (const Def*)(1); }
    };

    struct BreakHash {
        static hash_t hash(size_t i) { return i; }
        static bool eq(size_t i1, size_t i2) { return i1 == i2; }
        static size_t sentinel() { return size_t(-1); }
    };

    struct ExternalsHash {
        static hash_t hash(const std::string& s) { return thorin::hash(s.c_str()); }
        static bool eq(const std::string& s1, const std::string& s2) { return s1 == s2; }
        static std::string sentinel() { return std::string(); }
    };

    using Sea         = HashSet<const Def*, SeaHash>;///< This @p HashSet contains Thorin's "sea of nodes".
    using Breakpoints = HashSet<size_t, BreakHash>;
    using Externals   = HashMap<std::string, Def*, ExternalsHash>;

    World(World&&) = delete;
    World& operator=(const World&) = delete;

    explicit World(const std::string& name = {});
    ///  Inherits the @p state_ of the @p other @p World but does @em not perform a copy.
    explicit World(const World& other)
        : World(other.name())
    {
        state_ = other.state_;
    }
    ~World();

    /// @ getters
    //@{
    const std::string& name() const { return data_.name_; }
    const Sea& defs() const { return data_.defs_; }
    std::vector<Lam*> copy_lams() const; // TODO remove this
    //@}
    /// @name manage global identifier - a unique number for each Def
    //@{
    u32 cur_gid() const { return state_.cur_gid; }
    u32 next_gid() { return ++state_.cur_gid; }
    //@}
    /// @name Space and Kind
    //@{
    const Space* space() { return data_.space_;   }
    const Kind* kind() { return data_.kind_; }
    //@}
    /// @name Param
    //@{
    const Param* param(const Def* type, Def* nominal, const Def* dbg = {}) { return unify<Param>(1, type, nominal, dbg); }
    //@}
    /// @name Axiom
    //@{
    const Axiom* axiom(Def::NormalizeFn normalize, const Def* type, tag_t tag, flags_t flags, const Def* dbg = {}) {
        return unify<Axiom>(0, normalize, type, tag, flags, dbg);
    }
    const Axiom* axiom(const Def* type, tag_t tag, flags_t flags, const Def* dbg = {}) { return axiom(nullptr, type, tag, flags, dbg); }
    //@}
    /// @name Pi
    //@{
    const Pi* pi(const Def* domain, const Def* codomain, const Def* dbg = {});
    const Pi* pi(Defs domain, const Def* codomain, const Def* dbg = {}) { return pi(sigma(domain), codomain, dbg); }
    Pi* nom_pi(const Def* type, const Def* dbg = {}) { return insert<Pi>(2, type, dbg); } ///< @em nominal Pi.
    //@}
    /// @name Pi: continuation type, i.e., Pi type with codomain Bottom
    //@{
    const Pi* cn() { return cn(sigma()); }
    const Pi* cn(const Def* domain, const Def* dbg = {}) { return pi(domain, bot_kind(), dbg); }
    const Pi* cn(Defs domains, const Def* dbg = {}) { return cn(sigma(domains), dbg); }
    /// Same as cn/pi but adds a mem parameter to each pi
    const Pi* cn_mem(const Def* domain, const Def* dbg = {}) { return cn(sigma({ type_mem(), domain }), dbg); }
    const Pi* pi_mem(const Def* domain, const Def* codomain, const Def* dbg = {}) { auto d = sigma({type_mem(), domain}); return pi(d, sigma({type_mem(), codomain}), dbg); }
    const Pi* fn_mem(const Def* domain, const Def* codomain, const Def* dbg = {}) { return cn({type_mem(), domain, cn_mem(codomain)}, dbg); }
    //@}
    /// @name Lam%bda
    //@{
    Lam* nom_lam(const Pi* cn, Lam::CC cc = Lam::CC::C, const Def* dbg = {}) {
        auto lam = insert<Lam>(2, cn, cc, dbg);
        return lam;
    }
    Lam* nom_lam(const Pi* cn, const Def* dbg = {}) { return nom_lam(cn, Lam::CC::C, dbg); }
    const Lam* lam(const Def* domain, const Def* filter, const Def* body, const Def* dbg);
    const Lam* lam(const Def* domain, const Def* body, const Def* dbg) { return lam(domain, lit_true(), body, dbg); }
    //@}
    /// @name App
    //@{
    const Def* app(const Def* callee, const Def* arg, const Def* dbg = {});
    const Def* app(const Def* callee, Defs args, const Def* dbg = {}) { return app(callee, tuple(args), dbg); }
    const Def* raw_app(const Def* callee, const Def* arg, const Def* dbg = {});                                         /// Same as @p app but does @em not apply @p NormalizeFn.
    const Def* raw_app(const Def* callee, Defs args, const Def* dbg = {}) { return raw_app(callee, tuple(args), dbg); } /// Same as @p app but does @em not apply @p NormalizeFn.
    //@}
    /// @name Sigma
    //@{
    Sigma* nom_sigma(const Def* type, size_t size, const Def* dbg = {}) { return insert<Sigma>(size, type, size, dbg); }
    Sigma* nom_sigma(size_t size, const Def* dbg = {}) { return nom_sigma(kind(), size, dbg); } ///< a @em nominal @p Sigma of type @p kind
    const Def* sigma(const Def* type, Defs ops, const Def* dbg = {});
    /// a @em structural @p Sigma of type @p kind
    const Def* sigma(Defs ops, const Def* dbg = {}) { return sigma(kind(), ops, dbg); }
    const Sigma* sigma() { return data_.sigma_; } ///< the unit type within @p kind()
    //@}
    /// @name Arr
    //@{
    Arr* nom_arr(const Def* type, const Def* shape, const Def* dbg = {}) { return insert<Arr>(2, type, shape, dbg); }
    Arr* nom_arr(const Def* shape, const Def* dbg = {}) { return nom_arr(kind(), shape, dbg); }
    const Def* arr(const Def* shape, const Def* body, const Def* dbg = {});
    const Def* arr(Defs shape, const Def* body, const Def* dbg = {});
    const Def* arr(u64 n, const Def* body, const Def* dbg = {}) { return arr(lit_nat(n), body, dbg); }
    const Def* arr(ArrayRef<u64> shape, const Def* body, const Def* dbg = {}) {
        return arr(Array<const Def*>(shape.size(), [&](size_t i) { return lit_nat(shape[i], dbg); }), body, dbg);
    }
    const Def* arr_unsafe(const Def* body, const Def* dbg = {}) { return arr(top_nat(), body, dbg); }
    //@}
    /// @name Tuple
    //@{
    /// ascribes @p type to this tuple - needed for dependently typed and structural @p Sigma%s
    const Def* tuple(const Def* type, Defs ops, const Def* dbg = {});
    const Def* tuple(Defs ops, const Def* dbg = {});
    const Def* tuple_str(const char* s, const Def* = {});
    const Def* tuple_str(const std::string& s, const Def* dbg = {}) { return tuple_str(s.c_str(), dbg); }
    const Tuple* tuple() { return data_.tuple_; } ///< the unit value of type <code>[]</code>
    //@}
    /// @name Pack
    //@{
    const Def* pack(const Def* arity, const Def* body, const Def* dbg = {});
    const Def* pack(Defs shape, const Def* body, const Def* dbg = {});
    const Def* pack(u64 n, const Def* body, const Def* dbg = {}) { return pack(lit_nat(n), body, dbg); }
    const Def* pack(ArrayRef<u64> shape, const Def* body, const Def* dbg = {}) {
        return pack(Array<const Def*>(shape.size(), [&](auto i) { return lit_nat(shape[i], dbg); }), body, dbg);
    }
    //@}
    /// @name Extract
    //@{
    /// During a rebuild we cannot infer the type if it is not set yet; in this case we rely on @p ex_type.
    const Def* extract_(const Def* ex_type, const Def* tup, const Def* i, const Def* dbg = {});
    const Def* extract(const Def* tup, const Def* i, const Def* dbg = {}) { return extract_(nullptr, tup,             i, dbg); }
    const Def* extract(const Def* tup, u64 a, u64 i, const Def* dbg = {}) { return extract_(nullptr, tup, lit_int(a, i), dbg); }
    const Def* extract(const Def* tup,        u64 i, const Def* dbg = {}) { return extract(tup, as_lit(tup->arity()), i, dbg); }
    const Def* extract_unsafe(const Def* tup, u64 i, const Def* dbg = {}) { return extract_unsafe(tup, lit_int(0, i), dbg); }
    const Def* extract_unsafe(const Def* tup, const Def* i, const Def* dbg = {}) {
        return extract(tup, op(Conv::u2u, type_int(as_lit(tup->type()->reduce()->arity())), i, dbg), dbg);
    }
    //@}
    /// @name Insert
    //@{
    const Def* insert(const Def* tup, const Def* i, const Def* value, const Def* dbg = {});
    const Def* insert(const Def* tup, u64 a, u64 i, const Def* value, const Def* dbg = {}) { return insert(tup,           lit_int(a, i), value, dbg); }
    const Def* insert(const Def* tup,        u64 i, const Def* value, const Def* dbg = {}) { return insert(tup, as_lit(tup->arity()), i, value, dbg); }
    const Def* insert_unsafe(const Def* tup, u64 i, const Def* value, const Def* dbg = {}) { return insert_unsafe(tup, lit_int(0, i), value, dbg); }
    const Def* insert_unsafe(const Def* tup, const Def* i, const Def* value, const Def* dbg = {}) {
        return insert(tup, op(Conv::u2u, type_int(as_lit(tup->type()->reduce()->arity())), i), value, dbg);
    }
    //@}
    /// @name Union, Which, Match, Case, Ptrn
    //@{
    Union* nom_union(const Def* type, size_t size, const Def* dbg = {}) { return insert<Union>(size, type, size, dbg); }
    Union* nom_union(size_t size, const Def* dbg = {}) { return nom_union(kind(), size, dbg); } ///< a @em nominal @p Sigma of type @p kind
    const Def* union_(const Def* type, Defs ops, const Def* dbg = {});
    /// a @em structural @p Union of type @p kind
    const Def* union_(Defs ops, const Def* dbg = {}) { return union_(kind(), ops, dbg); }
    const Def* which(const Def* value, const Def* dbg = {});
    const Def* match(const Def* val, Defs ptrns, const Def* dbg = {});
    const Case* case_(const Def* domain, const Def* codomain, const Def* dbg = {}) { return unify<Case>(2, kind(), domain, codomain, dbg); }
    Ptrn* nom_ptrn(const Case* type, const Def* dbg = {}) { return insert<Ptrn>(2, type, dbg); }
    //@}
    /// @name Lit
    //@{
    const Lit* lit(const Def* type, u64 val, const Def* dbg = {}) { assert(type->level() == Sort::Type); return unify<Lit>(0, type, val, dbg); }
    //@}
    /// @name Lit: Nat
    //@{
    const Lit* lit_nat(nat_t a, const Def* dbg = {}) { return lit(type_nat(), a, dbg); }
    //@}
    /// @name Lit: Int
    //@{
    const Lit* lit_int(const Def* type, u64 val, const Def* dbg);
    const Lit* lit_int(nat_t bound, u64 val, const Def* dbg = {}) { return lit_int(type_int(bound), val, dbg); }
    const Lit* lit_int_mod(nat_t bound, u64 val, const Def* dbg = {}) { return lit_int(type_int(bound), bound == 0 ? val : (val % bound), dbg); }
    const Lit* lit_int_width(nat_t width, u64 val, const Def* dbg = {}) { return lit_int(type_int_width(width), val, dbg); }
    template<class I> const Lit* lit_int(I val, const Def* dbg = {}) {
        static_assert(std::is_integral<I>());
        return lit_int(type_int(width2bound(sizeof(I)*8)), val, dbg);
    }
    const Lit* lit_bool(bool val) { return data_.lit_bool_[size_t(val)]; }
    const Lit* lit_false() { return data_.lit_bool_[0]; }
    const Lit* lit_true()  { return data_.lit_bool_[1]; }
    //@}
    /// @name Lit: Real
    //@{
    const Lit* lit_real(nat_t width, r64 val, const Def* dbg = {}) {
        switch (width) {
            case 16: assert(r64(r16(val)) == val && "loosing precision"); return lit_real(r16(val), dbg);
            case 32: assert(r64(r32(val)) == val && "loosing precision"); return lit_real(r32(val), dbg);
            case 64: assert(r64(r64(val)) == val && "loosing precision"); return lit_real(r64(val), dbg);
            default: THORIN_UNREACHABLE;
        }
    }

    template<class R> const Lit* lit_real(R val, const Def* dbg = {}) {
        static_assert(std::is_floating_point<R>() || std::is_same<R, r16>());
        if constexpr (false) {}
        else if (sizeof(R) == 2) return lit(type_real(16), thorin::bitcast<u16>(val), dbg);
        else if (sizeof(R) == 4) return lit(type_real(32), thorin::bitcast<u32>(val), dbg);
        else if (sizeof(R) == 8) return lit(type_real(64), thorin::bitcast<u64>(val), dbg);
        else THORIN_UNREACHABLE;
    }
    //@}
    /// @name Top/Bottom
    //@{
    const Def* bot_top(bool is_top, const Def* type, const Def* dbg = {});
    const Def* bot(const Def* type, const Def* dbg = {}) { return bot_top(false, type, dbg); }
    const Def* top(const Def* type, const Def* dbg = {}) { return bot_top(true,  type, dbg); }
    const Def* bot_kind() { return data_.bot_kind_; }
    const Def* top_kind() { return data_.top_kind_; }
    const Def* top_nat () { return data_.top_nat_; }
    //@}
    /// @name misc types
    //@{
    const Nat* type_nat()    { return data_.type_nat_; }
    const Axiom* type_mem()  { return data_.type_mem_; }
    const Axiom* type_int()  { return data_.type_int_; }
    const Axiom* type_real() { return data_.type_real_; }
    const Axiom* type_ptr()  { return data_.type_ptr_; }
    const App* type_bool() { return data_.type_bool_; }
    const App* type_int_width(nat_t width) { return type_int(lit_nat(width2bound(width))); }
    const App* type_int (nat_t bound) { return type_int (lit_nat(bound)); }
    const App* type_real(nat_t width) { return type_real(lit_nat(width)); }
    const App* type_int (const Def* bound) { return app(type_int(),  bound)->as<App>(); }
    const App* type_real(const Def* width) { return app(type_real(), width)->as<App>(); }
    const App* type_ptr(const Def* pointee, nat_t addr_space = AddrSpace::Generic, const Def* dbg = {}) { return type_ptr(pointee, lit_nat(addr_space), dbg); }
    const App* type_ptr(const Def* pointee, const Def* addr_space, const Def* dbg = {}) { return app(type_ptr(), {pointee, addr_space}, dbg)->as<App>(); }
    //@}
    /// @name Bit
    //@{
    const Axiom* op(Bit o) const { return data_.Bit_[size_t(o)]; }
    const Def* op(Bit o, const Def* a, const Def* b, const Def* dbg = {}) { auto w = infer_size(a); return app(app(op(o), w), {a, b}, dbg); }
    const Def* op_neg(const Def* a, const Def* dbg = {}) { auto w = as_lit(isa_sized_type(a->type())); return op(Bit::_xor, lit_int(w, w-1_u64 ), a, dbg); }
    //@}
    /// @name Shr
    //@{
    const Axiom* op(Shr o) { return data_.Shr_[size_t(o)]; }
    const Def* op(Shr o, const Def* a, const Def* b, const Def* dbg = {}) { auto w = infer_size(a); return app(app(op(o), w), {a, b}, dbg); }
    //@}
    /// @name Wrap
    //@{
    const Axiom* op(Wrap o) { return data_.Wrap_[size_t(o)]; }
    const Def* op(Wrap o, const Def* wmode, const Def* a, const Def* b, const Def* dbg = {}) {
        auto w = infer_size(a);
        return app(app(op(o), {wmode, w}), {a, b}, dbg);
    }
    const Def* op(Wrap o, nat_t wmode, const Def* a, const Def* b, const Def* dbg = {}) { return op(o, lit_nat(wmode), a, b, dbg); }
    const Def* op_wminus(nat_t wmode, const Def* a, const Def* dbg = {}) { auto w = as_lit(isa_sized_type(a->type())); return op(Wrap::sub, wmode, lit_int(w, 0), a, dbg); }
    //@}
    /// @name Div
    //@{
    const Axiom* op(Div o) { return data_.Div_[size_t(o)]; }
    const Def* op(Div o, const Def* mem, const Def* a, const Def* b, const Def* dbg = {}) {
        auto w = infer_size(a);
        auto [m, x] = app(app(op(o), w), {mem, a, b}, dbg)->split<2>();
        return tuple({m, x});
    }
    //@}
    /// @name ROp
    //@{
    const Axiom* op(ROp o) { return data_.ROp_[size_t(o)]; }
    const Def* op(ROp o, nat_t rmode, const Def* a, const Def* b, const Def* dbg = {}) { return op(o, lit_nat(rmode), a, b, dbg); }
    const Def* op(ROp o, const Def* rmode, const Def* a, const Def* b, const Def* dbg = {}) { auto w = infer_size(a); return app(app(op(o), {rmode, w}), {a, b}, dbg); }
    const Def* op_rminus(const Def* rmode, const Def* a, const Def* dbg = {}) { auto w = as_lit(isa_sized_type(a->type())); return op(ROp::sub, rmode, lit_real(w, -0.0), a, dbg); }
    const Def* op_rminus(nat_t rmode, const Def* a, const Def* dbg = {}) { return op_rminus(lit_nat(rmode), a, dbg); }
    //@}
    /// @name Cmp
    //@{
    const Axiom* op(ICmp o) { return data_.ICmp_[size_t(o)]; }
    const Axiom* op(RCmp o) { return data_.RCmp_[size_t(o)]; }
    const Def* op(ICmp o, const Def* a, const Def* b, const Def* dbg = {}) { auto w = infer_size(a); return app(app(op(o), w), {a, b}, dbg); }
    const Def* op(RCmp o, nat_t rmode, const Def* a, const Def* b, const Def* dbg = {}) { return op(o, lit_nat(rmode), a, b, dbg); }
    const Def* op(RCmp o, const Def* rmode, const Def* a, const Def* b, const Def* dbg = {}) { auto w = infer_size(a); return app(app(op(o), {rmode, w}), {a, b}, dbg); }
    //@}
    /// @name Type Traits
    //@{
    const Def* op(Trait o) { return data_.Trait_[size_t(o)]; }
    const Def* op(Trait o, const Def* type, const Def* dbg = {}) { return app(op(o), type, dbg); }
    //@}
    /// @name Casts
    //@{
    const Axiom* op(Conv o) { return data_.Conv_[size_t(o)]; }
    const Def* op(Conv o, const Def* dst_type, const Def* src, const Def* dbg = {}) {
        auto d = dst_type   ->as<App>()->arg();
        auto s = src->type()->as<App>()->arg();
        return app(app(op(o), {d, s}), src, dbg);
    }
    //@}
    /// @name PE - partial evaluation related operations
    //@{
    const Def* op(PE o) { return data_.PE_[size_t(o)]; }
    const Def* op(PE o, const Def* def, const Def* dbg = {}) { return app(app(op(o), def->type()), def, dbg); }
    //@}
    /// @name Acc
    //@{
    const Axiom* op(Acc o) { return data_.Acc_[size_t(o)]; }
    const Def* op(Acc o, const Def* a, const Def* b, const Def* body, const Def* dbg = {}) { return app(op(o), {a, b, body}, dbg); }
    //@}
    /// @name bitcasts
    //@{
    const Axiom* op_bitcast() const { return data_.op_bitcast_; }
    const Def* op_bitcast(const Def* dst_type, const Def* src, const Def* dbg = {}) { return app(app(op_bitcast(), {dst_type, src->type()}), src, dbg); }
    //@}
    /// @name memory-related operations
    //@{
    const Def* op_load()  { return data_.op_load_;  }
    const Def* op_store() { return data_.op_store_; }
    const Def* op_slot()  { return data_.op_slot_;  }
    const Def* op_alloc() { return data_.op_alloc_; }
    const Def* op_load (const Def* mem, const Def* ptr, const Def* dbg = {})                 { auto [T, a] = as<Tag::Ptr>(ptr->type())->args<2>(); return app(app(op_load (), {T, a}), {mem, ptr},      dbg); }
    const Def* op_store(const Def* mem, const Def* ptr, const Def* val, const Def* dbg = {}) { auto [T, a] = as<Tag::Ptr>(ptr->type())->args<2>(); return app(app(op_store(), {T, a}), {mem, ptr, val}, dbg); }
    const Def* op_alloc(const Def* type, const Def* mem, const Def* dbg = {}) { return app(app(op_alloc(), {type, lit_nat(0)}), mem, dbg); }
    const Def* op_slot (const Def* type, const Def* mem, const Def* dbg = {}) { return app(app(op_slot(), {type, lit_nat(0)}), {mem, lit_nat(cur_gid())}, dbg); }
    const Def* global(const Def* id, const Def* init, bool is_mutable = true, const Def* dbg = {});
    const Def* global(const Def* init, bool is_mutable = true, const Def* dbg = {}) { return global(lit_nat(state_.cur_gid), init, is_mutable, dbg); }
    const Def* global_immutable_string(const std::string& str, const Def* dbg = {});
    //@}
    /// @name make atomic operations
    //@{
    const Def* op_atomic() { return data_.op_atomic_; }
    const Def* op_atomic(const Def* fn, const Def* dbg = {}) { return app(op_atomic(), fn, dbg); }
    const Def* op_atomic(const Def* fn, Defs args, const Def* dbg = {}) { return app(op_atomic(fn), args, dbg); }
    //@}
    /// @name Proxy - used internally for Pass%es
    //@{
    const Proxy* proxy(const Def* type, Defs ops, tag_t index, flags_t flags, const Def* dbg = {}) {
        return unify<Proxy>(ops.size(), type, ops, index, flags, dbg);
    }
    //@}
    /// @name misc operations
    //@{
    const Axiom* op_lea() const { return data_.op_lea_; }
    const Def* op_lea(const Def* ptr, const Def* index, const Def* dbg = {});
    const Def* op_lea_unsafe(const Def* ptr, const Def* i, const Def* dbg = {}) {
        auto safe_int = type_int(as<Tag::Ptr>(ptr->type())->arg(0)->arity());
        return op_lea(ptr, op(Conv::u2u, safe_int, i), dbg);
    }
    const Def* op_lea_unsafe(const Def* ptr, u64 i, const Def* dbg = {}) { return op_lea_unsafe(ptr, lit_int(i), dbg); }
    const Def* dbg(Debug);
    //@}
    /// @name AD
    //@{
    const Def* op_grad(const Def* fn, const Def* dbg = {});
    const Def* type_tangent_vector(const Def* primal_type, const Def* dbg = {});
    //@}
    /// @name partial evaluation done?
    //@{
    void mark_pe_done(bool flag = true) { state_.pe_done = flag; }
    bool is_pe_done() const { return state_.pe_done; }
    //@}
    /// @name manage externals
    //@{
    bool empty() { return data_.externals_.empty(); }
    const Externals& externals() const { return data_.externals_; }
    void make_external(Def* def) { data_.externals_.emplace(def->debug().name, def); }
    void make_internal(Def* def) { data_.externals_.erase(def->debug().name); }
    bool is_external(const Def* def) { return data_.externals_.contains(def->debug().name); }
    Def* lookup(const std::string& name) { return data_.externals_.lookup(name).value_or(nullptr); }
    //@}
    /// @name visit
    //@{
    /**
     * Transitively visits all @em reachable Scope%s in this @p World that do not have free variables.
     * We call these Scope%s @em top-level Scope%s.
     * Select with @p elide_empty whether you want to visit trivial @p Scope%s of @em nominals without body.
     */
    using VisitFn = std::function<void(const Scope&)>;
    template<bool elide_empty = true> void visit(VisitFn) const;
#if THORIN_ENABLE_CHECKS
    /// @name debugging features
    //@{
    void     breakpoint(size_t number) { state_.    breakpoints.insert(number); }
    void use_breakpoint(size_t number) { state_.use_breakpoints.insert(number); }
    bool track_history() const { return state_.track_history; }
    void enable_history(bool flag = true) { state_.track_history = flag; }
    const Def* gid2def(u32 gid);
    //@}
#endif
    /// @name logging
    //@{
    Stream& stream() { assert(state_.stream); return *state_.stream; }
    LogLevel min_level() const { return state_.min_level; }

    void set(LogLevel min_level) { state_.min_level = min_level; }
    void set(Stream& stream) { state_.stream = &stream; }
    void set(LogLevel min_level, Stream& stream) { set(min_level); set(stream); }

    template<class... Args>
    void log(LogLevel level, Loc loc, const char* fmt, Args&&... args) {
        if (state_.stream != nullptr && int(min_level()) <= int(level)) {
            stream().fmt("{}:{}: ", colorize(level2string(level), level2color(level)), colorize(loc.to_string(), 7));
            stream().fmt(fmt, std::forward<Args&&>(args)...).endl().flush();
        }
    }

    template<class... Args>
    [[noreturn]] void error(Loc loc, const char* fmt, Args&&... args) {
        log(LogLevel::Error, loc, fmt, std::forward<Args&&>(args)...);
        std::abort();
    }

    template<class... Args> void idef(const Def* def, const char* fmt, Args&&... args) { log(LogLevel::Info, def->debug().loc, fmt, std::forward<Args&&>(args)...); }
    template<class... Args> void wdef(const Def* def, const char* fmt, Args&&... args) { log(LogLevel::Warn, def->debug().loc, fmt, std::forward<Args&&>(args)...); }
    template<class... Args> void edef(const Def* def, const char* fmt, Args&&... args) { error(def->debug().loc, fmt, std::forward<Args&&>(args)...); }

    static const char* level2string(LogLevel level);
    static int level2color(LogLevel level);
    static std::string colorize(const std::string& str, int color);
    //@}
    /// @name stream
    //@{
    Stream& stream(Stream&) const;
    Stream& stream(RecStreamer&, const DepNode*) const;
    void debug_stream(); ///< Stream thorin IR if @p min_level is @p LogLevel::Debug.
    ///@}
    /// @name error handling
    //@{
    void set(std::unique_ptr<ErrorHandler>&& err);
    ErrorHandler* err() { return err_.get(); }
    //@}

    friend void swap(World& w1, World& w2) {
        using std::swap;
        swap(w1.arena_,  w2.arena_);
        swap(w1.state_,  w2.state_);
        swap(w1.data_,   w2.data_);
        swap(w1.err_,    w2.err_);

        swap(w1.data_.space_->world_, w2.data_.space_->world_);
        assert(&w1.space()->world() == &w1);
        assert(&w2.space()->world() == &w2);
    }

private:
    /// @name put into sea of nodes
    //@{
    template<class T, class... Args>
    const T* unify(size_t num_ops, Args&&... args) {
        auto def = arena_.allocate<T>(num_ops, args...);
#ifndef NDEBUG
        if (state_.breakpoints.contains(def->gid())) THORIN_BREAK;
#endif
        assert(!def->isa_nominal());
        auto [i, inserted] = data_.defs_.emplace(def);
        if (inserted) {
#ifndef NDEBUG
            for (auto op : def->ops()) {
                if (state_.use_breakpoints.contains(op->gid())) THORIN_BREAK;
            }
#endif
            def->finalize();
            return def;
        }

        arena_.deallocate<T>(def);
        return static_cast<const T*>(*i);
    }

    template<class T, class... Args>
    T* insert(size_t num_ops, Args&&... args) {
        auto def = arena_.allocate<T>(num_ops, args...);
#ifndef NDEBUG
        if (state_.breakpoints.contains(def->gid())) THORIN_BREAK;
#endif
        auto p = data_.defs_.emplace(def);
        assert_unused(p.second);
        return def;
    }
    //@}

    class Arena {
    public:
        Arena()
            : root_zone_(new Zone) // don't use 'new Zone()' - we keep the allocated Zone uninitialized
            , cur_zone_(root_zone_.get())
        {}

        struct Zone {
            static const size_t Size = 1024 * 1024 - sizeof(std::unique_ptr<int>); // 1MB - sizeof(next)
            char buffer[Size];
            std::unique_ptr<Zone> next;
        };

#ifndef NDEBUG
        struct Lock {
            Lock() { assert((guard_ = !guard_) && "you are not allowed to recursively invoke allocate"); }
            ~Lock() { guard_ = !guard_; }
            static bool guard_;
        };
#else
        struct Lock { ~Lock() {} };
#endif
        template<class T, class... Args>
        T* allocate(size_t num_ops, Args&&... args) {
            static_assert(sizeof(Def) == sizeof(T), "you are not allowed to introduce any additional data in subclasses of Def");
            Lock lock;
            size_t num_bytes = num_bytes_of<T>(num_ops);
            num_bytes = align(num_bytes);
            assert(num_bytes < Zone::Size);

            if (buffer_index_ + num_bytes >= Zone::Size) {
                auto zone = new Zone;
                cur_zone_->next.reset(zone);
                cur_zone_ = zone;
                buffer_index_ = 0;
            }

            auto result = new (cur_zone_->buffer + buffer_index_) T(args...);
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

        static constexpr inline size_t align(size_t n) { return (n + (sizeof(void*) - 1)) & ~(sizeof(void*)-1); }

        template<class T> static constexpr inline size_t num_bytes_of(size_t num_ops) {
            size_t result = sizeof(Def) + sizeof(const Def*)*num_ops;
            return align(result);
        }

    private:
        std::unique_ptr<Zone> root_zone_;
        Zone* cur_zone_;
        size_t buffer_index_ = 0;
    } arena_;

    struct State {
        Stream* stream = nullptr;
        LogLevel min_level = LogLevel::Error;
        u32 cur_gid = 0;
        bool pe_done = false;
#if THORIN_ENABLE_CHECKS
        bool track_history = false;
        Breakpoints breakpoints;
        Breakpoints use_breakpoints;
#endif
    } state_;

    struct Data {
        Space* space_;
        const Kind* kind_;
        const Bot* bot_kind_;
        const Top* top_kind_;
        const App* type_bool_;
        const Top* top_nat_;
        const Sigma* sigma_;
        const Tuple* tuple_;
        const Nat* type_nat_;
        const Def* table_id;
        const Def* table_not;
        std::array<const Lit*, 2> lit_bool_;
        std::array<const Axiom*, Num<Bit  >> Bit_;
        std::array<const Axiom*, Num<Shr  >> Shr_;
        std::array<const Axiom*, Num<Wrap >> Wrap_;
        std::array<const Axiom*, Num<Div  >> Div_;
        std::array<const Axiom*, Num<ROp  >> ROp_;
        std::array<const Axiom*, Num<ICmp >> ICmp_;
        std::array<const Axiom*, Num<RCmp >> RCmp_;
        std::array<const Axiom*, Num<Trait>> Trait_;
        std::array<const Axiom*, Num<Conv >> Conv_;
        std::array<const Axiom*, Num<PE   >> PE_;
        std::array<const Axiom*, Num<Acc  >> Acc_;
        const Axiom* op_alloc_;
        const Axiom* op_atomic_;
        const Axiom* op_bitcast_;
        const Axiom* op_grad_;
        const Axiom* op_lea_;
        const Axiom* op_load_;
        const Axiom* op_slot_;
        const Axiom* op_store_;
        const Axiom* type_int_;
        const Axiom* type_mem_;
        const Axiom* type_ptr_;
        const Axiom* type_real_;
        const Axiom* type_tangent_vector_;
        std::string name_;
        Externals externals_;
        Sea defs_;
        DefDefMap<Array<const Def*>> cache_;
    } data_;

    std::unique_ptr<ErrorHandler> err_;
    std::unique_ptr<Checker> checker_;

    friend class Cleaner;
    friend Array<const Def*> Def::apply(const Def*);
    friend void Def::replace(Tracker) const;
};

#define ELOG(...) log(thorin::LogLevel::Error,   Loc(__FILE__, {__LINE__, u32(-1)}, {__LINE__, u32(-1)}), __VA_ARGS__)
#define WLOG(...) log(thorin::LogLevel::Warn,    Loc(__FILE__, {__LINE__, u32(-1)}, {__LINE__, u32(-1)}), __VA_ARGS__)
#define ILOG(...) log(thorin::LogLevel::Info,    Loc(__FILE__, {__LINE__, u32(-1)}, {__LINE__, u32(-1)}), __VA_ARGS__)
#define VLOG(...) log(thorin::LogLevel::Verbose, Loc(__FILE__, {__LINE__, u32(-1)}, {__LINE__, u32(-1)}), __VA_ARGS__)
#ifndef NDEBUG
#define DLOG(...) log(thorin::LogLevel::Debug,   Loc(__FILE__, {__LINE__, u32(-1)}, {__LINE__, u32(-1)}), __VA_ARGS__)
#else
#define DLOG(...) do {} while (false)
#endif

}

#endif
