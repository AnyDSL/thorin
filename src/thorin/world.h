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

    World(std::string name = "");
    ~World();

    bool empty() { return externals().empty(); }

    const Kind* star() { return star_; }
    const Var* var(const Def* type, u64 index, Debug dbg = {}) { return unify(new Var(type, index, dbg)); }
    const VariantType* variant_type(const Def* type, Defs ops, Debug dbg = {}) { return unify(new VariantType(type, ops, dbg)); }

    ///@defgroup @p Sigma%s
    //@{
    const Sigma* unit() { return unit_; } ///< Returns unit, i.e., an empty @p Sigma.
    const Def* sigma(const Def* type, Defs ops, Debug dbg = {});
    const Def* sigma(Defs ops, Debug dbg = {}) { return sigma(star(), ops, dbg); }
    /// creates a @em nominal @p Sigma
    Sigma* sigma(const Def* type, size_t size, Debug dbg = {});
    //@}

#define THORIN_ALL_TYPE(T, M) \
    const PrimType* type_##T(size_t length = 1) { return type(PrimType_##T, length); }
#include "thorin/tables/primtypetable.h"
    const PrimType* type(PrimTypeTag tag, size_t length = 1, Debug dbg = {}) {
        size_t i = tag - Begin_PrimType;
        assert(i < (size_t) Num_PrimTypes);
        return length == 1 ? primtypes_[i] : unify(new PrimType(*this, tag, length, dbg));
    }
    const MemType* mem_type() const { return mem_; }
    const FrameType* frame_type() const { return frame_; }
    const PtrType* ptr_type(const Def* pointee,
                            size_t length = 1, int32_t device = -1, AddrSpace addr_space = AddrSpace::Generic, Debug dbg = {}) {
        return unify(new PtrType(star(), pointee, length, device, addr_space, dbg));
    }
    ///@defgroup @p Pi%s
    //@{
    const Pi* pi(Defs domain, const Def* codomain, Debug dbg = {}) { return pi(sigma(domain), codomain, dbg); }
    const Pi* pi(const Def* domain, const Def* codomain, Debug dbg = {});
    ///@defgroup continuation types - Pi types with codomain @p Bottom
    //@{
    const Pi* cn(Defs domains) { return cn(sigma(domains)); }
    const Pi* cn(const Def* domain) { return pi(domain, bottom()); }
    //@}
    //@}

    const DefiniteArrayType*   definite_array_type(const Def* elem, u64 dim, Debug dbg = {}) { return unify(new DefiniteArrayType(star(), elem, dim, dbg)); }
    const IndefiniteArrayType* indefinite_array_type(const Def* elem, Debug dbg = {}) { return unify(new IndefiniteArrayType(star(), elem, dbg)); }


    // literals

#define THORIN_ALL_TYPE(T, M) \
    const Def* literal_##T(T val, Debug dbg = {}, size_t length = 1) { return literal(PrimType_##T, Box(val), dbg, length); }
#include "thorin/tables/primtypetable.h"
    const Def* literal(PrimTypeTag tag, Box box, Debug dbg, size_t length = 1) { return splat(unify(new PrimLit(*this, tag, box, dbg)), length); }
    template<class T>
    const Def* literal(T value, Debug dbg = {}, size_t length = 1) { return literal(type2tag<T>::tag, Box(value), dbg, length); }
    const Def* zero(PrimTypeTag tag, Debug dbg = {}, size_t length = 1) { return literal(tag, 0, dbg, length); }
    const Def* zero(const Def* type, Debug dbg = {}, size_t length = 1) { return zero(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* one(PrimTypeTag tag, Debug dbg = {}, size_t length = 1) { return literal(tag, 1, dbg, length); }
    const Def* one(const Def* type, Debug dbg = {}, size_t length = 1) { return one(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* allset(PrimTypeTag tag, Debug dbg = {}, size_t length = 1);
    const Def* allset(const Def* type, Debug dbg = {}, size_t length = 1) { return allset(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* top   (const Def* type, Debug dbg = {}, size_t length = 1) { return splat(unify(new Top(type, dbg)), length); }
    const Def* bottom(Debug dbg = {}) { return unify(new Bottom(star(), dbg)); }
    const Def* bottom(const Def* type, Debug dbg = {}, size_t length = 1) { return splat(unify(new Bottom(type, dbg)), length); }
    const Def* bottom(PrimTypeTag tag, Debug dbg = {}, size_t length = 1) { return bottom(type(tag), dbg, length); }

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

    const Def* definite_array(const Def* elem, Defs args, Debug dbg = {}) {
        return try_fold_aggregate(unify(new DefiniteArray(*this, elem, args, dbg)));
    }
    /// Create definite_array with at least one element. The type of that element is the element type of the definite array.
    const Def* definite_array(Defs args, Debug dbg = {}) {
        assert(!args.empty());
        return definite_array(args.front()->type(), args, dbg);
    }
    const Def* indefinite_array(const Def* elem, const Def* dim, Debug dbg = {}) {
        return unify(new IndefiniteArray(*this, elem, dim, dbg));
    }
    const Def* tuple(Defs args, Debug dbg = {}) { return args.size() == 1 ? args.front() : try_fold_aggregate(unify(new Tuple(*this, args, dbg))); }
    const Def* variant(const VariantType* variant_type, const Def* value, Debug dbg = {}) { return unify(new Variant(variant_type, value, dbg)); }
    const Def* vector(Defs args, Debug dbg = {}) {
        if (args.size() == 1) return args[0];
        return try_fold_aggregate(unify(new Vector(*this, args, dbg)));
    }
    /// Splats \p arg to create a \p Vector with \p length.
    const Def* splat(const Def* arg, size_t length = 1, Debug dbg = {});
    const Def* extract(const Def* tuple, const Def* index, Debug dbg = {});
    const Def* extract(const Def* tuple, u32 index, Debug dbg = {}) {
        return extract(tuple, literal_qu32(index, dbg), dbg);
    }
    const Def* insert(const Def* tuple, const Def* index, const Def* value, Debug dbg = {});
    const Def* insert(const Def* tuple, u32 index, const Def* value, Debug dbg = {}) {
        return insert(tuple, literal_qu32(index, dbg), value, dbg);
    }

    const Def* select(const Def* cond, const Def* t, const Def* f, Debug dbg = {});
    const Def* size_of(const Def* type, Debug dbg = {});

    // memory stuff

    const Def* load(const Def* mem, const Def* ptr, Debug dbg = {});
    const Def* store(const Def* mem, const Def* ptr, const Def* val, Debug dbg = {});
    const Def* enter(const Def* mem, Debug dbg = {});
    const Def* slot(const Def* type, const Def* frame, Debug dbg = {}) { return unify(new Slot(type, frame, dbg)); }
    const Def* alloc(const Def* type, const Def* mem, const Def* extra, Debug dbg = {});
    const Def* alloc(const Def* type, const Def* mem, Debug dbg = {}) { return alloc(type, mem, literal_qu64(0, dbg), dbg); }
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
    const Lam* lam(const Def* domain, const Def* body, Debug dbg) { return lam(domain, literal_bool(true, Debug()), body, dbg); }
    //@{
    Lam* branch() const { return branch_; }
    Lam* match(const Def* type, size_t num_patterns);
    Lam* end_scope() const { return end_scope_; }
    const Def* app(const Def* callee, const Def* arg, Debug dbg = {});
    const Def* app(const Def* callee, Defs args, Debug dbg = {}) { return app(callee, tuple(args), dbg); }

    /// Performs dead code, unreachable code and unused type elimination.
    void cleanup();
    void opt();

    // getters

    const std::string& name() const { return name_; }
    const DefSet& defs() const { return defs_; }
    std::vector<Lam*> copy_lams() const;
    const LamSet& externals() const { return externals_; }

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
        swap(w1.name_,      w2.name_);
        swap(w1.externals_, w2.externals_);
        swap(w1.defs_,      w2.defs_);
        swap(w1.branch_,    w2.branch_);
        swap(w1.end_scope_, w2.end_scope_);
        swap(w1.pe_done_,   w2.pe_done_);
        swap(w1.unit_,      w2.unit_);
        swap(w1.bottom_,    w2.bottom_);
        swap(w1.mem_,       w2.mem_);
        swap(w1.frame_,     w2.frame_);

#define THORIN_ALL_TYPE(T, M) \
        swap(w1.T##_,       w2.T##_);

#include "thorin/tables/primtypetable.h"

#if THORIN_ENABLE_CHECKS
        swap(w1.breakpoints_,   w2.breakpoints_);
        swap(w1.track_history_, w2.track_history_);
#endif
    }

private:
    const Def* try_fold_aggregate(const Aggregate*);

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
    const T* unify(const T* def) {

#ifndef NDEBUG
        if (breakpoints_.contains(def->gid())) THORIN_BREAK;
#endif
        assert(!def->is_nominal());
        auto&& p = defs_.emplace(def);
        if (p.second) {
            //def->finalize();
            return static_cast<const T*>(def);
        }

        def->unregister_uses();
        delete def;
        return static_cast<const T*>(*p.first);
    }

    std::string name_;
    LamSet externals_;
    DefSet defs_;
    bool pe_done_ = false;
#if THORIN_ENABLE_CHECKS
    Breakpoints breakpoints_;
    bool track_history_ = false;
#endif
    const Kind* star_;
    const Sigma* unit_; ///< tuple().
    const Bottom* bottom_;
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
