#ifndef THORIN_WORLD_H
#define THORIN_WORLD_H

#include <cassert>
#include <iostream>
#include <functional>
#include <initializer_list>
#include <string>

#include "thorin/enums.h"
#include "thorin/lam.h"
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
class World : public TypeTable, public Streamable {
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

    // literals

#define THORIN_ALL_TYPE(T, M) \
    const Def* literal_##T(T val, Debug dbg = {}, size_t length = 1) { return literal(PrimType_##T, Box(val), dbg, length); }
#include "thorin/tables/primtypetable.h"
    const Def* literal(PrimTypeTag tag, Box box, Debug dbg, size_t length = 1) { return splat(unify(new PrimLit(*this, tag, box, dbg)), length); }
    template<class T>
    const Def* literal(T value, Debug dbg = {}, size_t length = 1) { return literal(type2tag<T>::tag, Box(value), dbg, length); }
    const Def* zero(PrimTypeTag tag, Debug dbg = {}, size_t length = 1) { return literal(tag, 0, dbg, length); }
    const Def* zero(const Type* type, Debug dbg = {}, size_t length = 1) { return zero(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* one(PrimTypeTag tag, Debug dbg = {}, size_t length = 1) { return literal(tag, 1, dbg, length); }
    const Def* one(const Type* type, Debug dbg = {}, size_t length = 1) { return one(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* allset(PrimTypeTag tag, Debug dbg = {}, size_t length = 1);
    const Def* allset(const Type* type, Debug dbg = {}, size_t length = 1) { return allset(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* top   (const Type* type, Debug dbg = {}, size_t length = 1) { return splat(unify(new Top(type, dbg)), length); }
    const Def* bottom(const Type* type, Debug dbg = {}, size_t length = 1) { return splat(unify(new Bottom(type, dbg)), length); }
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

    const Def* convert(const Type* to, const Def* from, Debug dbg = {});
    const Def* cast(const Type* to, const Def* from, Debug dbg = {});
    const Def* bitcast(const Type* to, const Def* from, Debug dbg = {});

    // aggregate operations

    const Def* definite_array(const Type* elem, Defs args, Debug dbg = {}) {
        return try_fold_aggregate(unify(new DefiniteArray(*this, elem, args, dbg)));
    }
    /// Create definite_array with at least one element. The type of that element is the element type of the definite array.
    const Def* definite_array(Defs args, Debug dbg = {}) {
        assert(!args.empty());
        return definite_array(args.front()->type(), args, dbg);
    }
    const Def* indefinite_array(const Type* elem, const Def* dim, Debug dbg = {}) {
        return unify(new IndefiniteArray(*this, elem, dim, dbg));
    }
    const Def* struct_agg(const StructType* struct_type, Defs args, Debug dbg = {}) {
        return try_fold_aggregate(unify(new StructAgg(struct_type, args, dbg)));
    }

    const Def* tuple(Defs args, Debug dbg = {}) { return args.size() == 1 ? args.front() : try_fold_aggregate(unify(new Tuple(*this, args, dbg))); }

    const Def* variant(const VariantType* variant_type, const Def* value, size_t index, Debug dbg = {}) { return unify(new Variant(variant_type, value, index, dbg)); }
    const Def* variant_index  (const Def* value, Debug dbg = {});
    const Def* variant_extract(const Def* value, size_t index, Debug dbg = {});

    const Def* closure(const Pi* pi, const Def* fn, const Def* env, Debug dbg = {}) { return unify(new Closure(pi, fn, env, dbg)); }

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
    const Def* align_of(const Type* type, Debug dbg = {});
    const Def* size_of(const Type* type, Debug dbg = {});

    // memory stuff

    const Def* load(const Def* mem, const Def* ptr, Debug dbg = {});
    const Def* store(const Def* mem, const Def* ptr, const Def* val, Debug dbg = {});
    const Def* enter(const Def* mem, Debug dbg = {});
    const Def* slot(const Type* type, const Def* frame, Debug dbg = {}) { return unify(new Slot(type, frame, dbg)); }
    const Def* alloc(const Type* type, const Def* mem, const Def* extra, Debug dbg = {});
    const Def* alloc(const Type* type, const Def* mem, Debug dbg = {}) { return alloc(type, mem, literal_qu64(0, dbg), dbg); }
    const Def* global(const Def* init, bool is_mutable = true, Debug dbg = {});
    const Def* global_immutable_string(const std::string& str, Debug dbg = {});
    const Def* lea(const Def* ptr, const Def* index, Debug dbg);
    const Assembly* assembly(const Type* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});
    const Assembly* assembly(Types types, const Def* mem, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});

    // partial evaluation related stuff

    const Def* hlt(const Def* def, Debug dbg = {});
    const Def* known(const Def* def, Debug dbg = {});
    const Def* run(const Def* def, Debug dbg = {});

    // lams
    const Param* param(Lam* lam, Debug dbg) { return unify(new Param(lam->domain(), lam, dbg)); }
    Lam* lam(const Pi* cn, Lam::Attributes attrs, Debug dbg = {}) {
        return insert(new Lam(cn, attrs, dbg));
    }
    Lam* lam(const Pi* cn, Debug dbg = {}) { return lam(cn, Lam::Attributes(), dbg); }
    Lam* lam(Debug dbg = {}) { return lam(cn(), Lam::Attributes(), dbg); }
    Lam* branch() const { return branch_; }
    Lam* match(const Type* type, size_t num_patterns);
    Lam* end_scope() const { return end_scope_; }

    const App* app(const Def* callee, const Def* arg, Debug dbg = {});
    const App* app(const Def* callee, Defs args, Debug dbg = {}) { return app(callee, tuple(args), dbg); }

    /// Performs dead code, unreachable code and unused type elimination.
    void cleanup();
    void opt();

    // getters

    const std::string& name() const { return name_; }
    const DefSet& defs() const { return defs_; }
    std::vector<Lam*> copy_lams() const;
    std::vector<Lam*> exported_lams() const;

    bool is_empty() const { return exported_lams().empty(); }

    // other stuff

    void mark_pe_done(bool flag = true) { pe_done_ = flag; }
    bool is_pe_done() const { return pe_done_; }

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
        swap(static_cast<TypeTable&>(w1), static_cast<TypeTable&>(w2));
        swap(w1.name_,          w2.name_);
        swap(w1.defs_,          w2.defs_);
        swap(w1.branch_,        w2.branch_);
        swap(w1.end_scope_,     w2.end_scope_);
        swap(w1.pe_done_,       w2.pe_done_);

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
    DefSet defs_;
    Lam* branch_;
    Lam* end_scope_;
    bool pe_done_ = false;
#if THORIN_ENABLE_CHECKS
    Breakpoints breakpoints_;
    bool track_history_ = false;
#endif

    friend class Cleaner;
    friend class Lam;
    friend void Def::replace(Tracker) const;
};

}

#endif
