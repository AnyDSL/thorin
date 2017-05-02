#ifndef THORIN_WORLD_H
#define THORIN_WORLD_H

#include <cassert>
#include <iostream>
#include <functional>
#include <initializer_list>
#include <string>

#include "thorin/enums.h"
#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/util/hash.h"
#include "thorin/util/stream.h"

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
 *  @p PrimOp%s do not explicitly belong to a Continuation.
 *  Instead they either implicitly belong to a Continuation--when
 *  they (possibly via multiple levels of indirection) depend on a Continuation's Param--or they are dead.
 *  Use @p cleanup to remove dead code and unreachable code.
 *
 *  You can create several worlds.
 *  All worlds are completely independent from each other.
 *  This is particular useful for multi-threading.
 */
class World : public TypeTableBase<World>, public Streamable {
public:
    typedef HashSet<const PrimOp*, PrimOpHash> PrimOpSet;

    struct BreakHash {
        static uint64_t hash(size_t i) { return i; }
        static bool eq(size_t i1, size_t i2) { return i1 == i2; }
        static size_t sentinel() { return size_t(-1); }
    };

    typedef HashSet<size_t, BreakHash> Breakpoints;

    World(std::string name = "");
    ~World();

    // types

#define THORIN_ALL_TYPE(T, M) \
    const PrimType* type_##T(size_t length = 1) { return length == 1 ? T##_ : unify(new PrimType(*this, PrimType_##T, length)); }
#include "thorin/tables/primtypetable.h"

    const PrimType* type(PrimTypeTag tag, size_t length = 1) {
        size_t i = tag - Begin_PrimType;
        assert(i < (size_t) Num_PrimTypes);
        return length == 1 ? primtypes_[i] : unify(new PrimType(*this, tag, length));
    }
    const MemType* mem_type() const { return mem_; }
    const FrameType* frame_type() const { return frame_; }
    const PtrType* ptr_type(const Type* pointee,
                            size_t length = 1, int32_t device = -1, AddrSpace addr_space = AddrSpace::Generic) {
        return unify(new PtrType(*this, pointee, length, device, addr_space));
    }
    const FnType* fn_type() { return fn0_; } ///< Returns an empty @p FnType.
    const FnType* fn_type(Types args) { return unify(new FnType(*this, args)); }
    const DefiniteArrayType*   definite_array_type(const Type* elem, u64 dim) { return unify(new DefiniteArrayType(*this, elem, dim)); }
    const IndefiniteArrayType* indefinite_array_type(const Type* elem) { return unify(new IndefiniteArrayType(*this, elem)); }

    // literals

#define THORIN_ALL_TYPE(T, M) \
    const Def* literal_##T(T val, Debug dbg, size_t length = 1) { return literal(PrimType_##T, Box(val), dbg, length); }
#include "thorin/tables/primtypetable.h"
    const Def* literal(PrimTypeTag tag, Box box, Debug dbg, size_t length = 1) { return splat(cse(new PrimLit(*this, tag, box, dbg)), length); }
    template<class T>
    const Def* literal(T value, Debug dbg = {}, size_t length = 1) { return literal(type2tag<T>::tag, Box(value), dbg, length); }
    const Def* zero(PrimTypeTag tag, Debug dbg = {}, size_t length = 1) { return literal(tag, 0, dbg, length); }
    const Def* zero(const Type* type, Debug dbg = {}, size_t length = 1) { return zero(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* one(PrimTypeTag tag, Debug dbg = {}, size_t length = 1) { return literal(tag, 1, dbg, length); }
    const Def* one(const Type* type, Debug dbg = {}, size_t length = 1) { return one(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* allset(PrimTypeTag tag, Debug dbg = {}, size_t length = 1) { return literal(tag, -1, dbg, length); }
    const Def* allset(const Type* type, Debug dbg = {}, size_t length = 1) { return allset(type->as<PrimType>()->primtype_tag(), dbg, length); }
    const Def* bottom(const Type* type, Debug dbg = {}, size_t length = 1) { return splat(cse(new Bottom(type, dbg)), length); }
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
        return cse(new DefiniteArray(*this, elem, args, dbg));
    }
    /// Create definite_array with at least one element. The type of that element is the element type of the definite array.
    const Def* definite_array(Defs args, Debug dbg = {}) {
        assert(!args.empty());
        return definite_array(args.front()->type(), args, dbg);
    }
    const Def* indefinite_array(const Type* elem, const Def* dim, Debug dbg = {}) {
        return cse(new IndefiniteArray(*this, elem, dim, dbg));
    }
    const Def* struct_agg(const StructType* struct_type, Defs args, Debug dbg = {}) {
        return cse(new StructAgg(struct_type, args, dbg));
    }
    const Def* tuple(Defs args, Debug dbg = {}) { return cse(new Tuple(*this, args, dbg)); }
    const Def* vector(Defs args, Debug dbg = {}) {
        if (args.size() == 1) return args[0];
        return cse(new Vector(*this, args, dbg));
    }
    /// Splats \p arg to create a \p Vector with \p length.
    const Def* splat(const Def* arg, size_t length = 1, Debug dbg = {});
    const Def* extract(const Def* tuple, const Def* index, Debug dbg = {});
    const Def* extract(const Def* tuple, size_t index, Debug dbg = {}) {
        return extract(tuple, literal_qu32(index, dbg), dbg);
    }
    const Def* insert(const Def* tuple, const Def* index, const Def* value, Debug dbg = {});
    const Def* insert(const Def* tuple, u32 index, const Def* value, Debug dbg = {}) {
        return insert(tuple, literal_qu32(index, dbg), value, dbg);
    }

    const Def* select(const Def* cond, const Def* t, const Def* f, Debug dbg = {});
    const Def* size_of(const Type* type, Debug dbg = {});

    // memory stuff

    const Def* load(const Def* mem, const Def* ptr, Debug dbg = {});
    const Def* store(const Def* mem, const Def* ptr, const Def* val, Debug dbg = {});
    const Def* enter(const Def* mem, Debug dbg = {});
    const Def* slot(const Type* type, const Def* frame, Debug dbg = {}) { return cse(new Slot(type, frame, dbg)); }
    const Def* alloc(const Type* type, const Def* mem, const Def* extra, Debug dbg = {});
    const Def* alloc(const Type* type, const Def* mem, Debug dbg = {}) { return alloc(type, mem, literal_qu64(0, dbg), dbg); }
    const Def* global(const Def* init, bool is_mutable = true, Debug dbg = {});
    const Def* global_immutable_string(const std::string& str, Debug dbg = {});
    const Def* lea(const Def* ptr, const Def* index, Debug dbg) { return cse(new LEA(ptr, index, dbg)); }
    const Assembly* assembly(const Type* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});
    const Assembly* assembly(Types types, const Def* mem, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints,
                             ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, Debug dbg = {});

    // guided partial evaluation

    const Def* run(const Def* begin, const Def* end, Debug dbg = {}) { return cse(new Run(begin, end, dbg)); }
    const Def* hlt(const Def* begin, const Def* end, Debug dbg = {}) { return cse(new Hlt(begin, end, dbg)); }

    // continuations

    Continuation* continuation(const FnType* fn, CC cc = CC::C, Intrinsic intrinsic = Intrinsic::None, Debug dbg = {});
    Continuation* continuation(const FnType* fn, Debug dbg = {}) { return continuation(fn, CC::C, Intrinsic::None, dbg); }
    Continuation* continuation(Debug dbg = {}) { return continuation(fn_type(), CC::C, Intrinsic::None, dbg); }
    Continuation* basicblock(Debug dbg = {});
    Continuation* branch() const { return branch_; }
    Continuation* match(const Type* type, size_t num_patterns);
    Continuation* end_scope() const { return end_scope_; }

    /// Performs dead code, unreachable code and unused type elimination.
    void cleanup();
    void opt();

    // getters

    const std::string& name() const { return name_; }
    const PrimOpSet& primops() const { return primops_; }
    const ContinuationSet& continuations() const { return continuations_; }
    Array<Continuation*> copy_continuations() const;
    const ContinuationSet& externals() const { return externals_; }
    bool empty() const { return continuations().size() <= 2; } // TODO rework intrinsic stuff. 2 = branch + end_scope

    // other stuff

    void add_external(Continuation* continuation) { externals_.insert(continuation); }
    void remove_external(Continuation* continuation) { externals_.erase(continuation); }
    bool is_external(const Continuation* continuation) { return externals().contains(const_cast<Continuation*>(continuation)); }
    void destroy(Continuation* continuation);
#ifndef NDEBUG
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
        swap(static_cast<TypeTableBase<World>&>(w1), static_cast<TypeTableBase<World>&>(w2));
        swap(w1.name_,          w2.name_);
#define THORIN_ALL_TYPE(T, M) \
        swap(w1.T##_,           w2.T##_);
#include "thorin/tables/primtypetable.h"
        swap(w1.fn0_,           w2.fn0_);
        swap(w1.mem_,           w2.mem_);
        swap(w1.frame_,         w2.frame_);
        swap(w1.continuations_, w2.continuations_);
        swap(w1.externals_,     w2.externals_);
        swap(w1.primops_,       w2.primops_);
        swap(w1.trackers_,      w2.trackers_);
        swap(w1.branch_,        w2.branch_);
        swap(w1.end_scope_,     w2.end_scope_);
#ifndef NDEBUG
        swap(w1.breakpoints_,   w2.breakpoints_);
#endif
    }

private:
    Trackers& trackers(const Def* def) {
        assert(def);
        return trackers_[def];
    }
    const Param* param(const Type* type, Continuation* continuation, size_t index, Debug dbg);
    const Def* cse_base(const PrimOp*);
    template<class T> const T* cse(const T* primop) { return cse_base(primop)->template as<T>(); }

    std::string name_;
    union {
        struct {
#define THORIN_ALL_TYPE(T, M) const PrimType* T##_;
#include "thorin/tables/primtypetable.h"
        };

        const PrimType* primtypes_[Num_PrimTypes];
    };
    const FnType* fn0_;
    const MemType* mem_;
    const FrameType* frame_;
    ContinuationSet continuations_;
    ContinuationSet externals_;
    PrimOpSet primops_;
    DefMap<Trackers> trackers_;
    Continuation* branch_;
    Continuation* end_scope_;
#ifndef NDEBUG
    Breakpoints breakpoints_;
    bool track_history_ = false;
#endif

    friend class Cleaner;
    friend class Continuation;
    friend class Tracker;
    friend void Def::replace(const Def*) const;
};

}

#endif
