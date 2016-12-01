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

    const PrimType* type(PrimTypeKind kind, size_t length = 1) {
        size_t i = kind - Begin_PrimType;
        assert(i < (size_t) Num_PrimTypes);
        return length == 1 ? primtypes_[i] : unify(new PrimType(*this, kind, length));
    }
    const MemType* mem_type() const { return mem_; }
    const FrameType* frame_type() const { return frame_; }
    const PtrType* ptr_type(const Type* referenced_type,
                            size_t length = 1, int32_t device = -1, AddrSpace addr_space = AddrSpace::Generic) {
        return unify(new PtrType(*this, referenced_type, length, device, addr_space));
    }
    const FnType* fn_type() { return fn0_; } ///< Returns an empty @p FnType.
    const FnType* fn_type(Types args) { return unify(new FnType(*this, args)); }
    const DefiniteArrayType*   definite_array_type(const Type* elem, u64 dim) { return unify(new DefiniteArrayType(*this, elem, dim)); }
    const IndefiniteArrayType* indefinite_array_type(const Type* elem) { return unify(new IndefiniteArrayType(*this, elem)); }

    // literals

#define THORIN_ALL_TYPE(T, M) \
    const Def* literal_##T(T val, const Location& loc, size_t length = 1) { return literal(PrimType_##T, Box(val), loc, length); }
#include "thorin/tables/primtypetable.h"
    const Def* literal(PrimTypeKind kind, Box box, const Location& loc, size_t length = 1) { return splat(cse(new PrimLit(*this, kind, box, loc, "")), length); }
    template<class T>
    const Def* literal(T value, const Location& loc, size_t length = 1) { return literal(type2kind<T>::kind, Box(value), loc, length); }
    const Def* zero(PrimTypeKind kind, const Location& loc, size_t length = 1) { return literal(kind, 0, loc, length); }
    const Def* zero(const Type* type, const Location& loc, size_t length = 1) { return zero(type->as<PrimType>()->primtype_kind(), loc, length); }
    const Def* one(PrimTypeKind kind, const Location& loc, size_t length = 1) { return literal(kind, 1, loc, length); }
    const Def* one(const Type* type, const Location& loc, size_t length = 1) { return one(type->as<PrimType>()->primtype_kind(), loc, length); }
    const Def* allset(PrimTypeKind kind, const Location& loc, size_t length = 1) { return literal(kind, -1, loc, length); }
    const Def* allset(const Type* type, const Location& loc, size_t length = 1) { return allset(type->as<PrimType>()->primtype_kind(), loc, length); }
    const Def* bottom(const Type* type, const Location& loc, size_t length = 1) { return splat(cse(new Bottom(type, loc, "")), length); }
    const Def* bottom(PrimTypeKind kind, const Location& loc, size_t length = 1) { return bottom(type(kind), loc, length); }

    // arithops

    /// Creates an \p ArithOp or a \p Cmp.
    const Def* binop(int kind, const Def* lhs, const Def* rhs, const Location& loc, const std::string& name = "");
    const Def* arithop_not(const Def* def, const Location& loc);
    const Def* arithop_minus(const Def* def, const Location& loc);
    const Def* arithop(ArithOpKind kind, const Def* lhs, const Def* rhs, const Location& loc, const std::string& name = "");
#define THORIN_ARITHOP(OP) \
    const Def* arithop_##OP(const Def* lhs, const Def* rhs, const Location& loc, const std::string& name = "") { \
        return arithop(ArithOp_##OP, lhs, rhs, loc, name); \
    }
#include "thorin/tables/arithoptable.h"

    // compares

    const Def* cmp(CmpKind kind, const Def* lhs, const Def* rhs, const Location& loc, const std::string& name = "");
#define THORIN_CMP(OP) \
    const Def* cmp_##OP(const Def* lhs, const Def* rhs, const Location& loc, const std::string& name = "") { \
        return cmp(Cmp_##OP, lhs, rhs, loc, name);  \
    }
#include "thorin/tables/cmptable.h"

    // casts

    const Def* convert(const Type* to, const Def* from, const Location& loc, const std::string& name = "");
    const Def* cast(const Type* to, const Def* from, const Location& loc, const std::string& name = "");
    const Def* bitcast(const Type* to, const Def* from, const Location& loc, const std::string& name = "");

    // aggregate operations

    const Def* definite_array(const Type* elem, Defs args, const Location& loc, const std::string& name = "") {
        return cse(new DefiniteArray(*this, elem, args, loc, name));
    }
    /// Create definite_array with at least one element. The type of that element is the element type of the definite array.
    const Def* definite_array(Defs args, const Location& loc, const std::string& name = "") {
        assert(!args.empty());
        return definite_array(args.front()->type(), args, loc, name);
    }
    const Def* indefinite_array(const Type* elem, const Def* dim, const Location& loc, const std::string& name = "") {
        return cse(new IndefiniteArray(*this, elem, dim, loc, name));
    }
    const Def* struct_agg(const StructType* struct_type, Defs args, const Location& loc, const std::string& name = "") {
        return cse(new StructAgg(struct_type, args, loc, name));
    }
    const Def* tuple(Defs args, const Location& loc, const std::string& name = "") { return cse(new Tuple(*this, args, loc, name)); }
    const Def* vector(Defs args, const Location& loc, const std::string& name = "") {
        if (args.size() == 1) return args[0];
        return cse(new Vector(*this, args, loc, name));
    }
    /// Splats \p arg to create a \p Vector with \p length.
    const Def* splat(const Def* arg, size_t length = 1, const std::string& name = "");
    template<class T> const Def* extract(const Def* tuple, const T* index, const Location& loc, const std::string& name = "");
    const Def* extract(const Def* tuple, u32 index, const Location& loc, const std::string& name = "") {
        return extract(tuple, literal_qu32(index, loc), loc, name);
    }
    const Def* insert(const Def* tuple, const Def* index, const Def* value, const Location& loc, const std::string& name = "");
    const Def* insert(const Def* tuple, u32 index, const Def* value, const Location& loc, const std::string& name = "") {
        return insert(tuple, literal_qu32(index, loc), value, loc, name);
    }

    const Def* select(const Def* cond, const Def* t, const Def* f, const Location& loc, const std::string& name = "");
    const Def* size_of(const Type* type, const Location& loc, const std::string& name = "") { return cse(new SizeOf(bottom(type, loc), loc, name)); }

    // memory stuff

    const Def* load(const Def* mem, const Def* ptr, const Location& loc, const std::string& name = "");
    const Def* store(const Def* mem, const Def* ptr, const Def* val, const Location& loc, const std::string& name = "");
    const Def* enter(const Def* mem, const Location& loc, const std::string& name = "");
    const Def* slot(const Type* type, const Def* frame, const Location& loc, const std::string& name = "") { return cse(new Slot(type, frame, loc, name)); }
    const Def* alloc(const Type* type, const Def* mem, const Def* extra, const Location& loc, const std::string& name = "");
    const Def* alloc(const Type* type, const Def* mem, const Location& loc, const std::string& name = "") { return alloc(type, mem, literal_qu64(0, loc), loc, name); }
    const Def* global(const Def* init, const Location& loc, bool is_mutable = true, const std::string& name = "");
    const Def* global_immutable_string(const Location& loc, const std::string& str, const std::string& name = "");
    const Def* lea(const Def* ptr, const Def* index, const Location& loc, const std::string& name = "") { return cse(new LEA(ptr, index, loc, name)); }
    const Assembly* assembly(const Type* type, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, const Location& loc);
    const Assembly* assembly(Types types, const Def* mem, Defs inputs, std::string asm_template, ArrayRef<std::string> output_constraints, ArrayRef<std::string> input_constraints, ArrayRef<std::string> clobbers, Assembly::Flags flags, const Location& loc);

    // guided partial evaluation

    const Def* run(const Def* begin, const Def* end, const Location& loc, const std::string& name = "") {
        return cse(new Run(begin, end, loc, name));
    }
    const Def* hlt(const Def* begin, const Def* end, const Location& loc, const std::string& name = "") {
        return cse(new Hlt(begin, end, loc, name));
    }

    // continuations

    Continuation* continuation(const FnType* fn, const Location& loc, CC cc = CC::C, Intrinsic intrinsic = Intrinsic::None, const std::string& name = "");
    Continuation* continuation(const FnType* fn, const Location& loc, const std::string& name) { return continuation(fn, loc, CC::C, Intrinsic::None, name); }
    Continuation* continuation(const Location& loc, const std::string& name) { return continuation(fn_type(), loc, CC::C, Intrinsic::None, name); }
    Continuation* basicblock(const Location& loc, const std::string& name = "");
    Continuation* branch() const { return branch_; }
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
    HashSet<Tracker*>& trackers(const Def* def) {
        assert(def);
        return trackers_[def];
    }
    const Param* param(const Type* type, Continuation* continuation, size_t index, const std::string& name = "");
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
    DefMap<HashSet<Tracker*>> trackers_;
    Continuation* branch_;
    Continuation* end_scope_;
#ifndef NDEBUG
    Breakpoints breakpoints_;
#endif

    friend class Cleaner;
    friend class Continuation;
    friend class Tracker;
    friend void Def::replace(const Def*) const;
};

}

#endif
