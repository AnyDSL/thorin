#ifndef THORIN_WORLD_H
#define THORIN_WORLD_H

#include <cassert>
#include <functional>
#include <initializer_list>
#include <string>

#include "thorin/enums.h"
#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/util/hash.h"

namespace thorin {

/**
 * @brief The World represents the whole program and manages creation and destruction of Thorin nodes.
 *
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
 *  @p PrimOp%s do not explicitly belong to a Lambda.
 *  Instead they either implicitly belong to a Lambda--when
 *  they (possibly via multiple levels of indirection) depend on a Lambda's Param--or they are dead.
 *  Use @p cleanup to remove dead code and unreachable code.
 *
 *  You can create several worlds.
 *  All worlds are completely independent from each other.
 *  This is particular useful for multi-threading.
 */
class World {
private:
    World& operator = (const World&); ///< Do not copy-assign a @p World instance.
    World(const World&);              ///< Do not copy-construct a @p World.

    struct TypeHash { uint64_t operator () (const TypeNode* t) const { return t->hash(); } };
    struct TypeEqual { bool operator () (const TypeNode* t1, const TypeNode* t2) const { return t1->equal(t2); } };

public:
    typedef HashSet<const PrimOp*, PrimOpHash, PrimOpEqual> PrimOps;
    typedef HashSet<const TypeNode*, TypeHash, TypeEqual> Types;

    World(std::string name = "");
    ~World();

    // types

#define THORIN_ALL_TYPE(T, M) PrimType type_##T(size_t length = 1) { \
    return length == 1 ? PrimType(T##_) : join(new PrimTypeNode(*this, PrimType_##T, length)); \
}
#include "thorin/tables/primtypetable.h"

    /// Get PrimType.
    PrimType    type(PrimTypeKind kind, size_t length = 1) {
        size_t i = kind - Begin_PrimType;
        assert(i < (size_t) Num_PrimTypes);
        return length == 1 ? PrimType(primtypes_[i]) : join(new PrimTypeNode(*this, kind, length));
    }
    MemType     mem_type() const { return MemType(mem_); }
    FrameType   frame_type() const { return FrameType(frame_); }
    PtrType     ptr_type(Type referenced_type, size_t length = 1, int32_t device = -1, AddressSpace addr_space = AddressSpace::Generic) {
        return join(new PtrTypeNode(*this, referenced_type, length, device, addr_space));
    }
    TupleType           tuple_type() { return tuple0_; } ///< Returns unit, i.e., an empty @p TupleType.
    TupleType           tuple_type(ArrayRef<Type> args) { return join(new TupleTypeNode(*this, args)); }
    StructAbsType       struct_abs_type(size_t size, const std::string& name = "") {
        return join(new StructAbsTypeNode(*this, size, name));
    }
    StructAppType       struct_app_type(StructAbsType struct_abs_type, ArrayRef<Type> args) {
        return join(new StructAppTypeNode(struct_abs_type, args));
    }
    FnType              fn_type() { return fn0_; }       ///< Returns an empty @p FnType.
    FnType              fn_type(ArrayRef<Type> args) { return join(new FnTypeNode(*this, args)); }
    TypeVar             type_var() { return join(new TypeVarNode(*this)); }
    DefiniteArrayType   definite_array_type(Type elem, u64 dim) { return join(new DefiniteArrayTypeNode(*this, elem, dim)); }
    IndefiniteArrayType indefinite_array_type(Type elem) { return join(new IndefiniteArrayTypeNode(*this, elem)); }

    // literals

#define THORIN_ALL_TYPE(T, M) \
    Def literal_##T(T val, const Location& loc, size_t length = 1) { return literal(PrimType_##T, Box(val), loc, length); }
#include "thorin/tables/primtypetable.h"
    Def literal(PrimTypeKind kind, Box box, const Location& loc, size_t length = 1) { return splat(cse(new PrimLit(*this, kind, box, loc, "")), length); }
    Def literal(PrimTypeKind kind, int64_t value, const Location& loc, size_t length = 1);
    template<class T>
    Def literal(T value, const Location& loc, size_t length = 1) { return literal(type2kind<T>::kind, Box(value), loc, length); }
    Def zero(PrimTypeKind kind, const Location& loc, size_t length = 1) { return literal(kind, 0, loc, length); }
    Def zero(Type type, const Location& loc, size_t length = 1) { return zero(type.as<PrimType>()->primtype_kind(), loc, length); }
    Def one(PrimTypeKind kind, const Location& loc, size_t length = 1) { return literal(kind, 1, loc, length); }
    Def one(Type type, const Location& loc, size_t length = 1) { return one(type.as<PrimType>()->primtype_kind(), loc, length); }
    Def allset(PrimTypeKind kind, const Location& loc, size_t length = 1) { return literal(kind, -1, loc, length); }
    Def allset(Type type, const Location& loc, size_t length = 1) { return allset(type.as<PrimType>()->primtype_kind(), loc, length); }
    Def bottom(Type type, const Location& loc, size_t length = 1) { return splat(cse(new Bottom(type, loc, "")), length); }
    Def bottom(PrimTypeKind kind, const Location& loc, size_t length = 1) { return bottom(type(kind), loc, length); }
    /// Creates a vector of all true while the length is derived from @p def.
    Def true_mask(Def def) { return literal(true, def->loc(), def->length()); }
    Def true_mask(size_t length, const Location& loc) { return literal(true, loc, length); }
    Def false_mask(Def def) { return literal(false, def->loc(), def->length()); }
    Def false_mask(size_t length, const Location& loc) { return literal(false, loc, length); }

    // arithops

    /// Creates an \p ArithOp or a \p Cmp.
    Def binop(int kind, Def cond, Def lhs, Def rhs, const Location& loc, const std::string& name = "");
    Def binop(int kind, Def lhs, Def rhs, const Location& loc, const std::string& name = "") {
        return binop(kind, true_mask(lhs), lhs, rhs, loc, name);
    }

    Def arithop(ArithOpKind kind, Def cond, Def lhs, Def rhs, const Location& loc, const std::string& name = "");
    Def arithop(ArithOpKind kind, Def lhs, Def rhs, const Location& loc, const std::string& name = "") {
        return arithop(kind, true_mask(lhs), lhs, rhs, loc, name);
    }
#define THORIN_ARITHOP(OP) \
    Def arithop_##OP(Def cond, Def lhs, Def rhs, const Location& loc, const std::string& name = "") { \
        return arithop(ArithOp_##OP, cond, lhs, rhs, loc, name); \
    } \
    Def arithop_##OP(Def lhs, Def rhs, const Location& loc, const std::string& name = "") { \
        return arithop(ArithOp_##OP, true_mask(lhs), lhs, rhs, loc, name); \
    }
#include "thorin/tables/arithoptable.h"

    Def arithop_not(Def cond, Def def, const Location& loc);
    Def arithop_not(Def def, const Location& loc) { return arithop_not(true_mask(def), def, loc); }
    Def arithop_minus(Def cond, Def def, const Location& loc);
    Def arithop_minus(Def def, const Location& loc) { return arithop_minus(true_mask(def), def, loc); }

    // compares

    Def cmp(CmpKind kind, Def cond, Def lhs, Def rhs, const Location& loc, const std::string& name = "");
    Def cmp(CmpKind kind, Def lhs, Def rhs, const Location& loc, const std::string& name = "") {
        return cmp(kind, true_mask(lhs), lhs, rhs, loc, name);
    }
#define THORIN_CMP(OP) \
    Def cmp_##OP(Def cond, Def lhs, Def rhs, const Location& loc, const std::string& name = "") { \
        return cmp(Cmp_##OP, cond, lhs, rhs, loc, name);  \
    } \
    Def cmp_##OP(Def lhs, Def rhs, const Location& loc, const std::string& name = "") { \
        return cmp(Cmp_##OP, true_mask(lhs), lhs, rhs, loc, name);  \
    }
#include "thorin/tables/cmptable.h"

    // casts

    Def convert(Type to, Def from, const Location& loc, const std::string& name = "");
    Def cast(Type to, Def cond, Def from, const Location& loc, const std::string& name = "");
    Def cast(Type to, Def from, const Location& loc, const std::string& name = "") { return cast(to, true_mask(from), from, loc, name); }
    Def bitcast(Type to, Def cond, Def from, const Location& loc, const std::string& name = "");
    Def bitcast(Type to, Def from, const Location& loc, const std::string& name = "") { return bitcast(to, true_mask(from), from, loc, name); }

    // aggregate operations

    Def definite_array(Type elem, ArrayRef<Def> args, const Location& loc, const std::string& name = "") {
        return cse(new DefiniteArray(*this, elem, args, loc, name));
    }
    /// Create definite_array with at least one element. The type of that element is the element type of the definite array.
    Def definite_array(ArrayRef<Def> args, const Location& loc, const std::string& name = "") {
        assert(!args.empty());
        return definite_array(args.front()->type(), args, loc, name);
    }
    Def indefinite_array(Type elem, Def dim, const Location& loc, const std::string& name = "") {
        return cse(new IndefiniteArray(*this, elem, dim, loc, name));
    }
    Def struct_agg(StructAppType struct_app_type, ArrayRef<Def> args, const Location& loc, const std::string& name = "") {
        return cse(new StructAgg(struct_app_type, args, loc, name));
    }
    Def tuple(ArrayRef<Def> args, const Location& loc, const std::string& name = "") { return cse(new Tuple(*this, args, loc, name)); }
    Def vector(ArrayRef<Def> args, const Location& loc, const std::string& name = "") {
        if (args.size() == 1) return args[0];
        return cse(new Vector(*this, args, loc, name));
    }
    /// Splats \p arg to create a \p Vector with \p length.
    Def splat(Def arg, size_t length = 1, const std::string& name = "");
    Def extract(Def tuple, Def index, const Location& loc, const std::string& name = "");
    Def extract(Def tuple, u32 index, const Location& loc, const std::string& name = "") { return extract(tuple, literal_qu32(index, loc), loc, name); }
    Def insert(Def tuple, Def index, Def value, const Location& loc, const std::string& name = "");
    Def insert(Def tuple, u32 index, Def value, const Location& loc, const std::string& name = "") {
        return insert(tuple, literal_qu32(index, loc), value, loc, name);
    }

    Def select(Def cond, Def t, Def f, const std::string& name = "");

    // memory stuff

    Def load(Def mem, Def ptr, const Location& loc, const std::string& name = "");
    Def store(Def mem, Def ptr, Def val, const Location& loc, const std::string& name = "");
    Def enter(Def mem, const Location& loc, const std::string& name = "");
    Def slot(Type type, Def frame, size_t index, const Location& loc, const std::string& name = "");
    Def alloc(Type type, Def mem, Def extra, const Location& loc, const std::string& name = "");
    Def alloc(Type type, Def mem, const Location& loc, const std::string& name = "") { return alloc(type, mem, literal_qu64(0, loc), loc, name); }
    Def global(Def init, const Location& loc, bool is_mutable = true, const std::string& name = "");
    Def global_immutable_string(const Location& loc, const std::string& str, const std::string& name = "");
    Def lea(Def ptr, Def index, const Location& loc, const std::string& name = "") { return cse(new LEA(ptr, index, loc, name)); }
    const Map* map(Def device, Def addr_space, Def mem, Def ptr, Def mem_offset, Def mem_size, const Location& loc, const std::string& name = "");
    const Map* map(uint32_t device, AddressSpace addr_space, Def mem, Def ptr, Def mem_offset,
                   Def mem_size, const Location& loc, const std::string& name = "") {
        return cse(new Map(device, addr_space, mem, ptr, mem_offset, mem_size, loc, name));
    }

    // misc

    Def run(Def def, const Location& loc, const std::string& name = "");
    Def hlt(Def def, const Location& loc, const std::string& name = "");
    Def select(Def cond, Def t, Def f, const Location& loc, const std::string& name = "");

    // lambdas

    Lambda* lambda(FnType fn, const Location& loc, CC cc = CC::C, Intrinsic intrinsic = Intrinsic::None, const std::string& name = "");
    Lambda* lambda(FnType fn, const Location& loc, const std::string& name) { return lambda(fn, loc, CC::C, Intrinsic::None, name); }
    Lambda* lambda(const Location& loc, const std::string& name) { return lambda(fn_type(), loc, CC::C, Intrinsic::None, name); }
    Lambda* basicblock(const Location& loc, const std::string& name = "");
    Lambda* meta_lambda();
    Lambda* branch() const { return branch_; }
    Lambda* end_scope() const { return end_scope_; }

    /// Performs dead code, unreachable code and unused type elimination.
    void cleanup();
    void opt();

    // getters

    const std::string& name() const { return name_; }
    const PrimOps& primops() const { return primops_; }
    const LambdaSet& lambdas() const { return lambdas_; }
    Array<Lambda*> copy_lambdas() const;
    const LambdaSet& externals() const { return externals_; }
    const Types& types() const { return types_; }
    size_t gid() const { return gid_; }
    bool empty() const { return lambdas().size() <= 2; } // TODO rework intrinsic stuff. 2 = branch + end_scope

    // other stuff

    void add_external(Lambda* lambda) { externals_.insert(lambda); }
    void remove_external(Lambda* lambda) { externals_.erase(lambda); }
    bool is_external(const Lambda* lambda) { return externals().contains(const_cast<Lambda*>(lambda)); }
    void destroy(Lambda* lambda);
#ifndef NDEBUG
    void breakpoint(size_t number) { breakpoints_.insert(number); }
    const HashSet<size_t>& breakpoints() const { return breakpoints_; }
#endif
    const TypeNode* unify_base(const TypeNode*);
    template<class T> Proxy<T> unify(const T* type) { return Proxy<T>(unify_base(type)->template as<T>()); }
    void dump() const;

private:
    const TypeNode* register_base(const TypeNode* type) {
        assert(type->gid_ == size_t(-1));
        type->gid_ = gid_++;
        garbage_.push_back(type);
        return type;
    }
    template<class T> Proxy<T> join(const T* t) { return Proxy<T>(register_base(t)->template as<T>()); }
    const DefNode* cse_base(const PrimOp*);
    template<class T> const T* cse(const T* primop) { return cse_base(primop)->template as<T>(); }

    const Param* param(Type type, Lambda* lambda, size_t index, const std::string& name = "");

    std::string name_;
    LambdaSet lambdas_;
    LambdaSet externals_;
    PrimOps primops_;
    Types types_;
    std::vector<const TypeNode*> garbage_;
#ifndef NDEBUG
    HashSet<size_t> breakpoints_;
#endif
    size_t gid_;
    Lambda* branch_;
    Lambda* end_scope_;

    union {
        struct {
            const TupleTypeNode* tuple0_;///< tuple().
            const FnTypeNode*    fn0_;   ///< fn().
            const MemTypeNode*   mem_;
            const FrameTypeNode* frame_;

            union {
                struct {
#define THORIN_ALL_TYPE(T, M) const PrimTypeNode* T##_;
#include "thorin/tables/primtypetable.h"
                };

                const PrimTypeNode* primtypes_[Num_PrimTypes];
            };
        };

        const TypeNode* keep_[Num_PrimTypes + 4];
    };

    friend class Cleaner;
    friend class Lambda;
};

}

#endif
