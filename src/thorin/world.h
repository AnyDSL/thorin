#ifndef THORIN_WORLD_H
#define THORIN_WORLD_H

#include <cassert>
#include <functional>
#include <initializer_list>
#include <queue>
#include <string>

#include "thorin/enums.h"
#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/util/box.h"
#include "thorin/util/hash.h"

namespace thorin {

class Any;
class Bottom;
class DefNode;
class Enter;
class Lambda;
class LEA;
class Load;
class Pi;
class PrimLit;
class PrimOp;
class PrimType;
class Sigma;
class Slot;
class Store;
class Type;
class TypeKeeper;

typedef std::unordered_set<const PrimOp*, PrimOpHash, PrimOpEqual> PrimOpSet;
typedef std::unordered_set<const Type*, TypeHash, TypeEqual> TypeSet;

struct Call {
    Call() {}
    Call(Lambda* to) 
        : to(to)
    {}
    Lambda* to;
    std::vector<Def> args;
    std::vector<size_t> idx;
};

struct CallHash { 
    size_t operator () (const Call& call) const { 
        auto hash = hash_combine(hash_value(call.to), ArrayRef<size_t>(call.idx));
        for (auto def : call.args)
            hash = hash_combine(hash, hash_value(*def));
        return  hash;
    }
};

struct CallEqual { 
    bool operator () (const Call& call1, const Call& call2) const { 
        assert(call1.args.size() == call2.args.size());
        assert(call1.idx.size() == call2.idx.size());
        assert(call1.idx.size() == call2.args.size());

        bool result = call1.to == call2.to && ArrayRef<size_t>(call1.idx) == ArrayRef<size_t>(call2.idx);
        for (size_t i = 0, e = call1.args.size(); i != e && result; ++i)
            result &= call1.args[i] == call2.args[i];
        return result;
    }
};

//------------------------------------------------------------------------------

/**
 * The World represents the whole program and manages creation and destruction of AIR nodes.
 * In particular, the following things are done by this class:
 *
 *  - \p Type unification: \n
 *      There exists only one unique \p Type for \p PrimType%s, Pi%s and \em unnamed \p Sigma%s.
 *      These \p Type%s are hashed into an internal map for fast access.
 *      The getters just calculate a hash and lookup the \p Type, if it is already present, or create a new one otherwise.
 *      There also exists the concept of \em named \p Sigma%s to allow for recursive \p Type%s.
 *      These \p Type%s are \em not unified, i.e., each instance is by definition a different \p Type;
 *      thus, two different pointers of the same named \p Sigma are always considered different \p Type%s.
 *  - Value unification: \n
 *      This is a built-in mechanism for the following things:
 *      - constant pooling
 *      - constant folding
 *      - common subexpression elimination
 *      - canonicalization of expressions
 *      - several local optimizations
 *
 *  \p PrimOp%s do not explicitly belong to a Lambda.
 *  Instead they either implicitly belong to a Lambda--when
 *  they (possibly via multiple levels of indirection) depend on a Lambda's Param--or they are dead.
 *  Use \p cleanup to remove dead code and unreachable code.
 *
 *  You can create several worlds.
 *  All worlds are completely independent from each other.
 *  This is particular useful for multi-threading.
 */
class World {
public:
    World();
    ~World();

    /*
     * types
     */

#define THORIN_UF_TYPE(T) const PrimType* type_##T(size_t length = 1) { \
    return length == 1 ? T##_ : unify(new PrimType(*this, PrimType_##T, length)); \
}
#include "thorin/tables/primtypetable.h"

    /// Get PrimType.
    const PrimType* type(PrimTypeKind kind, size_t length = 1) {
        size_t i = kind - Begin_PrimType;
        assert(0 <= i && i < (size_t) Num_PrimTypes);
        return length == 1 ? primtypes_[i] : unify(new PrimType(*this, kind, length));
    }
    const Mem* mem() const { return mem_; }
    const Frame* frame() const { return frame_; }
    const Ptr* ptr(const Type* referenced_type, size_t length = 1) { return unify(new Ptr(*this, referenced_type, length)); }
    const Sigma* sigma0() { return sigma0_; }   ///< Creates 'sigma()'.
    const Sigma* sigma(ArrayRef<const Type*> elems) { return unify(new Sigma(*this, elems)); }
    Sigma* named_sigma(size_t size, const std::string& name = ""); ///< Creates a fresh \em named sigma.
    const Pi* pi0() { return pi0_; }            ///< Creates 'pi()'.
    const Pi* pi(ArrayRef<const Type*> elems) { return unify(new Pi(*this, elems)); }
    const Generic* generic(size_t index) { return unify(new Generic(*this, index)); }
    const GenericRef* generic_ref(const Generic* generic, Lambda* lambda) { return unify(new GenericRef(*this, generic, lambda)); }
    const ArrayType* array_type(const Type* elem) { return unify(new ArrayType(*this, elem)); }

    /*
     * literals
     */

#define THORIN_UF_TYPE(T) \
    Def literal_##T(T val, size_t length = 1) { return literal(val, length); }
#include "thorin/tables/primtypetable.h"
    Def literal(PrimTypeKind kind, Box value, size_t length = 1);
    Def literal(PrimTypeKind kind, int value, size_t length = 1);
    template<class T>
    Def literal(T value, size_t length = 1) { return literal(type2kind<T>::kind, Box(value), length); }
    Def zero(PrimTypeKind kind, size_t length = 1) { return literal(kind, 0, length); }
    Def zero(const Type*, size_t length = 1);
    Def one(PrimTypeKind kind, size_t length = 1) { return literal(kind, 1, length); }
    Def one(const Type*, size_t length = 1);
    Def allset(PrimTypeKind kind, size_t length = 1) {
        assert(is_int(kind) && "must not be a float");
        return literal(kind, -1, length);
    }
    Def allset(const Type*, size_t length = 1);
    Def any(const Type* type, size_t length = 1);
    Def any(PrimTypeKind kind, size_t length = 1) { return any(type(kind), length); }
    Def bottom(const Type* type, size_t length = 1);
    Def bottom(PrimTypeKind kind, size_t length = 1) { return bottom(type(kind), length); }
    /// Creates a vector of all true while the length is derived from @p def.
    Def true_mask(Def def) { return literal(true, def->length()); }
    Def true_mask(size_t length) { return literal(true, length); }

    /*
     * arithop, relop, convop
     */

    /// Creates an \p ArithOp or a \p RelOp.
    Def binop(int kind, Def cond, Def lhs, Def rhs, const std::string& name = "");
    Def binop(int kind, Def lhs, Def rhs, const std::string& name = "") {
        return binop(kind, true_mask(lhs), lhs, rhs, name);
    }

    Def arithop(ArithOpKind kind, Def cond, Def lhs, Def rhs, const std::string& name = "");
    Def arithop(ArithOpKind kind, Def lhs, Def rhs, const std::string& name = "") {
        return arithop(kind, true_mask(lhs), lhs, rhs, name);
    }
#define THORIN_ARITHOP(OP) \
    Def arithop_##OP(Def cond, Def lhs, Def rhs, const std::string& name = "") { \
        return arithop(ArithOp_##OP, cond, lhs, rhs, name); \
    } \
    Def arithop_##OP(Def lhs, Def rhs, const std::string& name = "") { \
        return arithop(ArithOp_##OP, true_mask(lhs), lhs, rhs, name); \
    }
#include "thorin/tables/arithoptable.h"

    Def arithop_not(Def cond, Def def);
    Def arithop_not(Def def) { return arithop_not(true_mask(def), def); }
    Def arithop_minus(Def cond, Def def);
    Def arithop_minus(Def def) { return arithop_minus(true_mask(def), def); }

    Def relop(RelOpKind kind, Def cond, Def lhs, Def rhs, const std::string& name = "");
    Def relop(RelOpKind kind, Def lhs, Def rhs, const std::string& name = "") {
        return relop(kind, true_mask(lhs), lhs, rhs, name);
    }
#define THORIN_RELOP(OP) \
    Def relop_##OP(Def cond, Def lhs, Def rhs, const std::string& name = "") { \
        return relop(RelOp_##OP, cond, lhs, rhs, name);  \
    } \
    Def relop_##OP(Def lhs, Def rhs, const std::string& name = "") { \
        return relop(RelOp_##OP, true_mask(lhs), lhs, rhs, name);  \
    }
#include "thorin/tables/reloptable.h"

    Def convop(ConvOpKind kind, Def cond, Def from, const Type* to, const std::string& name = "");
    Def convop(ConvOpKind kind, Def from, const Type* to, const std::string& name = "") {
        return convop(kind, true_mask(from), from, to, name);
    }
#define THORIN_CONVOP(OP) \
    Def convop_##OP(Def from, Def cond, const Type* to, const std::string& name = "") { \
        return convop(ConvOp_##OP, cond, from, to, name); \
    } \
    Def convop_##OP(Def from, const Type* to, const std::string& name = "") { \
        return convop(ConvOp_##OP, true_mask(from), from, to, name); \
    }
#include "thorin/tables/convoptable.h"

    /*
     * aggregate stuff
     */

    Def array_agg(const Type* elem, ArrayRef<Def> args, const std::string& name = "") { 
        return cse(new ArrayAgg(*this, elem, args, name)); 
    }
    Def array_agg(ArrayRef<Def> args, const std::string& name = "") { 
        assert(!args.empty()); 
        return array_agg(args.front()->type(), args, name);
    }
    Def tuple(ArrayRef<Def> args, const std::string& name = "") { return cse(new Tuple(*this, args, name)); }
    Def vector(ArrayRef<Def> args, const std::string& name = "") {
        if (args.size() == 1) return args[0];
        return cse(new Vector(*this, args, name)); 
    }
    /// Splats \p arg to create a \p Vector with \p length.
    Def vector(Def arg, size_t length = 1, const std::string& name = "");
    Def extract(Def tuple, Def index, const std::string& name = "");
    Def extract(Def tuple, u32 index, const std::string& name = "");
    Def insert(Def tuple, Def index, Def value, const std::string& name = "");
    Def insert(Def tuple, u32 index, Def value, const std::string& name = "");

    /*
     * memops
     */

    const Load* load(Def mem, Def ptr, const std::string& name = "");
    const Store* store(Def mem, Def ptr, Def val, const std::string& name = "");
    const Enter* enter(Def mem, const std::string& name = "");
    Def leave(Def mem, Def frame, const std::string& name = "");
    const Slot* slot(const Type* type, Def frame, size_t index, const std::string& name = "");
    const LEA* lea(Def ptr, Def index, const std::string& name = "");

    /*
     * other stuff
     */

    Def select(Def cond, Def a, Def b, const std::string& name = "");
    const TypeKeeper* typekeeper(const Type* type, const std::string& name = "");
    const Addr* addr(Def lambda, const std::string& name = "");
    Def run(Def def, const std::string& name = "");
    Def halt(Def def, const std::string& name = "");

    Lambda* lambda(const Pi* pi, Lambda::Attribute attribute = Lambda::Attribute(0), const std::string& name = "");
    Lambda* lambda(const Pi* pi, const std::string& name) { return lambda(pi, Lambda::Attribute(0), name); }
    Lambda* lambda(const std::string& name) { return lambda(pi0(), Lambda::Attribute(0), name); }
    Lambda* basicblock(const std::string& name = "");

    /// Generic \p PrimOp constructor; inherits name from \p in.
    Def rebuild(const PrimOp* in, ArrayRef<Def> ops, const Type* type);
    /// Generic \p PrimOp constructor; inherits type and name name from \p in.
    Def rebuild(const PrimOp* in, ArrayRef<Def> ops) { return rebuild(in, ops, in->type()); }
    /// Generic \p Type constructor.
    const Type* rebuild(const Type* in, ArrayRef<const Type*> elems);

    /*
     * optimizations
     */

    void dead_code_elimination();
    void unreachable_code_elimination();
    void unused_type_elimination();
    void eliminate_params();

    /// Performs dead code, unreachable code and unused type elimination.
    void cleanup();
    void opt();

    /*
     * getters
     */

    const PrimOpSet& primops() const { return primops_; }
    const LambdaSet& lambdas() const { return lambdas_; }
    TypeSet types() const { return types_; }
    size_t gid() const { return gid_; }

    /*
     * other
     */

    const Type* insert_type(const Type*);
    size_t new_pass() { return pass_counter_++; }

#ifndef NDEBUG
    void breakpoint(size_t number) { breakpoints_.insert(number); }
#endif

protected:
    template<class T>
    const T* keep(const T* type) {
        auto tp = types_.insert(type);
        assert(tp.second);
        typekeeper(type);
        return type->template as<T>();
    }
    const Type* unify_base(const Type* type);
    template<class T> const T* unify(const T* type) { return unify_base(type)->template as<T>(); }
    const DefNode* cse_base(const PrimOp*);
    template<class T> const T* cse(const T* primop) { return cse_base(primop)->template as<T>(); }

private:
    PrimOp* release(const PrimOp*);
    const Param* param(const Type* type, Lambda* lambda, size_t index, const std::string& name = "");
    const Type* keep_nocast(const Type* type);
    void eliminate_proxies();
    Def dce_rebuild(Def2Def&, const size_t old_gid, Def def);
    void dce_mark(DefSet&, Def);
    void ute_insert(std::unordered_set<const Type*>&, const Type*);
    void uce_insert(LambdaSet&, Lambda*);
    template<class S, class W> static void wipe_out(S& set, W wipe); 

    PrimOpSet primops_;
    LambdaSet lambdas_;
    TypeSet types_;
#ifndef NDEBUG
    std::unordered_set<size_t> breakpoints_;
#endif

    size_t gid_;
    size_t pass_counter_;
    const Sigma* sigma0_;///< sigma().
    const Pi* pi0_;      ///< pi().
    const Mem* mem_;
    const Frame* frame_;

    union {
        struct {
#define THORIN_UF_TYPE(T) const PrimType* T##_;
#include "thorin/tables/primtypetable.h"
        };

        const PrimType* primtypes_[Num_PrimTypes];
    };

public:
    std::unordered_map<Call, Lambda*, CallHash, CallEqual> cache_;

    friend class Lambda;
};

//------------------------------------------------------------------------------

} // namespace thorin

#endif
