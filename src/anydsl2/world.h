#ifndef ANYDSL2_WORLD_H
#define ANYDSL2_WORLD_H

#include <cassert>
#include <functional>
#include <queue>
#include <string>
#include <set>
#include <unordered_set>

#include "anydsl2/enums.h"
#include "anydsl2/lambda.h"
#include "anydsl2/primop.h"
#include "anydsl2/util/box.h"

namespace anydsl2 {

class Any;
class Bottom;
class Def;
class Enter;
class Lambda;
class LEA;
class Leave;
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

struct LambdaLTGid : public std::binary_function<Lambda*, Lambda*, bool> {
    bool operator () (Lambda* l1, Lambda* l2) const { return l1->gid() < l2->gid(); };
};

typedef std::set<Lambda*, LambdaLTGid> LambdaSet;

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

    /*
     * constructor and destructor
     */

    World();
    ~World();

    /*
     * types
     */

#define ANYDSL2_UF_TYPE(T) const PrimType* type_##T(size_t length = 1) { \
    return length == 1 ? T##_ : unify(new PrimType(*this, PrimType_##T, length)); \
}
#include "anydsl2/tables/primtypetable.h"

    // primitive types

    /// Get PrimType.
    const PrimType* type(PrimTypeKind kind, size_t length = 1) {
        size_t i = kind - Begin_PrimType;
        assert(0 <= i && i < (size_t) Num_PrimTypes);
        return length == 1 ? primtypes_[i] : unify(new PrimType(*this, kind, length));
    }

    const Mem* mem() const { return mem_; }
    const Frame* frame() const { return frame_; }
    const Ptr* ptr(const Type* referenced_type, size_t length = 1) { 
        return unify(new Ptr(*this, referenced_type, length)); 
    }

    // sigmas

    /// Get unit AKA void AKA (unnamed) sigma().
    const Sigma* unit() const { return sigma0_; }
    /// Creates 'sigma()'.
    const Sigma* sigma0() { return sigma0_; }
    /// Creates 'sigma(t1)'.
    const Sigma* sigma1(const Type* t1) {
        const Type* types[1] = {t1};
        return sigma(types);
    }
    /// Creates 'sigma(t1, t2)'.
    const Sigma* sigma2(const Type* t1, const Type* t2) {
        const Type* types[2] = {t1, t2};
        return sigma(types);
    }
    /// Creates 'sigma(t1, t2, t3)'.
    const Sigma* sigma3(const Type* t1, const Type* t2, const Type* t3) {
        const Type* types[3] = {t1, t2, t3};
        return sigma(types);
    }
    const Sigma* sigma(ArrayRef<const Type*> elems) { return unify(new Sigma(*this, elems)); }

    /// Creates a fresh \em named sigma.
    Sigma* named_sigma(size_t size, const std::string& name = "");

    // pis

    /// Creates 'pi()'.
    const Pi* pi0() { return pi0_; }
    const Pi* pi1(const Type* t1) {
        const Type* types[1] = {t1};
        return pi(types);
    }
    /// Creates 'pi(t1, t2)'.
    const Pi* pi2(const Type* t1, const Type* t2) {
        const Type* types[2] = {t1, t2};
        return pi(types);
    }
    /// Creates 'pi(t1, t2, t3)'.
    const Pi* pi3(const Type* t1, const Type* t2, const Type* t3) {
        const Type* types[3] = {t1, t2, t3};
        return pi(types);
    }
    const Pi* pi(ArrayRef<const Type*> elems) { return unify(new Pi(*this, elems)); }

    const Generic* generic(size_t index) { return unify(new Generic(*this, index)); }
    const GenericRef* generic_ref(const Generic* generic, Lambda* lambda) { return unify(new GenericRef(*this, generic, lambda)); }

    /*
     * literals
     */

#define ANYDSL2_UF_TYPE(T) \
    const Def* literal_##T(T val, size_t length = 1) { return literal(val, length); }
#include "anydsl2/tables/primtypetable.h"
    const Def* literal(PrimTypeKind kind, Box value, size_t length = 1);
    const Def* literal(PrimTypeKind kind, int value, size_t length = 1);
    template<class T>
    const Def* literal(T value, size_t length = 1) { return literal(type2kind<T>::kind, Box(value), length); }

    const Def* zero(PrimTypeKind kind, size_t length = 1) { return literal(kind, 0, length); }
    const Def* zero(const Type*, size_t length = 1);
    const Def* one(PrimTypeKind kind, size_t length = 1) { return literal(kind, 1, length); }
    const Def* one(const Type*, size_t length = 1);
    const Def* allset(PrimTypeKind kind, size_t length = 1) {
        assert(is_int(kind) && "must not be a float");
        return literal(kind, -1, length);
    }
    const Def* allset(const Type*, size_t length = 1);

    const Def* any(const Type* type, size_t length = 1);
    const Def* any(PrimTypeKind kind, size_t length = 1) { return any(type(kind), length); }
    const Def* bottom(const Type* type, size_t length = 1);
    const Def* bottom(PrimTypeKind kind, size_t length = 1) { return bottom(type(kind), length); }

    /// Creates a vector of all true while the length is derived from @p def.
    const Def* true_mask(const Def* def) { return literal(true, def->length()); }
    const Def* true_mask(size_t length) { return literal(true, length); }

    /*
     * arithop, relop, convop
     */

    /// Creates an \p ArithOp or a \p RelOp.
    const Def* binop(int kind, const Def* cond, const Def* lhs, const Def* rhs, const std::string& name = "");
    const Def* binop(int kind, const Def* lhs, const Def* rhs, const std::string& name = "") {
        return binop(kind, true_mask(lhs), lhs, rhs, name);
    }

    const Def* arithop(ArithOpKind kind, const Def* cond, const Def* lhs, const Def* rhs, const std::string& name = "");
    const Def* arithop(ArithOpKind kind, const Def* lhs, const Def* rhs, const std::string& name = "") {
        return arithop(kind, true_mask(lhs), lhs, rhs, name);
    }
#define ANYDSL2_ARITHOP(OP) \
    const Def* arithop_##OP(const Def* cond, const Def* lhs, const Def* rhs, const std::string& name = "") { \
        return arithop(ArithOp_##OP, cond, lhs, rhs, name); \
    } \
    const Def* arithop_##OP(const Def* lhs, const Def* rhs, const std::string& name = "") { \
        return arithop(ArithOp_##OP, true_mask(lhs), lhs, rhs, name); \
    }
#include "anydsl2/tables/arithoptable.h"

    const Def* arithop_not(const Def* cond, const Def* def);
    const Def* arithop_not(const Def* def) { return arithop_not(true_mask(def), def); }
    const Def* arithop_minus(const Def* cond, const Def* def);
    const Def* arithop_minus(const Def* def) { return arithop_minus(true_mask(def), def); }

    const Def* relop(RelOpKind kind, const Def* cond, const Def* lhs, const Def* rhs, const std::string& name = "");
    const Def* relop(RelOpKind kind, const Def* lhs, const Def* rhs, const std::string& name = "") {
        return relop(kind, true_mask(lhs), lhs, rhs, name);
    }
#define ANYDSL2_RELOP(OP) \
    const Def* relop_##OP(const Def* cond, const Def* lhs, const Def* rhs, const std::string& name = "") { \
        return relop(RelOp_##OP, cond, lhs, rhs, name);  \
    } \
    const Def* relop_##OP(const Def* lhs, const Def* rhs, const std::string& name = "") { \
        return relop(RelOp_##OP, true_mask(lhs), lhs, rhs, name);  \
    }
#include "anydsl2/tables/reloptable.h"

    const Def* convop(ConvOpKind kind, const Def* cond, const Def* from, const Type* to, const std::string& name = "");
    const Def* convop(ConvOpKind kind, const Def* from, const Type* to, const std::string& name = "") {
        return convop(kind, true_mask(from), from, to, name);
    }
#define ANYDSL2_CONVOP(OP) \
    const Def* convop_##OP(const Def* from, const Def* cond, const Type* to, const std::string& name = "") { \
        return convop(ConvOp_##OP, cond, from, to, name); \
    } \
    const Def* convop_##OP(const Def* from, const Type* to, const std::string& name = "") { \
        return convop(ConvOp_##OP, true_mask(from), from, to, name); \
    }
#include "anydsl2/tables/convoptable.h"

    /*
     * aggregate stuff
     */

    const Def* tuple_extract(const Def* tuple, const Def* index, const std::string& name = "");
    const Def* tuple_extract(const Def* tuple, u32 index, const std::string& name = "");
    const Def* tuple_insert(const Def* tuple, const Def* index, const Def* value, const std::string& name = "");
    const Def* tuple_insert(const Def* tuple, u32 index, const Def* value, const std::string& name = "");
    const Def* tuple(ArrayRef<const Def*> args, const std::string& name = "") { return cse(new Tuple(*this, args, name)); }
    const Def* vector(ArrayRef<const Def*> args, const std::string& name = "") {
        if (args.size() == 1) return args[0];
        return cse(new Vector(*this, args, name)); 
    }
    const Def* vector2(const Def* arg1, const Def* arg2, const std::string& name = "") {
        const Def* args[] = { arg1, arg2 }; return vector(args, name);
    }
    const Def* vector4(const Def* arg1, const Def* arg2, const Def* arg3, const Def* arg4, const std::string& name = "") {
        const Def* args[] = { arg1, arg2, arg3, arg4 }; return vector(args, name);
    }
    const Def* vector(const Def* arg, size_t length = 1, const std::string& name = "");

    /*
     * memops
     */

    const Load* load(const Def* mem, const Def* ptr, const std::string& name = "");
    const Store* store(const Def* mem, const Def* ptr, const Def* val, const std::string& name = "");
    const Enter* enter(const Def* mem, const std::string& name = "");
    const Leave* leave(const Def* mem, const Def* frame, const std::string& name = "");
    const Slot* slot(const Type* type, const Def* frame, size_t index, const std::string& name = "");
    const LEA* lea(const Def* ptr, const Def* index, const std::string& name = "");

    /*
     * other stuff
     */

    const Def* select(const Def* cond, const Def* a, const Def* b, const std::string& name = "");
    const TypeKeeper* typekeeper(const Type* type, const std::string& name = "");

    Lambda* lambda(const Pi* pi, LambdaAttr attr = LambdaAttr(0), const std::string& name = "");
    Lambda* lambda(const Pi* pi, const std::string& name) { return lambda(pi, LambdaAttr(0), name); }
    Lambda* lambda(const std::string& name) { return lambda(pi0(), LambdaAttr(0), name); }
    Lambda* basicblock(const std::string& name = "");

    /// Generic \p PrimOp constructor; inherits name from \p in.
    const Def* rebuild(const PrimOp* in, ArrayRef<const Def*> ops, const Type* type);
    /// Generic \p PrimOp constructor; inherits type and name name from \p in.
    const Def* rebuild(const PrimOp* in, ArrayRef<const Def*> ops) { return rebuild(in, ops, in->type()); }
    /// Generic \p Type constructor.
    const Type* rebuild(const Type* in, ArrayRef<const Type*> elems);

    /*
     * optimizations
     */

    void dead_code_elimination();
    void unreachable_code_elimination();
    void unused_type_elimination();

    /// Performs dead code, unreachable code and unused type elimination.
    void cleanup();
    void opt();

    /*
     * getters
     */

    const PrimOpSet& primops() const { return primops_; }
    LambdaSet lambdas() const { return lambdas_; }
    TypeSet types() const { return types_; }
    size_t gid() const { return gid_; }

    /*
     * other
     */

    void dump(bool fancy = false);
    const Type* insert_type(const Type*);
    PrimOp* release(const PrimOp*);
    void reinsert(const PrimOp*);
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
    const Def* cse_base(const PrimOp*);
    template<class T> const T* cse(const T* primop) { return cse_base(primop)->template as<T>(); }

private:

    const Param* param(const Type* type, Lambda* lambda, size_t index, const std::string& name = "");

    const Type* keep_nocast(const Type* type);

    void dce_insert(const size_t pass, const Def* def);
    void ute_insert(const size_t pass, const Type* type);
    void uce_insert(const size_t pass, Lambda*);
    template<class S> static void unregister_uses(const size_t pass, S& set);
    template<class S> static void wipe_out(const size_t pass, S& set);

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
#define ANYDSL2_UF_TYPE(T) const PrimType* T##_;
#include "anydsl2/tables/primtypetable.h"
        };

        const PrimType* primtypes_[Num_PrimTypes];
    };

    friend class Lambda;
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
