#ifndef ANYDSL2_WORLD_H
#define ANYDSL2_WORLD_H

#include <cassert>
#include <functional>
#include <queue>
#include <string>

#include <boost/unordered_set.hpp>

#include "anydsl2/enums.h"
#include "anydsl2/lambda.h"
#include "anydsl2/primop.h"
#include "anydsl2/util/box.h"

namespace anydsl2 {

class Any;
class Bottom;
class CCall;
class Def;
class Enter;
class Lambda;
class Leave;
class Opaque;
class Pi;
class PrimLit;
class PrimOp;
class PrimType;
class Sigma;
class Slot;
class Type;
class TypeKeeper;

typedef boost::unordered_set<const PrimOp*, PrimOpHash, PrimOpEqual> PrimOpSet;
typedef boost::unordered_set<const Type*, TypeHash, TypeEqual> TypeSet;

struct LambdaLT : public std::binary_function<Lambda*, Lambda*, bool> {
    bool operator () (Lambda* l1, Lambda* l2) const { return l1->gid() < l2->gid(); };
};

typedef std::set<Lambda*, LambdaLT> LambdaSet;

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

#define ANYDSL2_UF_TYPE(T) const PrimType* type_##T() const { return T##_; }
#include "anydsl2/tables/primtypetable.h"

    // primitive types

    /// Get PrimType.
    const PrimType* type(PrimTypeKind kind) const {
        size_t i = kind - Begin_PrimType;
        assert(0 <= i && i < (size_t) Num_PrimTypes);
        return primtypes_[i];
    }

    const Mem* mem() const { return mem_; }
    const Frame* frame() const { return frame_; }
    const Ptr* ptr(const Type* ref);

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
    const Sigma* sigma(ArrayRef<const Type*> elems);

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
    const Pi* pi(ArrayRef<const Type*> elems);

    const Generic* generic(size_t index);
    const Opaque* opaque(ArrayRef<const Type*> elems, ArrayRef<uint32_t> flags);
    const Opaque* opaque(const Type* type, ArrayRef<uint32_t> flags) {
        const Type* types[1] = { type }; 
        return opaque(types, flags);
    }
    const Opaque* opaque1(const Type* type, uint32_t flag1) { 
        uint32_t flags[1] = { flag1 };
        return opaque(type, flags);
    }
    const Opaque* opaque2(const Type* type, uint32_t flag1, uint32_t flag2) { 
        uint32_t flags[2] = { flag1, flag2 };
        return opaque(type, flags);
    }
    const Opaque* opaque3(const Type* type, uint32_t flag1, uint32_t flag2, uint32_t flag3) { 
        uint32_t flags[3] = { flag1, flag2, flag3 };
        return opaque(type, flags);
    }

    /*
     * literals
     */

#define ANYDSL2_UF_TYPE(T) \
    const PrimLit* literal_##T(T val) { return literal(val); } \
    const PrimLit* literal_##T(Box box) { return literal(PrimType_##T, box); }
#include "anydsl2/tables/primtypetable.h"
    const PrimLit* literal_u1(bool val) { return literal(PrimType_u1, Box(val)); }
    const PrimLit* literal(PrimTypeKind kind, Box value);
    const PrimLit* literal(PrimTypeKind kind, int value);
    const PrimLit* literal(const Type* type, int value);
    template<class T>
    const PrimLit* literal(T value) { return literal(type2kind<T>::kind, Box(value)); }

    const PrimLit* zero(PrimTypeKind kind) { return literal(kind, 0); }
    const PrimLit* zero(const Type*);
    const PrimLit* one(PrimTypeKind kind) { return literal(kind, 1); }
    const PrimLit* one(const Type*);
    const PrimLit* allset(PrimTypeKind kind) {
        assert(is_float(kind) && "must not be a float");
        return literal(kind, -1);
    }
    const PrimLit* allset(const Type*);

    const Any* any(const Type* type);
    const Any* any(PrimTypeKind kind) { return any(type(kind)); }
    const Bottom* bottom(const Type* type);
    const Bottom* bottom(PrimTypeKind kind) { return bottom(type(kind)); }

    /*
     * arithop, relop, convop
     */

    /// Creates an \p ArithOp or a \p RelOp.
    const Def* binop(int kind, const Def* lhs, const Def* rhs, const std::string& name = "");

    const Def* arithop(ArithOpKind kind, const Def* lhs, const Def* rhs, const std::string& name = "");
#define ANYDSL2_ARITHOP(OP) \
    const Def* arithop_##OP(const Def* lhs, const Def* rhs, const std::string& name = "") { \
        return arithop(ArithOp_##OP, lhs, rhs, name); \
    }
#include "anydsl2/tables/arithoptable.h"

    const Def* arithop_not(const Def* def);
    const Def* arithop_minus(const Def* def);

    const Def* relop(RelOpKind kind, const Def* lhs, const Def* rhs, const std::string& name = "");
#define ANYDSL2_RELOP(OP) \
    const Def* relop_##OP(const Def* lhs, const Def* rhs, const std::string& name = "") { \
        return relop(RelOp_##OP, lhs, rhs, name);  \
    }
#include "anydsl2/tables/reloptable.h"

    const Def* convop(ConvOpKind kind, const Type* to, const Def* from, const std::string& name = "");
#define ANYDSL2_CONVOP(OP) \
    const Def* convop_##OP(const Type* to, const Def* from, const std::string& name) { \
        return convop(ConvOp_##OP, to, from, name); \
    }
#include "anydsl2/tables/convoptable.h"

    /*
     * tuple stuff
     */

    const Def* extract(const Def* tuple, const Def* index, const std::string& name = "");
    const Def* extract(const Def* tuple, u32 index, const std::string& name = "");
    const Def* insert(const Def* tuple, const Def* index, const Def* value, const std::string& name = "");
    const Def* insert(const Def* tuple, u32 index, const Def* value, const std::string& name = "");
    const Def* tuple(ArrayRef<const Def*> args, const std::string& name = "");

    /*
     * memops
     */

    const Def* load(const Def* mem, const Def* ptr, const std::string& name = "");
    const Def* store(const Def* mem, const Def* ptr, const Def* val, const std::string& name = "");
    const Enter* enter(const Def* mem, const std::string& name = "");
    const Leave* leave(const Def* mem, const Def* frame, const std::string& name = "");
    const Slot* slot(const Type* type, size_t index, const Def* frame, const std::string& name = "");
    const CCall* c_call(const std::string& callee, const Def* mem, ArrayRef<const Def*> args,
                        const Type* rettype, bool vararg = false, const std::string& name = "");

    /*
     * other stuff
     */

    const Def* select(const Def* cond, const Def* a, const Def* b, const std::string& name = "");
    const TypeKeeper* typekeeper(const Type* type, const std::string& name = "");

    Lambda* lambda(const Pi* pi, LambdaAttr attr = LambdaAttr(0), const std::string& name = "");
    Lambda* lambda(const Pi* pi, const std::string& name) { return lambda(pi, LambdaAttr(0), name); }
    Lambda* lambda(const std::string& name) { return lambda(pi0(), LambdaAttr(0), name); }
    Lambda* basicblock(const std::string& name);

    /// Generic \p PrimOp constructor.
    const Def* rebuild(const PrimOp* in, ArrayRef<const Def*> ops, const std::string& name);
    /// Generic \p PrimOp constructor; inherits name from \p in.
    const Def* rebuild(const PrimOp* in, ArrayRef<const Def*> ops);

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

    /*
     * other
     */

    void dump(bool fancy = false);
    const Def* update(const Def* what, size_t x, const Def* op);
    const Type* insert_type(const Type* def);
    PrimOp* release(const PrimOp* primop);
    size_t new_pass() { return pass_counter_++; }

#ifndef NDEBUG
    void breakpoint(size_t number) { breakpoints_.insert(number); }
#endif

protected:

    template<class T>
    const T* keep(const T* type) {
        std::pair<TypeSet::iterator, bool> tp = types_.insert(type);
        assert(tp.second);
        typekeeper(type);
        return type->template as<T>();
    }
    template<class T, class U> const U* unify(const T& tuple);

    template<class T, class U>
    const U* cse(const T& tuple, const std::string& name) {
        PrimOpSet::iterator i = primops_.find(tuple, std::ptr_fun<const T&, size_t>(hash_tuple),
                                                     std::ptr_fun<const T&, const Node*, bool>(smart_eq<T, U>));
        if (i != primops_.end()) return (*i)->as<U>();

        std::pair<PrimOpSet::iterator, bool> p = primops_.insert(new U(tuple, name));
        assert(p.second && "hash/equal broken");
        const U* u = (*p.first)->as<U>();
        cse_break(u);
        return u;
    }

    void cse_break(const PrimOp* primop);

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
    boost::unordered_set<size_t> breakpoints_;
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
