#ifndef ANYDSL_WORLD_H
#define ANYDSL_WORLD_H

#include <cassert>
#include <string>
#include <queue>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include "anydsl/enums.h"
#include "anydsl/type.h"
#include "anydsl/util/autoptr.h"
#include "anydsl/util/box.h"

namespace anydsl {

class Any;
class Bottom;
class Def;
class Enter;
class Lambda;
class Leave;
class Pi;
class PrimLit;
class PrimOp;
class PrimType;
class Sigma;
class Slot;
class Type;

typedef boost::unordered_set<const PrimOp*, NodeHash, NodeEqual> PrimOpSet;
typedef boost::unordered_set<Lambda*> LambdaSet;
typedef boost::unordered_set<const Type*, NodeHash, NodeEqual> TypeSet;

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

#define ANYDSL_UF_TYPE(T) const PrimType* type_##T() const { return T##_; }
#include "anydsl/tables/primtypetable.h"

    // primitive types

    /// Get PrimType.
    const PrimType* type(PrimTypeKind kind) const {
        size_t i = kind - Begin_PrimType;
        assert(0 <= i && i < (size_t) Num_PrimTypes);
        return primTypes_[i];
    }

    const Mem* mem() const { return mem_; }
    const Frame* frame() const { return frame_; }
    const Ptr* ptr(const Type* ref) { return consume(new Ptr(ref))->as<Ptr>(); }

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
    const Sigma* sigma(ArrayRef<const Type*> elems) { return consume(new Sigma(*this, elems))->as<Sigma>(); }

    /// Creates a fresh \em named sigma.
    Sigma* named_sigma(size_t num, const std::string& name = "");

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
    const Pi* pi(ArrayRef<const Type*> elems) { return consume(new Pi(*this, elems))->as<Pi>(); }

    /*
     * literals
     */

#define ANYDSL_UF_TYPE(T) \
    const PrimLit* literal_##T(T val) { return literal(val); } \
    const PrimLit* literal_##T(Box box) { return literal(PrimType_##T, box); }
#include "anydsl/tables/primtypetable.h"
    const PrimLit* literal_u1(bool val) { return literal(PrimType_u1, Box(val)); }
    const PrimLit* literal(PrimTypeKind kind, Box boxue);
    const PrimLit* literal(PrimTypeKind kind, int value);
    template<class T>
    const PrimLit* literal(T value) { return literal(type2kind<T>::kind, Box(value)); }

    const PrimLit* zero(PrimTypeKind kind) { return literal(kind, 0); }
    const PrimLit* one(PrimTypeKind kind) { return literal(kind, 1); }
    const PrimLit* allset(PrimTypeKind kind) { 
        anydsl_assert(is_float(kind), "must not be a float"); 
        return literal(kind, -1); 
    }

    const Any* any(const Type* type);
    const Any* any(PrimTypeKind kind) { return any(type(kind)); }
    const Bottom* bottom(const Type* type);
    const Bottom* bottom(PrimTypeKind kind) { return bottom(type(kind)); }

    /*
     * arithop, relop, convop
     */

    /// Creates an \p ArithOp or a \p RelOp.
    const Def* binop(int kind, const Def* lhs, const Def* rhs);

    const Def* arithop(ArithOpKind kind, const Def* lhs, const Def* rhs);
#define ANYDSL_ARITHOP(OP) const Def* arithop_##OP(const Def* lhs, const Def* rhs) { return arithop(ArithOp_##OP, lhs, rhs); }
#include "anydsl/tables/arithoptable.h"

    const Def* relop(RelOpKind kind, const Def* lhs, const Def* rhs);
#define ANYDSL_RELOP(OP) const Def* relop_##OP(const Def* lhs, const Def* rhs) { return relop(RelOp_##OP, lhs, rhs); }
#include "anydsl/tables/reloptable.h"

    const Def* convop(ConvOpKind kind, const Type* to, const Def* from);
#define ANYDSL_CONVOP(OP) const Def* convop_##OP(const Type* to, const Def* from) { return convop(ConvOp_##OP, to, from); }
#include "anydsl/tables/convoptable.h"

    /*
     * tuple stuff
     */

    const Def* extract(const Def* tuple, u32 i);
    const Def* insert(const Def* tuple, u32 i, const Def* value);
    const Def* tuple(ArrayRef<const Def*> args);

    /*
     * memops
     */

    const Def* load(const Def* mem, const Def* ptr);
    const Def* store(const Def* mem, const Def* ptr, const Def* val);
    const Enter* enter(const Def* mem);
    const Leave* leave(const Def* mem, const Def* frame);
    const Slot* slot(const Enter* enter, const Type* type);

    /*
     * other stuff
     */

    const Def* select(const Def* cond, const Def* tdef, const Def* fdef);
    const Def* typekeeper(const Type* type);

    Lambda* lambda(const Pi* pi, uint32_t flags = 0);
    Lambda* lambda(uint32_t flags = 0) { return lambda(pi0()); }

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
    //void replace(const PrimOp* what, const PrimOp* with);
    const Def* update(const Def* what, size_t i, const Def* op);
    const Def* update(const Def* what, Array<const Def*> ops);
    const PrimOp* consume(const PrimOp* primop);
    const Type* consume(const Type* def);
    PrimOp* release(const PrimOp* primop);

private:

    void dce_insert(const Def* def);
    void ute_insert(const Type* type);
    void uce_insert(Lambda* lambda);

    PrimOpSet primops_;
    LambdaSet lambdas_;
    TypeSet types_;

    size_t gid_counter_;
    const Sigma* sigma0_;///< sigma().
    const Pi* pi0_;      ///< pi().
    const Mem* mem_;
    const Frame* frame_;

    union {
        struct {
#define ANYDSL_UF_TYPE(T) const PrimType* T##_;
#include "anydsl/tables/primtypetable.h"
        };

        const PrimType* primTypes_[Num_PrimTypes];
    };
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
