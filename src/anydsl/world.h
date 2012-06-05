#ifndef ANYDSL_WORLD_H
#define ANYDSL_WORLD_H

#include <cassert>
#include <string>

#include <boost/unordered_set.hpp>

#include "anydsl/enums.h"
#include "anydsl/type.h"
#include "anydsl/util/autoptr.h"
#include "anydsl/util/box.h"
#include "anydsl/util/for_all.h"

namespace anydsl {

class Def;
class ErrorLit;
class Lambda;
class Pi;
class PrimLit;
class PrimType;
class Sigma;
class Terminator;
class Type;
class Value;
class Undef;

//------------------------------------------------------------------------------

typedef boost::unordered_set<Value*, ValueHash, ValueEqual> ValueMap;
typedef boost::unordered_set<const Type*, TypeHash, TypeEqual> TypeMap;
typedef std::vector<Sigma*> NamedSigmas;
typedef boost::unordered_set<Lambda*> Lambdas;

//------------------------------------------------------------------------------

/**
 * The World represents the whole program and manages creation and destruction of AIRNodes.
 * In particular, the following things are done by this class:
 *  - Type unification:
 *      There exists only one unique type for PrimType%s, Pi%s and \em unnamed Sigma%s.
 *      These types are hashed into internal maps for fast access.
 *      The getters just calculate a hash and lookup the type, if it is already present, or create a new one otherwise.
 *      There also exists the concept of \em named \p Sigma%s to allow for recursive types.
 *      These types are \em not unified, i.e., each instance is by definition a different type;
 *      thus, two different pointers of the same named sigma are always considered different types.
 *  - Value unification:
 *      This is a built-in mechanism for the following things:
 *      - constant pooling
 *      - constant folding 
 *      - common subexpression elimination
 *      - dead code elimination
 *      - canonicalization of expressions
 *      - several local optimizations
 *      PrimOp%s do not explicitly belong to a Lambda.
 *      Instead they either implicitly belong to a Lambda--when 
 *      they (possibly via multiple levels of indirection) depend on a Lambda's Param--or they are dead. 
 *      Use \p cleanup to remove dead code.
 *  - Lambda%s are register here in order to not have dangling pointers 
 *  and to perform unreachable code elimination.
 *  The aforementioned \p cleanup will also delete these lambdas.
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

#define ANYDSL_U_TYPE(T) const PrimType* type_##T() const { return T##_; }
#define ANYDSL_F_TYPE(T) const PrimType* type_##T() const { return T##_; }
#include "anydsl/tables/primtypetable.h"

    // primitive types

    /// Get PrimType.
    const PrimType* type(PrimTypeKind kind) const { 
        size_t i = kind - Begin_PrimType;
        assert(0 <= i && i < (size_t) Num_PrimTypes); 
        return primTypes_[i];
    }

    const NoRet* noret(const Pi* pi) { return tfind(new NoRet(*this, pi)); }

    // sigmas

    /// Get unit AKA void AKA (unnamed) sigma(). 
    const Sigma* unit() const { return unit_; }
    /// Creates 'sigma()'.
    const Sigma* sigma0() { return unit_; }
    /// Creates 'sigma(t1)'.
    const Sigma* sigma1(const Type* t1) { return sigma((const Type*[]){t1}); }
    /// Creates 'sigma(t1, t2)'.
    const Sigma* sigma2(const Type* t1, const Type* t2) { return sigma((const Type*[]){t1, t2}); }
    /// Creates 'sigma(t1, t2, t3)'.
    const Sigma* sigma3(const Type* t1, const Type* t2, const Type* t3) { return sigma((const Type*[]){t1, t2, t3}); }

    const Sigma* sigma(const Type* const* begin, const Type* const* end) { 
        return tfind(new Sigma(*this, begin, end)); 
    }
    const Sigma* sigma(const Types& types) { return sigma(types.begin().base(), types.end().base()); }

    template<size_t N>
    const Sigma* sigma(const Type* const (&array)[N]) { 
        return sigma(array, array + N); 
    }

    /// Creates a fresh \em named sigma.
    Sigma* namedSigma(const std::string& name = "");

    // pis

    /// Creates 'pi()'.
    const Pi* pi0() { return pi0_; }
    /// Creates 'pi(t1)'.
    const Pi* pi1(const Type* t1) { return pi(sigma1(t1)); }
    /// Creates 'pi(t1, t2)'.
    const Pi* pi2(const Type* t1, const Type* t2) { return pi(sigma2(t1, t2)); }
    /// Creates 'pi(t1, t2, t3)'.
    const Pi* pi3(const Type* t1, const Type* t2, const Type* t3) { return pi(sigma3(t1, t2, t3)); }

    const Pi* pi(const Sigma* sigma) { return tfind(new Pi(sigma)); }
    const Pi* pi(const Type* const* begin, const Type* const* end) { return tfind(new Pi(sigma(begin, end))); }
    const Pi* pi(const Types& types) { return pi(sigma(types)); }
    template<size_t N>
    const Pi* pi(const Type* const (&array)[N]) { return pi(sigma(array)); }


    /*
     * literals
     */

#define ANYDSL_U_TYPE(T) \
    PrimLit* literal_##T(T val) { return literal(val); } \
    PrimLit* literal_##T(Box val) { return literal(PrimType_##T, val); }
#define ANYDSL_F_TYPE(T) \
    PrimLit* literal_##T(T val) { return literal(val); } \
    PrimLit* literal_##T(Box val) { return literal(PrimType_##T, val); }
#include "anydsl/tables/primtypetable.h"

    PrimLit* literal(PrimLitKind kind, Box value);
    PrimLit* literal(PrimTypeKind kind, Box value) { return literal(type2lit(kind), value); }
    PrimLit* literal(const PrimType* p, Box value);
    template<class T>
    PrimLit* literal(T value) { return literal(type2kind<T>::kind, Box(value)); }
    Undef* undef(const Type* type);
    Undef* undef(PrimTypeKind kind) { return undef(type(kind)); }
    ErrorLit* literal_error(const Type* type);
    ErrorLit* literal_error(PrimTypeKind kind) { return literal_error(type(kind)); }

    /*
     * create
     */

    Lambda* createLambda(const Pi* type = 0);
    Jump* createJump(Def* to, Def* const* arg_begin, Def* const* arg_end);
    Jump* createJump(Def* to) { 
        return (Jump*) createJump(to, (Def**) 0, (Def**) 0); 
    }
    Jump* createBranch(Def* cond, Def* tto, Def* fto, Def* const* arg_begin, Def* const* arg_end);
    Jump* createBranch(Def* cond, Def* tto, Def* fto);

    Value* createArithOp(ArithOpKind kind, Def* ldef, Def* rdef);
    Value* createRelOp(RelOpKind kind, Def* ldef, Def* rdef);
    Value* createProj(Def* tuple, PrimLit* i);
    Value* createInsert(Def* tuple, PrimLit* i, Def* value);
    Value* createSelect(Def* cond, Def* tdef, Def* fdef);
    Value* createTuple(Def* const* begin, Def* const* end);
    Value* createTuple(const std::vector<Def*>& defs) { 
        return createTuple(defs.begin().base(), defs.end().base());
    }
    template<size_t N>
    Value* createTuple(Def* const (&array)[N]) { return createTuple(array, array + N); }


    /*
     * optimize
     */

    /// Performs dead code and unreachable code elimination.
    void cleanup();

private:

    Value* findValue(Value* value);
    const Type* findType(const Type* type);

    template<class T> 
    T* tfind(T* type) { return (T*) findType(type); }

    template<class T> 
    T* vfind(T* value) { return (T*) findValue(value); }

    Value* tryFold(IndexKind kind, Def* ldef, Def* rdef);

    template<class T, class C>
    static void kill(C& container);

    ValueMap values_;
    TypeMap types_;

    const Sigma* unit_; ///< sigma().
    const Pi* pi0_;     ///< pi().

    union {
        struct {
#define ANYDSL_U_TYPE(T) const PrimType* T##_;
#define ANYDSL_F_TYPE(T) const PrimType* T##_;
#include "anydsl/tables/primtypetable.h"
        };

        const PrimType* primTypes_[Num_PrimTypes];
    };

    NamedSigmas namedSigmas_;
    Lambdas lambdas_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
