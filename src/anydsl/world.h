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
class Type;
class Def;
class Undef;

//------------------------------------------------------------------------------

typedef std::vector<Sigma*> NamedSigmas;

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

    const NoRet* noret(const Pi* pi) { return find(new NoRet(*this, pi)); }

    // sigmas

    /// Get unit AKA void AKA (unnamed) sigma(). 
    const Sigma* unit() const { return unit_; }
    /// Creates 'sigma()'.
    const Sigma* sigma0() { return unit_; }
    /// Creates 'sigma(t1)'.
    const Sigma* sigma1(const Type* t1) { 
        const Type* types[] = {t1};
        return sigma(types); 
    }
    /// Creates 'sigma(t1, t2)'.
    const Sigma* sigma2(const Type* t1, const Type* t2) { 
        const Type* types[] = {t1, t2};
        return sigma(types);
    }
    /// Creates 'sigma(t1, t2, t3)'.
    const Sigma* sigma3(const Type* t1, const Type* t2, const Type* t3) { 
        const Type* types[] = {t1, t2, t3};
        return sigma(types);
    }
    const Sigma* sigma(const Type* const* begin, const Type* const* end) { 
        return find(new Sigma(*this, begin, end)); 
    }
    template<size_t N>
    const Sigma* sigma(const Type* const (&array)[N]) { 
        return sigma(array, array + N); 
    }

    /// Creates a fresh \em named sigma.
    Sigma* namedSigma(size_t num, const std::string& name = "");

    // pis

    /// Creates 'pi()'.
    const Pi* pi0() { return pi0_; }
    const Pi* pi1(const Type* t1) { 
        const Type* types[] = {t1};
        return pi(types); 
    }
    /// Creates 'pi(t1, t2)'.
    const Pi* pi2(const Type* t1, const Type* t2) { 
        const Type* types[] = {t1, t2};
        return pi(types);
    }
    /// Creates 'pi(t1, t2, t3)'.
    const Pi* pi3(const Type* t1, const Type* t2, const Type* t3) { 
        const Type* types[] = {t1, t2, t3};
        return pi(types);
    }
    const Pi* pi(const Type* const* begin, const Type* const* end) { 
        return find(new Pi(*this, begin, end)); 
    }
    template<size_t N>
    const Pi* pi(const Type* const (&array)[N]) { 
        return pi(array, array + N); 
    }

    /*
     * literals
     */

#define ANYDSL_U_TYPE(T) \
    const PrimLit* literal_##T(T val) { return literal(val); } \
    const PrimLit* literal_##T(Box val) { return literal(PrimType_##T, val); }
#define ANYDSL_F_TYPE(T) \
    const PrimLit* literal_##T(T val) { return literal(val); } \
    const PrimLit* literal_##T(Box val) { return literal(PrimType_##T, val); }
#include "anydsl/tables/primtypetable.h"

    const PrimLit* literal(PrimLitKind kind, Box value);
    const PrimLit* literal(PrimTypeKind kind, Box value) { return literal(type2lit(kind), value); }
    const PrimLit* literal(const PrimType* p, Box value);
    template<class T>
    const PrimLit* literal(T value) { return literal(type2kind<T>::kind, Box(value)); }
    const Undef* undef(const Type* type);
    const Undef* undef(PrimTypeKind kind) { return undef(type(kind)); }
    const ErrorLit* literal_error(const Type* type);
    const ErrorLit* literal_error(PrimTypeKind kind) { return literal_error(type(kind)); }

    /*
     * create
     */

    const Jump* createJump(const Def* to, const Def* const* arg_begin, const Def* const* arg_end);
    const Jump* createJump(const Def* to) { 
        return (Jump*) createJump(to, 0, 0); 
    }
    template<size_t N>
    const Jump* createJump(const Def* to, const Def* const (&array)[N]) { return createJump(to, array, array + N); }

    const Jump* createBranch(const Def* cond, const Def* tto, const Def* fto);
    const Jump* createBranch(const Def* cond, const Def* tto, const Def* fto, 
                             const Def* const* arg_begin, const Def* const* arg_end);

    const Def* createArithOp(ArithOpKind kind, const Def* ldef, const Def* rdef);
    const Def* createRelOp(RelOpKind kind, const Def* ldef, const Def* rdef);
    const Def* createExtract(const Def* tuple, const PrimLit* i);
    const Def* createInsert(const Def* tuple, const PrimLit* i, const Def* value);
    const Def* createSelect(const Def* cond, const Def* tdef, const Def* fdef);
    const Def* createTuple(const Def* const* begin, const Def* const* end);
    template<size_t N>
    const Def* createTuple(const Def* const (&array)[N]) { return createTuple(array, array + N); }

    const Lambda* finalize(const Lambda* lambda);

    void setLive(const Def* def) { live_.insert(def); }

    /// Performs dead code and unreachable code elimination.
    void cleanup();

private:

    typedef boost::unordered_set<const Def*, DefHash, DefEqual> DefMap;
    DefMap::iterator remove(DefMap::iterator i);
    const Def* findDef(const Def* def);

    void insert(const Def* def);

    template<class T> 
    const T* find(const T* val) { return (T*) findDef(val); }

    const Def* tryFold(IndexKind kind, const Def* ldef, const Def* rdef);

    DefMap defs_;

    typedef boost::unordered_set<const Def*> LiveSet;
    LiveSet live_;
    NamedSigmas namedSigmas_;

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
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
