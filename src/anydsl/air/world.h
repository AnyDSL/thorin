#ifndef ANYDSL_SUPPORT_WORLD_H
#define ANYDSL_SUPPORT_WORLD_H

#include <cassert>
#include <string>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include "anydsl/air/enums.h"
#include "anydsl/air/type.h"
#include "anydsl/util/autoptr.h"
#include "anydsl/util/box.h"
#include "anydsl/util/foreach.h"

namespace anydsl {

class Def;
class ErrorLit;
class Jump;
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

typedef boost::unordered_map<ValueNumber, Value*> ValueMap;
typedef boost::unordered_map<ValueNumber, const Type*> TypeMap;
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

    const NoRet* noret(const Pi* pi) { return findType<NoRet>(NoRet::VN(pi)); }

    // sigmas

    /// Get unit AKA void AKA (unnamed) sigma(). 
    const Sigma* unit() const { return unit_; }
    /// Creates 'sigma()'.
    const Sigma* sigma0();
    /// Creates 'sigma(t1)'.
    const Sigma* sigma1(const Type* t1);
    /// Creates 'sigma(t1, t2)'.
    const Sigma* sigma2(const Type* t1, const Type* t2);
    /// Creates 'sigma(t1, t2, t3)'.
    const Sigma* sigma3(const Type* t1, const Type* t2, const Type* t3);
    template<class T> 
    const Sigma* sigma(T container) { return sigma(container.begin(), container.end()); }
    template<size_t N>
    const Sigma* sigma(const Type* (&array)[N]) { return sigma(array, array + N); }
    /** 
     * @brief Get \em unamed \p Sigma with element types of given range.
     * 
     * @param T Must be a forward iterator which yields a const Type* upon using unary 'operator *'.
     * @param begin Iterator which points to the beginning of the range.
     * @param end Iterator which points to one element past the end of the range.
     * 
     * @return The Sigma.
     */
    template<class T>
    const Sigma* sigma(T begin, T end) { return findType<Sigma>(Sigma::VN(begin, end)); }
    /// Creates a fresh \em named sigma.
    Sigma* namedSigma(const std::string& name = "");

    // pis

    /// Creates 'pi()'.
    const Pi* pi0() { return pi0_; }
    /// Creates 'pi(t1)'.
    const Pi* pi1(const Type* t1);
    /// Creates 'pi(t1, t2)'.
    const Pi* pi2(const Type* t1, const Type* t2);
    /// Creates 'pi(t1, t2, t3)'.
    const Pi* pi3(const Type* t1, const Type* t2, const Type* t3);
    template<class T> 
    const Pi* pi(T container) { return pi(container.begin(), container.end()); }
    template<size_t N>
    const Pi* pi(const Type* (&array)[N]) { return pi(array, array + N); }
    /** 
     * @brief Get \p Pi with element types of given range.
     * 
     * @param T Must be a forward iterator which yields a const Type* upon using unary 'operator *'.
     * @param begin Iterator which points to the beginning of the range.
     * @param end Iterator which points to one element past the end of the range.
     * 
     * @return The Sigma.
     */
    template<class T>
    const Pi* pi(T begin, T end) { return findType<Pi>(Pi::VN(begin, end)); }

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
    Jump* createJump(Lambda* parent, Def* to);
    Jump* createBranch(Lambda* parent, Def* cond, Def* tto, Def* fto);

    Value* createArithOp(ArithOpKind kind, Def* ldef, Def* rdef);
    Value* createRelOp(RelOpKind kind, Def* ldef, Def* rdef);
    Value* createProj(Def* tuple, PrimLit* i);
    Value* createSelect(Def* cond, Def* tdef, Def* fdef);

    /*
     * optimize
     */

    /// Performs dead code and unreachable code elimination.
    void cleanup();

private:

    template<class T> T* findValue(const ValueNumber& vn);
    template<class T> const T* findType(const ValueNumber& vn);
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

template<class T>
const T* World::findType(const ValueNumber& vn) {
    TypeMap::iterator i = types_.find(vn);
    if (i != types_.end())
        return scast<T>(i->second);

    const T* t = new T(*this, vn);
    types_[vn] = t;

    return t;
}

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_SUPPORT_WORLD_H
