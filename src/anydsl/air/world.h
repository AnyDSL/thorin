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
class ErrorType;
class Goto;
class Invoke;
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

typedef boost::unordered_multimap<uint64_t, Value*> ValueMap;
typedef boost::unordered_multimap<uint64_t, Pi*> PiMap;
typedef boost::unordered_multimap<uint64_t, Sigma*> SigmaMap;
typedef std::vector<Sigma*> NamedSigmas;
typedef boost::unordered_set<Lambda*> Lambdas;

//------------------------------------------------------------------------------

/**
 * This class manages the following things for the whole program:
 *  - Type unification:
 *      There exists only one unique type for PrimType%s, Pi%s and \em unnamed Sigma%s.
 *      These types are hashed into internal maps for fast access.
 *      The getters just calculate a hash and lookup the type, if it is already present, or create a new one otherwise.
 *      There also exists the concept of \em named \p Sigma%s to allow for recursive types.
 *      These types are \em not unified, i.e., each instance is by definition a different type;
 *      thus, two different pointers of the same named sigma are considered different types.
 *  - PrimOp unification:
 *      This is a built-in mechanism for the following things:
 *      - common subexpression elimination
 *      - constant folding 
 *      - copy propagation
 *      - dead code elimination
 *      - canonicalization of expressions
 *      - several local optimizations
 *      PrimOp%s do not explicitly belong to a Lambda.
 *      Instead they either implicitly belong to a Lambda 
 *      when they (possibly via multiple steps) depend on an Lambda's Param or they are dead. 
 *      Use \p cleanup to remove dead code.
 *  - Lambda%s are register here in order to not have dangling pointers 
 *  and to perform unreachable code elimination.
 *  The aforementioned \p cleanup will also delete these lambdas.
 *
 *  You can create several worlds. 
 *  All worlds are independent from each other.
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

#define ANYDSL_U_TYPE(T) PrimType* type_##T() const { return T##_; }
#define ANYDSL_F_TYPE(T) PrimType* type_##T() const { return T##_; }
#include "anydsl/tables/primtypetable.h"

    const ErrorType* type_error() { return type_error_; }

    // primitive types

    /// Get PrimType.
    PrimType* type(PrimTypeKind kind) const { 
        size_t i = kind - Begin_PrimType;
        assert(0 <= i && i < (size_t) Num_PrimTypes); 
        return primTypes_[i];
    }

    // sigmas

    /// Get unit AKA sigma() AKA void.
    const Sigma* unit() const { return unit_; }
    /// Creates a fresh \em named sigma.
    Sigma* sigma(const std::string& name = "");
    template<class T> 
    const Sigma* sigma(T container, bool named = false) { return sigma(container.begin(), container.end(), named); }
    template<size_t N>
    const Sigma* sigma(const Type* (&array)[N], bool named = false) { return sigma(array, array + N, named); }
    /** 
     * @brief Get named or unnamed \p Sigma (according to \p named) with element types of given range.
     * 
     * @param T Must be a forward iterator which yields a const Type* upon using unary 'operator *'.
     * @param begin Iterator which points to the beginning of the range.
     * @param end Iterator which points to one element past the end of the range.
     * @param named Whether you want to receive a named or unnamed Sigma.
     * 
     * @return The Sigma.
     */
    template<class T>
    const Sigma* sigma(T begin, T end, bool named = false) {
        if (named) {
            Sigma* res = new Sigma(*this, begin, end, named);
            namedSigmas_.push_back(res);

            return res;
        }

        return getSigmaOrPi<Sigma>(sigmas_, begin, end);
    }

    // pis

    /// Creates 'pi()'.
    const Pi* pi() const { return pi_; }
    template<class T> 
    const Pi* pi(T container) { return pi(container.begin(), container.end()); }
    template<size_t N>
    const Pi* pi(const Type* (&array)[N]) { return pi(array, array + N); }
    /** 
     * @brief Get named or unnamed \p Sigma (according to \p named) with element types of given range.
     * 
     * @param T Must be a forward iterator which yields a const Type* upon using unary 'operator *'.
     * @param begin Iterator which points to the beginning of the range.
     * @param end Iterator which points to one element past the end of the range.
     * @param named Whether you want to receive a named or unnamed Sigma.
     * 
     * @return The Sigma.
     */
    template<class T>
    const Pi* pi(T begin, T end) { return getSigmaOrPi<Pi>(pis_, begin, end); }

    /*
     * literals
     */

    template<class T>
    PrimLit* literal(T value) { return literal(type2kind<T>::kind, Box(value)); }
    PrimLit* literal(PrimTypeKind kind, Box value) { return literal(type2lit(kind), value); }
    PrimLit* literal(PrimLitKind kind, Box value);
    Undef* undef(const Type* type);
    ErrorLit* literal_error(const Type* type);

    /*
     * create
     */

    Lambda* createLambda(const Pi* type = 0);
    Goto* createGoto(Lambda* parent, Lambda* to);
    Terminator* createBranch(Lambda* parent, Def* cond, Lambda* tto, Lambda* fto);
    Invoke* createInvoke(Lambda* parent, Def* fct);

    Value* createArithOp(ArithOpKind kind, Def* ldef, Def* rdef);
    Value* createRelOp(RelOpKind kind, Def* ldef, Def* rdef);

    /*
     * optimize
     */

    /// Performs dead code and unreachable code elimination.
    void cleanup();

private:

    template<class T, class C>
    static void kill(C& container);

    template<class T, class M, class Iter>
    const T* getSigmaOrPi(M& map, Iter begin, Iter end);

    AutoPtr<const ErrorType> type_error_;
    const Pi* pi_; ///< pi().
    const Sigma* unit_; ///< sigma().

    union {
        struct {
#define ANYDSL_U_TYPE(T) PrimType* T##_;
#define ANYDSL_F_TYPE(T) PrimType* T##_;
#include "anydsl/tables/primtypetable.h"
        };

        PrimType* primTypes_[Num_PrimTypes];
    };

    ValueMap values_;
    PiMap pis_;
    SigmaMap sigmas_;
    NamedSigmas namedSigmas_;
    Lambdas lambdas_;
};

template<class T, class M, class Iter>
const T* World::getSigmaOrPi(M& map, Iter begin, Iter end) {
    uint64_t h = T::hash(begin, end);

    FOREACHT(p, map.equal_range(h))
        if (p.second->equal(begin, end))
            return p.second;

    std::cout << h << std::endl;
    return map.insert(std::make_pair(h, new T(*this, begin, end)))->second;
}

} // namespace anydsl

#endif // ANYDSL_SUPPORT_WORLD_H
