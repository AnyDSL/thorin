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
class Error;
class Lambda;
class Pi;
class PrimLit;
class PrimType;
class Sigma;
class Type;
class Def;
class Undef;

typedef std::vector<const Param*> Params;
typedef boost::unordered_set<const Def*, DefHash, DefEqual> DefSet;

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

    // sigmas

    /// Get unit AKA void AKA (unnamed) sigma(). 
    const Sigma* unit() const { return unit_; }
    /// Creates 'sigma()'.
    const Sigma* sigma0() { return unit_; }
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
    const Error* error(const Type* type);
    const Error* error(PrimTypeKind kind) { return error(type(kind)); }

    /*
     * create
     */

    const Def* arithOp(ArithOpKind kind, const Def* ldef, const Def* rdef);
    const Def* relOp(RelOpKind kind, const Def* ldef, const Def* rdef);
    const Def* extract(const Def* tuple, size_t index);
    const Def* insert(const Def* tuple, size_t index, const Def* value);
    const Def* select(const Def* cond, const Def* tdef, const Def* fdef);
    const Def* tuple(const Def* const* begin, const Def* const* end);
    template<size_t N>
    const Def* tuple(const Def* const (&array)[N]) { return createTuple(array, array + N); }
    const Param* param(const Type* type, const Lambda* parent, size_t index);

    void jump(Lambda*& from, const Def* to, const Def* const* begin, const Def* const* end);
    template<size_t N>
    void jump(Lambda*& from, const Def* to, const Def* const (&args)[N]) { 
        return jump(from, to, args, args + N); 
    }
    void jump1(Lambda*& from, const Def* to, const Def* arg1) { 
        const Def* args[1] = { arg1 };
        return jump(from, to, args, args + 1); 
    }
    void jump2(Lambda*& from, const Def* to, const Def* arg1, const Def* arg2) { 
        const Def* args[2] = { arg1 };
        return jump(from, to, args, args + 2); 
    }
    void jump3(Lambda*& from, const Def* to, const Def* arg1, const Def* arg2, const Def* arg3) { 
        const Def* args[3] = { arg1, arg2, arg3 };
        return jump(from, to, args, args + 3); 
    }
    void branch(Lambda*& lambda, const Def* cond, const Def* tto, const Def* fto);

    /*
     * optimizations
     */

    /// Tell the world which Def%s are axiomatically live.
    void setLive(const Def* def);
    /// Tell the world which Lambda%s axiomatically reachable.
    void setReachable(const Lambda* lambda);

    /// Dead code elimination.
    void dce();
    /// Unreachable code elimination.
    void uce();

    /// Performs dead code and unreachable code elimination.
    void cleanup();

    /*
     * getters
     */

    const DefSet& defs() const { return defs_; }

    /*
     * other
     */

    Params findParams(const Lambda* lambda);
    void dump(bool fancy = false);

private:

    const Lambda* finalize(Lambda*& lambda);
    void unmark();
    void destroy(const Def* def);

    const Def* findDef(const Def* def);

    template<class T> 
    const T* find(const T* val) { return (const T*) findDef(val); }

    typedef boost::unordered_set<const Def*> Live;
    typedef boost::unordered_set<const Lambda*> Reachable;

    void dce_insert(const Def* def);
    void uce_insert(Reachable& reachable, const Lambda* lambda);

    const Def* tryFold(IndexKind kind, const Def* ldef, const Def* rdef);

    DefSet defs_;

    Live live_;
    Reachable reachable_;

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
