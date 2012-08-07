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
class Bottom;
class Lambda;
class Pi;
class PrimLit;
class PrimType;
class Sigma;
class Type;
class Def;
class Any;

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

#define ANYDSL_UF_TYPE(T) const PrimType* type_##T() const { return T##_; }
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
    const Sigma* sigma(ArrayRef<const Type*> elems) { return sigma(elems); }

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
    const Pi* pi(ArrayRef<const Type*> elems) { return find(new Pi(*this, elems)); }

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
        anydsl_assert(isFloat(kind), "must not be a float"); 
        return literal(kind, -1); 
    }

    const Any* any(const Type* type);
    const Any* any(PrimTypeKind kind) { return any(type(kind)); }
    const Bottom* bottom(const Type* type);
    const Bottom* bottom(PrimTypeKind kind) { return bottom(type(kind)); }

    /*
     * create
     */

    const Def* arithop(ArithOpKind kind, const Def* ldef, const Def* rdef);
#define ANYDSL_ARITHOP(OP) const Def* arithop_##OP(const Def* ldef, const Def* rdef) { return arithop(ArithOp_##OP, ldef, rdef); }
#include "anydsl/tables/arithoptable.h"

    const Def* relop(RelOpKind kind, const Def* ldef, const Def* rdef);
#define ANYDSL_RELOP(OP) const Def* relop_##OP(const Def* ldef, const Def* rdef) { return relop(RelOp_##OP, ldef, rdef); }
#include "anydsl/tables/reloptable.h"

    const Def* convop(ConvOpKind kind, const Def* from, const Type* to);
#define ANYDSL_CONVOP(OP) const Def* convop_##OP(const Def* from, const Type* to) { return convop(ConvOp_##OP, from, to); }
#include "anydsl/tables/convoptable.h"

    const Def* extract(const Def* tuple, u32 i);
    const Def* insert(const Def* tuple, u32 i, const Def* value);
    const Def* select(const Def* cond, const Def* tdef, const Def* fdef);
    const Def* tuple(ArrayRef<const Def*> args);
    const Param* param(const Type* type, const Lambda* parent, u32 i);

    void jump(Lambda*& from, const Def* to, ArrayRef<const Def*> args);
    void jump1(Lambda*& from, const Def* to, const Def* arg1) {
        const Def* args[1] = { arg1 };
        return jump(from, to, args);
    }
    void jump2(Lambda*& from, const Def* to, const Def* arg1, const Def* arg2) {
        const Def* args[2] = { arg1, arg2 };
        return jump(from, to, args);
    }
    void jump3(Lambda*& from, const Def* to, const Def* arg1, const Def* arg2, const Def* arg3) {
        const Def* args[3] = { arg1, arg2, arg3 };
        return jump(from, to, args);
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
    const Def* rehash(const Def* def);
    Def* release(const Def* def);

    template<class T>
    const T* find(const T* val) { return (const T*) findDef(val); }

    /*
     * debug printing
     */

    void printPostOrder();
    void printReversePostOrder();
    void printDominators();

private:

    const Lambda* finalize(Lambda*& lambda);
    void unmark();
    void destroy(const Def* def);

    const Def* findDef(const Def* def);

    typedef boost::unordered_set<const Def*> Live;
    typedef boost::unordered_set<const Lambda*> Reachable;

    void dce_insert(const Def* def);
    void uce_insert(Reachable& reachable, const Lambda* lambda);

    DefSet defs_;

    Live live_;
    Reachable reachable_;

    const Sigma* unit_; ///< sigma().
    const Pi* pi0_;     ///< pi().

    union {
        struct {
#define ANYDSL_UF_TYPE(T) const PrimType* T##_;
#include "anydsl/tables/primtypetable.h"
        };

        const PrimType* primTypes_[Num_PrimTypes];
    };

    friend void Def::replace(const Def*) const;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif
