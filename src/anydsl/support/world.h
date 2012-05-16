#ifndef ANYDSL_SUPPORT_WORLD_H
#define ANYDSL_SUPPORT_WORLD_H

#include <cassert>
#include <string>

#include <boost/unordered_map.hpp>

#include "anydsl/air/enums.h"
#include "anydsl/util/box.h"
#include "anydsl/util/autoptr.h"

namespace anydsl {

class ArithOp;
class Def;
class Pi;
class PrimConst;
class PrimType;
class Sigma;
class Value;

//------------------------------------------------------------------------------

typedef boost::unordered_multimap<uint64_t, Value*> Values;
typedef boost::unordered_multimap<uint64_t, Pi*> Pis;
typedef boost::unordered_multimap<uint64_t, Sigma*> Sigmas;
typedef std::vector<Sigma*> NamedSigmas;

//------------------------------------------------------------------------------

class World {
public:

    World();
    ~World();

    ArithOp* createArithOp(ArithOpKind arithOpKind,
                           Def* ldef, Def* rdef, 
                           const std::string& ldebug = "", 
                           const std::string& rdebug = "", 
                           const std::string&  debug = "");

#define ANYDSL_U_TYPE(T) PrimType* type_##T() const { return T##_; }
#define ANYDSL_F_TYPE(T) PrimType* type_##T() const { return T##_; }
#include "anydsl/tables/primtypetable.h"

    PrimType* type(PrimTypeKind kind) const { 
        size_t i = kind - Begin_PrimType;
        assert(0 <= i && i < (size_t) Num_PrimTypes); 
        return primTypes_[i];
    }

    template<class T>
    PrimConst* constant(T value) { 
        return constant(Type2PrimTypeKind<T>::kind, Box(value));
    }

    template<class T>
    const Sigma* sigma(T begin, T end);

    /// Creates a fresh named sigma
    Sigma* getNamedSigma(const std::string& name = "");

    const Pi* emptyPi() const { return emptyPi_; }
    const Sigma* unit() const { return unit_; }

    PrimConst* constant(PrimTypeKind kind, Box value);

private:

    Values values_;
    Pis pis_;
    Sigmas sigmas_;
    NamedSigmas namedSigmas_;

    AutoPtr<Pi> emptyPi_; ///< pi().
    AutoPtr<Sigma> unit_;

    union {
        struct {
#define ANYDSL_U_TYPE(T) PrimType* T##_;
#define ANYDSL_F_TYPE(T) PrimType* T##_;
#include "anydsl/tables/primtypetable.h"
        };

        PrimType* primTypes_[Num_PrimTypes];
    };


};

} // namespace anydsl

#endif // ANYDSL_SUPPORT_WORLD_H
