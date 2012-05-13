#ifndef ANYDSL_SUPPORT_WORLD_H
#define ANYDSL_SUPPORT_WORLD_H

#include <cassert>
#include <string>

#include <boost/unordered_map.hpp>

#include "anydsl/air/enums.h"
#include "anydsl/util/box.h"

namespace anydsl {

class ArithOp;
class Def;
class PrimConst;
class PrimType;
class Sigma;
class Value;

//------------------------------------------------------------------------------

typedef boost::unordered_multimap<uint64_t, Value*> Values;
typedef Values::iterator ValIter;
typedef Values::const_iterator ValConstIter;

/// Make std::pair<ValIter, ValIter> work with the FOREACH macro
struct ValRange : public std::pair<ValIter, ValIter> {
    ValRange(const std::pair<ValIter, ValIter>& p)
        : std::pair<ValIter, ValIter>(p)
    {}

    ValIter begin() { return first; }
    ValIter   end() { return second; }

    typedef ValIter iterator;
    typedef ValConstIter const_iterator;
};

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

    Sigma* getSigma();

    PrimConst* constant(PrimTypeKind kind, Box value);

private:

    Values values_;

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
