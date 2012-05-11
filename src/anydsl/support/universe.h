#ifndef ANYDSL_SUPPORT_UNIVERSE_H
#define ANYDSL_SUPPORT_UNIVERSE_H

#include <cassert>
#include <string>

#include "anydsl/air/enums.h"

namespace anydsl {

class ArithOp;
class Def;
class PrimType;

class Universe {
public:

    Universe();
    ~Universe();

    ArithOp* createArithOp(ArithOpKind arithOpKind,
                           Def* ldef, Def* rdef, 
                           const std::string& ldebug = "", 
                           const std::string& rdebug = "", 
                           const std::string&  debug = "");

#define ANYDSL_U_TYPE(T) PrimType* get_##T() const { return T##_; }
#define ANYDSL_F_TYPE(T) PrimType* get_##T() const { return T##_; }
#include "anydsl/tables/primtypetable.h"

    PrimType* get(PrimTypeKind kind) const { 
        size_t i = kind - Begin_PrimType;
        assert(0 <= i && i < (size_t) Num_PrimTypes); 
        return primTypes_[i];
    }

    PrimType* get(PrimConstKind kind) const { 
        size_t i = const2type(kind) - Begin_PrimType;
        assert(0 <= i && i < (size_t) Num_PrimTypes); 
        return primTypes_[i];
    }

private:

    bool dummy_;

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

#endif // ANYDSL_SUPPORT_UNIVERSE_H
