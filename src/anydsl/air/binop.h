#ifndef ANYDSL_BINOP_H
#define ANYDSL_BINOP_H

#include <boost/array.hpp>

#include "anydsl/air/enums.h"
#include "anydsl/air/def.h"
#include "anydsl/air/use.h"

namespace anydsl {

//------------------------------------------------------------------------------

class BinOp : public PrimOp {
protected:

    BinOp(IndexKind index, const Type* type,
            Def* ldef, Def* rdef, 
            const std::string& ldebug, const std::string& rdebug,
            const std::string& debug)
        : PrimOp(index, type, debug)
        , luse(ldef, this, ldebug)
        , ruse(rdef, this, rdebug)
    {}

public:

    typedef boost::array<Use*, 2> LRUse;
    typedef boost::array<const Use*, 2> ConstLRUse;

    ArithOpKind arithOpKind() { return (ArithOpKind) index(); }
    virtual uint64_t hash() const;

    LRUse lruse() { return (LRUse){{ &luse, &ruse }}; }
    ConstLRUse lruse() const { return (ConstLRUse){{ &ruse, &ruse }}; }

public:

    Use luse;
    Use ruse;
};

//------------------------------------------------------------------------------

class ArithOp : public BinOp {
private:

    ArithOp(ArithOpKind arithOpKind, 
            Def* ldef, Def* rdef, 
            const std::string& ldebug, const std::string& rdebug,
            const std::string& debug)
        : BinOp((IndexKind) arithOpKind, ldef->type(), ldef, rdef, ldebug, rdebug, debug)
    {
        anydsl_assert(ldef->type() == rdef->type(), "type are not equal");
    }

    friend class World;
};

//------------------------------------------------------------------------------

class RelOp : public BinOp {
private:

    RelOp(ArithOpKind arithOpKind, 
            Def* ldef, Def* rdef, 
            const std::string& ldebug, const std::string& rdebug,
            const std::string& debug);

    friend class World;
};

//------------------------------------------------------------------------------


} // namespace anydsl

#endif // ANYDSL_PRIMOP_H
