#ifndef ANYDSL_AIR_SIGMA_OP_H
#define ANYDSL_AIR_SIGMA_OP_H

#include "anydsl/air/def.h"
#include "anydsl/air/use.h"

namespace anydsl {

class PrimLit;

//------------------------------------------------------------------------------

class SigmaOp : public PrimOp {
protected:

    SigmaOp(IndexKind index, const Type* type,
            Def* tuple, PrimLit* elem, 
            const std::string& tupleDebug, const std::string& debug);

public:

    const PrimLit* elem() const { return elem_; }
    virtual uint64_t hash() const;

    Use tuple;

private:

    PrimLit* elem_;
};

//------------------------------------------------------------------------------

class Extract : public SigmaOp {
private:

    Extract(Def* tuple, PrimLit* elem, 
            const std::string& tupleDebug,
            const std::string& debug);
};

//------------------------------------------------------------------------------

class Insert : public SigmaOp {
private:

    Insert(Def* tuple, PrimLit* elem, Def* value, 
           const std::string& tupleDebug, const std::string& valueDebug,
           const std::string& debug)
        : SigmaOp(Index_Insert, tuple->type(), tuple, elem, tupleDebug, debug)
        , value(value, this, valueDebug)
    {}

    virtual uint64_t hash() const;

public:

    Use value;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_SIGMA_OP_H
