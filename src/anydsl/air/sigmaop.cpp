#include "anydsl/air/sigmaop.h"

#include "anydsl/air/type.h"
#include "anydsl/support/hash.h"

namespace anydsl {

//------------------------------------------------------------------------------

SigmaOp::SigmaOp(IndexKind index, Type* type,
                 Def* tuple, PrimConst* elem, 
                 const std::string& tupleDebug,
                 const std::string debug)
    : PrimOp(index, type, debug)
    , tuple_(tuple, this, tupleDebug)
    , elem_(elem)
{
    anydsl_assert(tuple->getAs<Sigma>(), "must be of Sigma type");
}

uint64_t SigmaOp::hash() const {
    return hashBinOp(index(), tuple_.def(), elem_);
}

//------------------------------------------------------------------------------

Extract::Extract(Def* tuple, PrimConst* elem, 
        const std::string& tupleDebug,
        const std::string debug)
    : SigmaOp(Index_Extract, 
              scast<Sigma>(tuple->type())->get(elem),
              tuple, elem, tupleDebug, debug)
{}

//------------------------------------------------------------------------------

uint64_t Insert::hash() const {
    return SigmaOp::hash() * 31 ^ ((uintptr_t) value_.def());
}

//------------------------------------------------------------------------------

} // namespace anydsl
