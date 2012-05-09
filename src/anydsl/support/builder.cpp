#include "anydsl/support/builder.h"

namespace anydsl {


ArithOp* Builder::createArithOp(ArithOpKind arithOpKind,
                                Def* ldef, Def* rdef, 
                                const std::string& ldebug /*= ""*/, 
                                const std::string& rdebug /*= ""*/, 
                                const std::string&  debug /*= ""*/) {
    // TODO

    return new ArithOp(arithOpKind, ldef, rdef, ldebug, rdebug, debug);
}



} // namespace anydsl
