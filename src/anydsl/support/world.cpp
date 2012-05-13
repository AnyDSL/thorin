#include "anydsl/support/world.h"

#include "anydsl/air/binop.h"
#include "anydsl/air/constant.h"
#include "anydsl/air/type.h"

namespace anydsl {

World::World() 
    : values_()
#define ANYDSL_U_TYPE(T) ,T##_(new PrimType(*this, PrimType_##T, #T))
#define ANYDSL_F_TYPE(T) ,T##_(new PrimType(*this, PrimType_##T, #T))
#include "anydsl/tables/primtypetable.h"
{
}

World::~World() {
    for (size_t i = 0; i < Num_PrimTypes; ++i)
        delete primTypes_[i];
}

PrimConst* World::constant(PrimTypeKind kind, Box value) {
    return new PrimConst(kind, value, "todo");
}

ArithOp* World::createArithOp(ArithOpKind arithOpKind,
                                Def* ldef, Def* rdef, 
                                const std::string& ldebug /*= ""*/, 
                                const std::string& rdebug /*= ""*/, 
                                const std::string&  debug /*= ""*/) {
    // TODO

    return new ArithOp(arithOpKind, ldef, rdef, ldebug, rdebug, debug);
}

} // namespace anydsl
