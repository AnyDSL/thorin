#include "anydsl/support/world.h"

#include "anydsl/air/binop.h"
#include "anydsl/air/constant.h"
#include "anydsl/air/type.h"
#include "anydsl/util/foreach.h"

namespace anydsl {

World::World() 
    : emptyPi_(new Pi(*this, "pi()"))
#define ANYDSL_U_TYPE(T) ,T##_(new PrimType(*this, PrimType_##T, #T))
#define ANYDSL_F_TYPE(T) ,T##_(new PrimType(*this, PrimType_##T, #T))
#include "anydsl/tables/primtypetable.h"
{}

World::~World() {
    for (size_t i = 0; i < Num_PrimTypes; ++i)
        delete primTypes_[i];
}

PrimConst* World::constant(PrimTypeKind kind, Box value) {
    //Values::iterator i = values_.find(
    return new PrimConst(kind, value, "todo");
}

ArithOp* World::createArithOp(ArithOpKind arithOpKind,
                                Def* ldef, Def* rdef, 
                                const std::string& ldebug /*= ""*/, 
                                const std::string& rdebug /*= ""*/, 
                                const std::string&  debug /*= ""*/) {
    //ValRange range = values_.equal_range(0);
    FOREACH(p, values_.equal_range(0))
    {
        std::cout << p.second << std::endl;
    }

    ArithOp* op = new ArithOp(arithOpKind, ldef, rdef, ldebug, rdebug, debug);
    values_.insert(std::make_pair(op->hash(), op));
    return op;
}

} // namespace anydsl
