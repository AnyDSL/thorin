#include "anydsl/support/world.h"

#include "anydsl/air/binop.h"
#include "anydsl/air/constant.h"
#include "anydsl/air/type.h"
#include "anydsl/util/foreach.h"

namespace anydsl {

World::World() 
    : emptyPi_(new Pi(*this, "pi()"))
    , unit_(new Sigma(*this, "sigma()"))
#define ANYDSL_U_TYPE(T) ,T##_(new PrimType(*this, PrimType_##T))
#define ANYDSL_F_TYPE(T) ,T##_(new PrimType(*this, PrimType_##T))
#include "anydsl/tables/primtypetable.h"
{}

World::~World() {
    for (size_t i = 0; i < Num_PrimTypes; ++i)
        delete primTypes_[i];

    FOREACH(sigma, namedSigmas_)
        delete sigma;
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

Sigma* World::getNamedSigma(const std::string& name /*= ""*/) {
    Sigma* sigma = new Sigma(*this, name);
    namedSigmas_.push_back(sigma);
    return sigma;
}

} // namespace anydsl
