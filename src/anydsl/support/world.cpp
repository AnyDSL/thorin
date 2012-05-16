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

    // clean up hash multi maps
    FOREACH(p, defs_)   delete p.second;
    FOREACH(p, pis_)    delete p.second;
    FOREACH(p, sigmas_) delete p.second;
}

PrimConst* World::constant(PrimTypeKind kind, Box value) {
    //Values::iterator i = values_.find(
    PrimConst* prim = new PrimConst(*this, kind, value, "todo");
    defs_.insert(std::make_pair(prim->hash(), prim));
    return prim;
}

ArithOp* World::createArithOp(ArithOpKind arithOpKind,
                                Def* ldef, Def* rdef, 
                                const std::string& ldebug /*= ""*/, 
                                const std::string& rdebug /*= ""*/, 
                                const std::string&  debug /*= ""*/) {
    ////ValRange range = values_.equal_range(0);
    //FOREACH(p, values_.equal_range(0))
    //{
        //std::cout << p.second << std::endl;
    //}

    ArithOp* op = new ArithOp(arithOpKind, ldef, rdef, ldebug, rdebug, debug);
    defs_.insert(std::make_pair(op->hash(), op));
    return op;
}

Sigma* World::getNamedSigma(const std::string& name /*= ""*/) {
    Sigma* sigma = new Sigma(*this, name);
    namedSigmas_.push_back(sigma);
    return sigma;
}

void World::cleanup() {
    size_t oldSize;

    // repeaut until defs do not change anymore
    do {
        oldSize = defs_.size();

        for (DefIter i = defs_.begin(), e = defs_.end(); i != e; ++i) {
            Def* def = i->second;
            if (def->uses().empty()) {
                delete def;
                i = defs_.erase(i);
            }
        }
    } while (oldSize != defs_.size());
}

} // namespace anydsl
