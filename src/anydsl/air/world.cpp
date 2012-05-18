#include "anydsl/air/world.h"

#include "anydsl/air/binop.h"
#include "anydsl/air/literal.h"
#include "anydsl/air/type.h"
#include "anydsl/air/terminator.h"
#include "anydsl/util/foreach.h"

namespace anydsl {

/*
 * constructor and destructor
 */

World::World() 
    : emptyPi_(new Pi(*this))
    , unit_(new Sigma(*this, /*named*/ false))
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
    FOREACH(p, values_)   delete p.second;
    FOREACH(p, pis_)    delete p.second;
    FOREACH(p, sigmas_) delete p.second;
}

/*
 * types
 */

Sigma* World::sigma(const std::string& name /*= ""*/) {
    Sigma* sigma = new Sigma(*this, true);
    sigma->debug = name;
    namedSigmas_.push_back(sigma);

    return sigma;
}

/*
 * literals
 */

PrimLit* World::literal(PrimLitKind kind, Box value) {
    //Values::iterator i = values_.find(
    std::ostringstream oss;
    oss << value.u64_;
    PrimLit* prim = new PrimLit(*this, kind, value);
    values_.insert(std::make_pair(prim->hash(), prim));
    return prim;
}

Undef* World::undef(const Type* type) {
    Undef* u = new Undef(type);
    return u;
}

/*
 * create
 */

Lambda* World::createLambda(Lambda* parent) {
    return new Lambda(*this, parent);
}

Goto* World::createGoto(Lambda* parent, Lambda* to) {
    return new Goto(parent, to);
}

const Value* World::createArithOp(ArithOpKind arithOpKind, Def* ldef, Def* rdef) {
    ////ValRange range = values_.equal_range(0);
    //FOREACH(p, values_.equal_range(0))
    //{
        //std::cout << p.second << std::endl;
    //}

    ArithOp* op = new ArithOp(arithOpKind, ldef, rdef);
    values_.insert(std::make_pair(op->hash(), op));
    return op;
}

Terminator* World::createBranch(Lambda* parent, Def* cond, Lambda* tto, Lambda* fto) {
    Terminator* result; 

    if (PrimLit* lit = cond->isa<PrimLit>()) {
        if (lit->box().get_u1() == true) 
            result = new Goto(parent, tto);
        else
            result = new Goto(parent, fto);
    }
    else
        result = new Branch(parent, cond, tto, fto);

    parent->setTerminator(result);

    return result;
}

/*
 * optimize
 */

void World::cleanup() {
    size_t oldSize;

    // repeaut until defs do not change anymore
    do {
        oldSize = values_.size();

        for (DefIter i = values_.begin(), e = values_.end(); i != e; ++i) {
            Def* def = i->second;
            if (def->uses().empty()) {
                delete def;
                i = values_.erase(i);
            }
        }
    } while (oldSize != values_.size());
}

} // namespace anydsl
