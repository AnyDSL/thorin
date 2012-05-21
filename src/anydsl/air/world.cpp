#include "anydsl/air/world.h"

#include "anydsl/air/binop.h"
#include "anydsl/air/literal.h"
#include "anydsl/air/type.h"
#include "anydsl/air/terminator.h"

namespace anydsl {

/*
 * constructor and destructor
 */

World::World() 
    : type_error_(new ErrorType(*this))
#define ANYDSL_U_TYPE(T) ,T##_(new PrimType(*this, PrimType_##T))
#define ANYDSL_F_TYPE(T) ,T##_(new PrimType(*this, PrimType_##T))
#include "anydsl/tables/primtypetable.h"
{
    {
        Sigma* s = new Sigma(*this);
        unit_ = s;
        uint64_t h = Sigma::hash((const Type**) 0, (const Type**) 0);
        sigmas_.insert(std::make_pair(h, s));
    }
    {
        Pi* p = new Pi(*this);
        pi0_ = p;
        uint64_t h = Pi::hash((const Type**) 0, (const Type**) 0);
        pis_.insert(std::make_pair(h, p));
    }
}

World::~World() {
    cleanup();

    for (size_t i = 0; i < Num_PrimTypes; ++i)
        delete primTypes_[i];

    FOREACH(sigma,  namedSigmas_) delete sigma;
    //FOREACH(lambda, lambdas_)     delete lambda;

    FOREACH(p, values_) delete p.second;
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

const Sigma* World::sigma0(bool named /*= false*/) {
    if (named) {
        Sigma* s = new Sigma(*this, true);
        namedSigmas_.push_back(s);
        return s;
    }

    return unit_;
}

const Sigma* World::sigma1(const Type* t1, bool named /*= false*/) {
    return sigma(&t1, (&t1) + 1, named);
}

const Sigma* World::sigma2(const Type* t1, const Type* t2, bool named /*= false*/) {
    const Type* types[2] = {t1, t2};
    return sigma(types, named);
}

const Sigma* World::sigma3(const Type* t1, const Type* t2, const Type* t3, bool named /*= false*/) {
    const Type* types[3] = {t1, t2, t3};
    return sigma(types, named);
}

const Pi* World::pi1(const Type* t1) {
    return pi(&t1, (&t1) + 1);
}

const Pi* World::pi2(const Type* t1, const Type* t2) {
    const Type* types[2] = {t1, t2};
    return pi(types);
}

const Pi* World::pi3(const Type* t1, const Type* t2, const Type* t3) {
    const Type* types[3] = {t1, t2, t3};
    return pi(types);
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

ErrorLit* World::literal_error(const Type* type) {
    ErrorLit* e = new ErrorLit(type);
    return e;
}

/*
 * create
 */

Lambda* World::createLambda(const Pi* type) {
    assert(type == 0 || type->isa<Pi>());
    Lambda* lambda = type ? new Lambda(type) : new Lambda(*this);
    lambdas_.insert(lambda);
    return lambda;
}

Goto* World::createGoto(Lambda* parent, Lambda* to) {
    Goto* res = new Goto(parent, to);
    parent->setTerminator(res);
    return res;
}

Invoke* World::createInvoke(Lambda* parent, Def* fct) {
    Invoke* res = new Invoke(parent, fct);
    parent->setTerminator(res);
    return res;
}

Value* World::createArithOp(ArithOpKind kind, Def* ldef, Def* rdef) {
    ArithOp* op = new ArithOp(kind, ldef, rdef);
    values_.insert(std::make_pair(op->hash(), op));
    return op;
}

Value* World::createRelOp(RelOpKind kind, Def* ldef, Def* rdef) {
    RelOp* op = new RelOp(kind, ldef, rdef);
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

Lambda* getvalue(Lambda* l) { return l; }
Value* getvalue(std::pair<bool, Value*> p) { return p.second; }

// TODO this is inefficient for Value -- there may be many iterations needed till the outer loop terminates
template<class T, class C>
void World::kill(C& container) {
    size_t oldSize;
    do {
        oldSize = container.size();

        for (typename C::iterator i = container.begin(), e = container.end(); i != e; ++i) {
            T* l = getvalue(*i);
            if (l->uses().empty()) {
                delete l;
                i = container.erase(i);
                if (i == e) 
                    break;
            }
        }
    } while (oldSize != container.size());
}

void World::cleanup() {
    //kill<Lambda>(lambdas_);
    //kill<Value>(values_);
}

} // namespace anydsl
