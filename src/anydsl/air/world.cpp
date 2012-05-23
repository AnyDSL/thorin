#include "anydsl/air/world.h"

#include "anydsl/air/primop.h"
#include "anydsl/air/literal.h"
#include "anydsl/air/type.h"
#include "anydsl/air/terminator.h"
#include "anydsl/fold.h"

namespace anydsl {

/*
 * helpers
 */

static inline bool isCommutative(ArithOpKind kind) {
    switch (kind) {
        case ArithOp_add:
        case ArithOp_mul:
            return true;
        default:
            return false;
    }
}

static inline RelOpKind normalizeRel(RelOpKind kind, bool& swap) {
    swap = false;
    switch (kind) {
        case RelOp_cmp_ugt: swap = true; return RelOp_cmp_ult;
        case RelOp_cmp_uge: swap = true; return RelOp_cmp_ule;
        case RelOp_cmp_sgt: swap = true; return RelOp_cmp_slt;
        case RelOp_cmp_sge: swap = true; return RelOp_cmp_sle;

        case RelOp_fcmp_ogt: swap = true; return RelOp_fcmp_olt;
        case RelOp_fcmp_oge: swap = true; return RelOp_fcmp_ole;
        case RelOp_fcmp_ugt: swap = true; return RelOp_fcmp_ult;
        case RelOp_fcmp_uge: swap = true; return RelOp_fcmp_ule;
        default: return kind;
    }
}

static void examineDef(Def* def, FoldValue& v) {
    if (def->isa<Undef>())
        v.kind = FoldValue::Undef;
    else if (def->isa<ErrorLit>())
        v.kind = FoldValue::Error;
    if (PrimLit* lit = def->isa<PrimLit>()) {
        v.kind = FoldValue::Valid;
        v.box = lit->box();
    }
   
}
    
/*
 * constructor and destructor
 */

World::World() 
    : values_(1031)
    , types_(1031)
    , unit_(findType<Sigma>(Sigma::VN((const Type**) 0, (const Type**) 0)))
    , pi0_ (findType<Pi>   (Pi   ::VN((const Type**) 0, (const Type**) 0)))
#define ANYDSL_U_TYPE(T) ,T##_(findType<PrimType>(PrimType::VN(PrimType_##T)))
#define ANYDSL_F_TYPE(T) ,T##_(findType<PrimType>(PrimType::VN(PrimType_##T)))
#include "anydsl/tables/primtypetable.h"
{}

World::~World() {
    cleanup();

    FOREACH(sigma,  namedSigmas_) delete sigma;
    //FOREACH(lambda, lambdas_)     delete lambda;

    std::cout << types_.size() << std::endl;
    std::cout << values_.size() << std::endl;
    FOREACH(p, values_) delete p.second;
    FOREACH(t, types_) delete t.second;
}

/*
 * types
 */

Sigma* World::namedSigma(const std::string& name /*= ""*/) {
    Sigma* s = new Sigma(*this);
    s->debug = name;
    namedSigmas_.push_back(s);

    return s;
}

const Sigma* World::sigma1(const Type* t1) {
    return sigma(&t1, (&t1) + 1);
}

const Sigma* World::sigma2(const Type* t1, const Type* t2) {
    const Type* types[2] = {t1, t2};
    return sigma(types);
}

const Sigma* World::sigma3(const Type* t1, const Type* t2, const Type* t3) {
    const Type* types[3] = {t1, t2, t3};
    return sigma(types);
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

template<class T>
T* World::findValue(const ValueNumber& vn) {
    ValueMap::iterator i = values_.find(vn);
    if (i != values_.end())
        return scast<T>(i->second);

    T* value = new T(vn);
    values_[vn] = value;

    return value;
}

/*
 * literals
 */

PrimLit* World::literal(PrimLitKind kind, Box value) {
    return findValue<PrimLit>(PrimLit::VN(type(lit2type(kind)), value));
}

PrimLit* World::literal(const PrimType* p, Box value) {
    return findValue<PrimLit>(PrimLit::VN(p, value));
}

Undef* World::undef(const Type* type) {
    return findValue<Undef>(Undef::VN(type));
}

ErrorLit* World::literal_error(const Type* type) {
    return findValue<ErrorLit>(ErrorLit::VN(type));
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

Value* World::tryFold(IndexKind kind, Def* ldef, Def* rdef) {
    FoldValue a(ldef->type()->as<PrimType>()->kind());
    FoldValue b(a.type);

    examineDef(ldef, a);
    examineDef(rdef, b);

    if (ldef->isa<Literal>() && rdef->isa<Literal>()) {
        const PrimType* p = ldef->type()->as<PrimType>();
        FoldValue res = fold_bin(kind, p->kind(), a, b);

        switch (res.kind) {
            case FoldValue::Valid: return literal(res.type, res.box);
            case FoldValue::Undef: return undef(res.type);
            case FoldValue::Error: return literal_error(res.type);
        }
    }

    return 0;
}

Value* World::createArithOp(ArithOpKind kind, Def* ldef, Def* rdef) {
    if (Value* value = tryFold((IndexKind) kind, ldef, rdef))
        return value;

    if (isCommutative(kind))
        if (ldef > rdef)
            std::swap(ldef, rdef);

    return findValue<ArithOp>(ArithOp::VN(kind, ldef, rdef));
}

Value* World::createRelOp(RelOpKind kind, Def* ldef, Def* rdef) {
    if (Value* value = tryFold((IndexKind) kind, ldef, rdef))
        return value;

    bool swap;
    kind = normalizeRel(kind, swap);
    if (swap)
        std::swap(ldef, rdef);

    return findValue<RelOp>(RelOp::VN(kind, ldef, rdef));
}

Value* World::createProj(Def* tuple, PrimLit* i) {
    // TODO folding
    return findValue<Proj>(Proj::VN(tuple, i));
}

Terminator* World::createBranch(Lambda* parent, Def* cond, Lambda* tto, Lambda* fto) {
    Terminator* result; 

    if (PrimLit* lit = cond->isa<PrimLit>()) {
        if (lit->box().get_u1() == true) 
            result = new Goto(parent, tto);
        else
            result = new Goto(parent, fto);
    } else
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
