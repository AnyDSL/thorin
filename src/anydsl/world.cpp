#include "anydsl/world.h"

#include "anydsl/primop.h"
#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/jump.h"
#include "anydsl/fold.h"

namespace anydsl {

/*
 * helpers
 */

const Type* World::findType(const Type* type) {
    TypeMap::iterator i = types_.find(type);
    if (i != types_.end()) {
        delete type;
        return *i;
    }

    types_.insert(type);

    return type;
}

Value* World::findValue(Value* value) {
    ValueMap::iterator i = values_.find(value);
    if (i != values_.end()) {
        delete value;
        return *i;
    }

    values_.insert(value);

    return value;
}


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
    , unit_ (tfind(new Sigma(*this, (const Type* const*) 0, (const Type* const*) 0)))
    , pi0_  (tfind(new Pi(unit_)))
#define ANYDSL_U_TYPE(T) ,T##_(tfind(new PrimType(*this, PrimType_##T)))
#define ANYDSL_F_TYPE(T) ,T##_(tfind(new PrimType(*this, PrimType_##T)))
#include "anydsl/tables/primtypetable.h"
{}

World::~World() {
    cleanup();

    for_all (sigma,  namedSigmas_) delete sigma;
    //for_all (lambda, lambdas_)     delete lambda;

    for_all (v, values_) delete v;
    for_all (t, types_) delete t;
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

/*
 * literals
 */

PrimLit* World::literal(PrimLitKind kind, Box value) {
    return vfind(new PrimLit(type(lit2type(kind)), value));
}

PrimLit* World::literal(const PrimType* p, Box value) {
    return vfind(new PrimLit(p, value));
}

Undef* World::undef(const Type* type) {
    return vfind(new Undef(type));
}

ErrorLit* World::literal_error(const Type* type) {
    return vfind(new ErrorLit(type));
}

/*
 * create
 */

Jump* World::createJump(Def* to, Def* const* arg_begin, Def* const* arg_end) {
    return vfind(new Jump(to, arg_begin, arg_end));
}

Jump* World::createBranch(Def* cond, Def* tto, Def* fto, Def* const* arg_begin, Def* const* arg_end) {
    return createJump(createSelect(cond, tto, fto), arg_begin, arg_end);
}

Jump* World::createBranch(Def* cond, Def* tto, Def* fto) {
    return createBranch(cond, tto, fto, 0, 0);
}

Value* World::createTuple(Def* const* begin, Def* const* end) { 
    return vfind(new Tuple(*this, begin, end));
}

Lambda* World::createLambda(const Pi* pi) {
    Lambda* lambda = new Lambda(pi);
    lambdas_.insert(lambda);
    return lambda;
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

    return vfind(new ArithOp(kind, ldef, rdef));
}

Value* World::createRelOp(RelOpKind kind, Def* ldef, Def* rdef) {
    if (Value* value = tryFold((IndexKind) kind, ldef, rdef))
        return value;

    bool swap;
    kind = normalizeRel(kind, swap);
    if (swap)
        std::swap(ldef, rdef);

    return vfind(new RelOp(kind, ldef, rdef));
}

Value* World::createProj(Def* tuple, PrimLit* i) {
    // TODO folding
    return vfind(new Proj(tuple, i));
}

Value* World::createInsert(Def* tuple, PrimLit* i, Def* value) {
    // TODO folding
    return vfind(new Insert(tuple, i, value));
}


Value* World::createSelect(Def* cond, Def* tdef, Def* fdef) {
    return vfind(new Select(cond, tdef, fdef));
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
