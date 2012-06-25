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

static void examineDef(const Def* def, FoldValue& v) {
    if (def->isa<Undef>())
        v.kind = FoldValue::Undef;
    else if (def->isa<ErrorLit>())
        v.kind = FoldValue::Error;
    if (const PrimLit* lit = def->isa<PrimLit>()) {
        v.kind = FoldValue::Valid;
        v.box = lit->box();
    }
   
}

/*
 * constructor and destructor
 */

World::World() 
    : values_(1031)
    , unit_ (find(new Sigma(*this, (const Type* const*) 0, (const Type* const*) 0)))
    , pi0_  (find(new Pi(*this, (const Type* const*) 0, (const Type* const*) 0)))
#define ANYDSL_U_TYPE(T) ,T##_(find(new PrimType(*this, PrimType_##T)))
#define ANYDSL_F_TYPE(T) ,T##_(find(new PrimType(*this, PrimType_##T)))
#include "anydsl/tables/primtypetable.h"
{}

World::~World() {
    for_all (sigma,  namedSigmas_) delete sigma;
    for_all (lambda, lambdas_)     delete lambda;

    cleanup();
    //anydsl_assert(values_.empty(), "cleanup should catch everything");
}

/*
 * types
 */

Sigma* World::namedSigma(size_t num, const std::string& name /*= ""*/) {
    Sigma* s = new Sigma(*this, num);
    s->debug = name;
    namedSigmas_.push_back(s);

    return s;
}

/*
 * literals
 */

const PrimLit* World::literal(PrimLitKind kind, Box value) {
    return find(new PrimLit(type(lit2type(kind)), value));
}

const PrimLit* World::literal(const PrimType* p, Box value) {
    return find(new PrimLit(p, value));
}

const Undef* World::undef(const Type* type) {
    return find(new Undef(type));
}

const ErrorLit* World::literal_error(const Type* type) {
    return find(new ErrorLit(type));
}

/*
 * create
 */

const Jump* World::createJump(const Def* to, const Def* const* arg_begin, const Def* const* arg_end) {
    return find(new Jump(to, arg_begin, arg_end));
}

const Jump* World::createBranch(const Def* cond, const Def* tto, const Def* fto, 
                                const Def* const* arg_begin, const Def* const* arg_end) {
    return createJump(createSelect(cond, tto, fto), arg_begin, arg_end);
}

const Jump* World::createBranch(const Def* cond, const Def* tto, const Def* fto) {
    return createBranch(cond, tto, fto, 0, 0);
}

const Value* World::createTuple(const Def* const* begin, const Def* const* end) { 
    return find(new Tuple(*this, begin, end));
}

Lambda* World::createLambda(const Pi* pi) {
    Lambda* lambda = new Lambda(pi);
    lambdas_.insert(lambda);
    return lambda;
}

const Value* World::tryFold(IndexKind kind, const Def* ldef, const Def* rdef) {
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

const Value* World::createArithOp(ArithOpKind kind, const Def* ldef, const Def* rdef) {
    if (const Value* value = tryFold((IndexKind) kind, ldef, rdef))
        return value;

    if (isCommutative(kind))
        if (ldef > rdef)
            std::swap(ldef, rdef);

    return find(new ArithOp(kind, ldef, rdef));
}

const Value* World::createRelOp(RelOpKind kind, const Def* ldef, const Def* rdef) {
    if (const Value* value = tryFold((IndexKind) kind, ldef, rdef))
        return value;

    bool swap;
    kind = normalizeRel(kind, swap);
    if (swap)
        std::swap(ldef, rdef);

    return find(new RelOp(kind, ldef, rdef));
}

const Value* World::createProj(const Def* tuple, const PrimLit* i) {
    // TODO folding
    return find(new Proj(tuple, i));
}

const Value* World::createInsert(const Def* tuple, const PrimLit* i, const Def* value) {
    // TODO folding
    return find(new Insert(tuple, i, value));
}


const Value* World::createSelect(const Def* cond, const Def* tdef, const Def* fdef) {
    return find(new Select(cond, tdef, fdef));
}

void World::cleanup() {
    // TODO
}

const Value* World::findValue(const Value* value) {
    ValueMap::iterator i = values_.find(value);
    if (i != values_.end()) {
        delete value;
        return *i;
    }

    values_.insert(value);

    return value;
}


} // namespace anydsl
