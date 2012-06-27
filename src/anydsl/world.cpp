#include "anydsl/world.h"

#include "anydsl/def.h"
#include "anydsl/primop.h"
#include "anydsl/lambda.h"
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
    : defs_(1031)
    , unit_ (find(new Sigma(*this, (const Type* const*) 0, (const Type* const*) 0)))
    , pi0_  (find(new Pi   (*this, (const Type* const*) 0, (const Type* const*) 0)))
    , noret_(find(new NoRet(*this)))
#define ANYDSL_U_TYPE(T) ,T##_(find(new PrimType(*this, PrimType_##T)))
#define ANYDSL_F_TYPE(T) ,T##_(find(new PrimType(*this, PrimType_##T)))
#include "anydsl/tables/primtypetable.h"
{
    live_.insert(unit_);
    live_.insert(pi0_);
    live_.insert(noret_);
    for (size_t i = 0; i < Num_PrimTypes; ++i)
        live_.insert(primTypes_[i]);
}

World::~World() {
    for_all (sigma,  namedSigmas_) delete sigma;

    live_.clear();
    cleanup();

    anydsl_assert(defs_.empty(), "cleanup should catch everything");
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

const Jump* World::createGoto(const Def* to, const Def* const* begin, const Def* const* end) {
    return find(new Goto(*this, to, begin, end));
}

const Jump* World::createBranch(const Def* cond, 
                                const Def* tto, const Def* const* tbegin, const Def* const* tend,
                                const Def* fto, const Def* const* fbegin, const Def* const* fend) {
    return find(new Branch(*this, cond, tto, tbegin, tend, fto, fbegin, fend));
}

const Jump* World::createBranch(const Def* cond, const Def* tto, const Def* fto) {
    return createBranch(cond, tto, (const Def* const*) 0, (const Def* const*) 0, 
                              fto, (const Def* const*) 0, (const Def* const*) 0);
}

const Def* World::createTuple(const Def* const* begin, const Def* const* end) { 
    return find(new Tuple(*this, begin, end));
}

const Def* World::tryFold(IndexKind kind, const Def* ldef, const Def* rdef) {
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

const Def* World::createArithOp(ArithOpKind kind, const Def* ldef, const Def* rdef) {
    if (const Def* value = tryFold((IndexKind) kind, ldef, rdef))
        return value;

    if (isCommutative(kind))
        if (ldef > rdef)
            std::swap(ldef, rdef);

    return find(new ArithOp(kind, ldef, rdef));
}

const Def* World::createRelOp(RelOpKind kind, const Def* ldef, const Def* rdef) {
    if (const Def* value = tryFold((IndexKind) kind, ldef, rdef))
        return value;

    bool swap;
    kind = normalizeRel(kind, swap);
    if (swap)
        std::swap(ldef, rdef);

    return find(new RelOp(kind, ldef, rdef));
}

const Def* World::createExtract(const Def* tuple, const PrimLit* i) {
    // TODO folding
    return find(new Extract(tuple, i));
}

const Def* World::createInsert(const Def* tuple, const PrimLit* i, const Def* value) {
    // TODO folding
    return find(new Insert(tuple, i, value));
}


const Def* World::createSelect(const Def* cond, const Def* tdef, const Def* fdef) {
    return find(new Select(cond, tdef, fdef));
}

const Lambda* World::finalize(const Lambda* lambda) {
    anydsl_assert(lambda->type(), "must be set");
    anydsl_assert(lambda->jump(), "must be set");

    const Lambda* l = find<Lambda>(lambda);

    for_all (param, l->params())
        findDef(param.def());

    return find<Lambda>(lambda);
}

void World::insert(const Def* def) {
    if (def->flag_)
        return;

    def->flag_ = true;

    for_all (def, def->ops())
        insert(def);

    if (const Type* type = def->type())
        insert(type);
}

void World::cleanup() {
    for_all (def, defs_)
        def->flag_ = false;

    for_all (def, live_)
        insert(def);

    DefMap::iterator i = defs_.begin();
    while (i != defs_.end()) {
        const Def* def = *i;
        if (!def->flag_) {
            delete def;
            i = defs_.erase(i);
        } else
            ++i;
    }
}

const Def* World::findDef(const Def* def) {
    DefMap::iterator i = defs_.find(def);
    if (i != defs_.end()) {
        delete def;
        return *i;
    }

    defs_.insert(def);

    return def;
}

} // namespace anydsl
