#include "anydsl/world.h"

#include <cmath>
#include <algorithm>
#include <queue>

#include "anydsl/def.h"
#include "anydsl/primop.h"
#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/util/array.h"
#include "anydsl/util/for_all.h"

// debug includes
#include "anydsl/order.h"
#include "anydsl/dom.h"
#include "anydsl/printer.h"

#define ANYDSL_NO_U_TYPE \
    case PrimType_u1: \
    case PrimType_u8: \
    case PrimType_u16: \
    case PrimType_u32: \
    case PrimType_u64: ANYDSL_UNREACHABLE;

#define ANYDSL_NO_F_TYPE \
    case PrimType_f32: \
    case PrimType_f64: ANYDSL_UNREACHABLE;

#define for_all_lambdas(l) \
    for_all (__def, defs_) \
        if (const Lambda* l =__def->isa<Lambda>())

namespace anydsl {

/*
 * constructor and destructor
 */

World::World()
    : defs_(1031)
    , sigma0_ (find(new Sigma(*this, ArrayRef<const Type*>(0, 0)))->as<Sigma>())
    , pi0_  (find(new Pi   (*this, ArrayRef<const Type*>(0, 0)))->as<Pi>())
#define ANYDSL_UF_TYPE(T) ,T##_(find(new PrimType(*this, PrimType_##T))->as<PrimType>())
#include "anydsl/tables/primtypetable.h"
{}

World::~World() {
    for_all (def, defs_)
        delete def;
}

/*
 * types
 */

Sigma* World::namedSigma(size_t num, const std::string& name /*= ""*/) {
    Sigma* s = new Sigma(*this, num);
    s->debug = name;

    anydsl_assert(defs_.find(s) == defs_.end(), "must not be inside");
    defs_.insert(s);

    return s;
}

/*
 * literals
 */

const PrimLit* World::literal(PrimTypeKind kind, Box box) {
    return find(new PrimLit(type(kind), box))->as<PrimLit>();
}

const PrimLit* World::literal(PrimTypeKind kind, int value) {
    switch (kind) {
#define ANYDSL_U_TYPE(T) case PrimType_##T: return literal(T(value));
#define ANYDSL_F_TYPE(T) ANYDSL_U_TYPE(T)
#include "anydsl/tables/primtypetable.h"
        default: ANYDSL_UNREACHABLE;
    }
}

const Any* World::any(const Type* type) {
    return find(new Any(type))->as<Any>();
}

const Bottom* World::bottom(const Type* type) {
    return find(new Bottom(type))->as<Bottom>();
}

/*
 * create
 */

const Def* World::binop(int kind, const Def* lhs, const Def* rhs) {
    if (isArithOp(kind))
        return arithop((ArithOpKind) kind, lhs, rhs);

    anydsl_assert(isRelOp(kind), "must be a RelOp");
    return relop((RelOpKind) kind, lhs, rhs);
}

const Def* World::tuple(ArrayRef<const Def*> args) {
    Array<const Type*> elems(args.size());

    size_t i = 0;
    bool bot = false;

    for_all (arg, args) {
        elems[i++] = arg->type();

        if (arg->isa<Bottom>())
            bot = true;
    }

    if (bot)
        return bottom(sigma(elems));

    return find(new Tuple(*this, args));
}

const Def* World::arithop(ArithOpKind kind, const Def* a, const Def* b) {
    PrimTypeKind rtype = a->type()->as<PrimType>()->primtype_kind();

    // bottom op bottom -> bottom
    if (a->isa<Bottom>() || b->isa<Bottom>()) 
        return bottom(rtype);

    const PrimLit* llit = a->isa<PrimLit>();
    const PrimLit* rlit = b->isa<PrimLit>();

    if (llit && rlit) {
        Box l = llit->box();
        Box r = rlit->box();
        PrimTypeKind type = llit->primtype_kind();

        switch (kind) {
            case ArithOp_add:
                switch (type) {
#define ANYDSL_UF_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())));
#include "anydsl/tables/primtypetable.h"
                }
            case ArithOp_sub:
                switch (type) {
#define ANYDSL_UF_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())));
#include "anydsl/tables/primtypetable.h"
                }
            case ArithOp_mul:
                switch (type) {
#define ANYDSL_UF_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())));
#include "anydsl/tables/primtypetable.h"
                }
            case ArithOp_udiv:
                switch (type) {
#define ANYDSL_JUST_U_TYPE(T) \
                    case PrimType_##T: \
                        return rlit->isZero() \
                             ? (const Def*) bottom(rtype) \
                             : (const Def*) literal(type, Box(T(l.get_##T() / r.get_##T())));
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_F_TYPE;
                }
            case ArithOp_sdiv:
                switch (type) {
#define ANYDSL_JUST_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed<T>::type S; \
                        return literal(type, Box(bcast<T , S>(bcast<S, T >(l.get_##T()) / bcast<S, T >(r.get_##T())))); \
                    }
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_F_TYPE;
                }
            case ArithOp_fadd:
                switch (type) {
#define ANYDSL_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())));
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_U_TYPE;
                }
            case ArithOp_fsub:
                switch (type) {
#define ANYDSL_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())));
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_U_TYPE;
                }
            case ArithOp_fmul:
                switch (type) {
#define ANYDSL_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())));
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_U_TYPE;
                }
            case ArithOp_fdiv:
                switch (type) {
#define ANYDSL_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() / r.get_##T())));
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_U_TYPE;
                }
            case ArithOp_frem:
                switch (type) {
#define ANYDSL_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(std::fmod(l.get_##T(), r.get_##T())));
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_U_TYPE;
                }
            default: 
                ANYDSL_UNREACHABLE;
        }
    }

    // normalize -- put literal or smaller pointer to the left
    if (ArithOp::isCommutative(kind))
        if ((rlit || a > b) && (!llit))
            std::swap(a, b);

    return find(new ArithOp(kind, a, b));
}

const Def* World::relop(RelOpKind kind, const Def* a, const Def* b) {
    if (a->isa<Bottom>() || b->isa<Bottom>()) 
        return bottom(type_u1());

    RelOpKind oldkind = kind;
    switch (kind) {
        case RelOp_cmp_ugt:  kind = RelOp_cmp_ult; break;
        case RelOp_cmp_uge:  kind = RelOp_cmp_ule; break;
        case RelOp_cmp_sgt:  kind = RelOp_cmp_slt; break;
        case RelOp_cmp_sge:  kind = RelOp_cmp_sle; break;
        case RelOp_fcmp_ogt: kind = RelOp_fcmp_olt; break;
        case RelOp_fcmp_oge: kind = RelOp_fcmp_ole; break;
        case RelOp_fcmp_ugt: kind = RelOp_fcmp_ult; break;
        case RelOp_fcmp_uge: kind = RelOp_fcmp_ule; break;
        default: break;
    }

    if (oldkind != kind)
        std::swap(a, b);

    const PrimLit* llit = a->isa<PrimLit>();
    const PrimLit* rlit = b->isa<PrimLit>();

    if (llit && rlit) {
        Box l = llit->box();
        Box r = rlit->box();
        PrimTypeKind type = llit->primtype_kind();

        switch (kind) {
            case RelOp_cmp_eq:
                switch (type) {
#define ANYDSL_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() == r.get_##T());
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_F_TYPE;
                }
            case RelOp_cmp_ne:
                switch (type) {
#define ANYDSL_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() != r.get_##T());
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_F_TYPE;
                }
            case RelOp_cmp_ult:
                switch (type) {
#define ANYDSL_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <  r.get_##T());
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_F_TYPE;
                }
            case RelOp_cmp_ule:
                switch (type) {
#define ANYDSL_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <= r.get_##T());
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_F_TYPE;
                }
            case RelOp_cmp_slt:
                switch (type) {
#define ANYDSL_JUST_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed< T >::type S; \
                        return literal_u1(bcast<S, T>(l.get_##T()) <  bcast<S, T>(r.get_##T())); \
                    }
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_F_TYPE;
                }
            case RelOp_cmp_sle:
                switch (type) {
#define ANYDSL_JUST_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed< T >::type S; \
                        return literal_u1(bcast<S, T>(l.get_##T()) <= bcast<S, T>(r.get_##T())); \
                    }
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_F_TYPE;
                }
            case RelOp_fcmp_oeq:
                switch (type) {
#define ANYDSL_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() == r.get_##T());
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_U_TYPE;
                }
            case RelOp_fcmp_one:
                switch (type) {
#define ANYDSL_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() != r.get_##T());
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_U_TYPE;
                }
            case RelOp_fcmp_olt:
                switch (type) {
#define ANYDSL_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <  r.get_##T());
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_U_TYPE;
                }
            case RelOp_fcmp_ole:
                switch (type) {
#define ANYDSL_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <= r.get_##T());
#include "anydsl/tables/primtypetable.h"
                    ANYDSL_NO_U_TYPE;
                }
            default: 
                ANYDSL_UNREACHABLE;
        }
    }

    return find(new RelOp(kind, a, b));
}

const Def* World::convop(ConvOpKind kind, const Def* from, const Type* to) {
    if (from->isa<Bottom>())
        return bottom(to);

#if 0
    if (const PrimLit* lit = from->isa<PrimLit>())
        Box box = lit->box();
        PrimTypeKind type = lit->primtype_kind();

        // TODO folding
    }
#endif

    return find(new ConvOp(kind, from, to));
}

const Def* World::extract(const Def* agg, u32 i) {
    if (agg->isa<Bottom>())
        return bottom(agg->type()->as<Sigma>()->elem(i));

    if (const Tuple* tuple = agg->isa<Tuple>())
        return tuple->op(i);

    return find(new Extract(agg, i));
}

const Def* World::insert(const Def* agg, u32 index, const Def* value) {
    if (agg->isa<Bottom>() || value->isa<Bottom>())
        return bottom(agg->type());

    if (const Tuple* tup = agg->isa<Tuple>()) {
        Array<const Def*> args(tup->size());

        for (size_t i = 0, e = args.size(); i != e; ++i)
            if (i != index)
                args[i] = agg->op(i);
            else
                args[i] = value;

        return tuple(args);
    }

    return find(new Insert(agg, index, value));
}


const Def* World::select(const Def* cond, const Def* a, const Def* b) {
    if (cond->isa<Bottom>() || a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(a->type());

    if (const PrimLit* lit = cond->isa<PrimLit>())
        return lit->box().get_u1().get() ? a : b;

    return find(new Select(cond, a, b));
}

const Param* World::param(const Type* type, Lambda* parent, u32 i) {
    return find(new Param(type, parent, i))->as<Param>();
}

void World::jump(Lambda* lambda, const Def* to, ArrayRef<const Def*> args) {
    lambda->alloc(args.size() + 1);

    lambda->setOp(0, to);

    size_t x = 1;
    for_all (arg, args)
        lambda->setOp(x++, arg);

    lambda->close();

    find(lambda);
}

void World::branch(Lambda* lambda, const Def* cond, const Def* tto, const Def*  fto) {
    return jump(lambda, select(cond, tto, fto), Array<const Def*>(0));
}

/*
 * optimizations
 */

void World::dce() {
    // mark all as dead
    unmark();

    dce_insert(sigma0_);
    dce_insert(pi0_);
    for (size_t i = 0; i < Num_PrimTypes; ++i)
        dce_insert(primTypes_[i]);

    for_all_lambdas (lambda)
        if (lambda->isExtern()) {
            for (Params::const_iterator i = lambda->ho_begin(), e = lambda->ho_end(); i != e; lambda->ho_next(i)) {
                const Param* param = *i;
                for_all (use, param->uses())
                    dce_insert(use.def());
            }

        }

    // kill the living dead
    DefSet::iterator i = defs_.begin();
    while (i != defs_.end()) {
        const Def* def = *i;
        if (!def->marker_) {
            delete def;
            i = defs_.erase(i);
        } else
            ++i;
    }
}

void World::dce_insert(const Def* def) {
    if (def->marker_)
        return;

    def->marker_ = true;

    if (const Type* type = def->type())
        dce_insert(type);

    for_all (op, def->ops())
        dce_insert(op);

    if (const Lambda* lambda = def->isa<Lambda>()) {
        // insert control-dependent lambdas
        for_all (caller, lambda->callers())
            dce_insert(caller);
    } else if (const Param* param = def->isa<Param>()) {
        for_all (op, param->phiOps()) {
            // look through "phi-args"
            dce_insert(op.def());
            dce_insert(op.from());
        }
    }
}

void World::uce() {
    // mark all as unreachable
    unmark();

    // find all reachable lambdas
    for_all_lambdas (lambda)
        if (lambda->isExtern())
            uce_insert(lambda);

    // destroy all unreachable lambdas
    DefSet::iterator i = defs_.begin();
    while (i != defs_.end()) {
        if (const Lambda* lambda = (*i)->isa<Lambda>()) {
            if (!lambda->marker_) {
                delete lambda;
                i = defs_.erase(i);
                continue;
            }
        }
        ++i;
    }
}

void World::uce_insert(const Lambda* lambda) {
    assert(defs_.find(lambda) != defs_.end());

    if (lambda->marker_)
        return;

    lambda->marker_ = true;

    if (const Type* type = lambda->type())
        dce_insert(type);

    for_all (succ, lambda->succ())
        uce_insert(succ);
}

void World::cleanup() {
    uce();
    dce();
    std::cout << "yeah!!!" << std::endl;
}

void World::opt() {
    cleanup();
    cfg_simplify();
    cleanup();
    param_opt();
    cleanup();
}

void World::cfg_simplify() {
#if 0
    LambdaSet lambdas;
    for_all_lambdas (lambda)
        if (const Lambda* to = lambda->to()->isa<Lambda>())
            if (to->uses().size() == 1) {
                if (lambda
                assert(lambda == to->uses().begin()->def());
                lambdas.insert(lambda);
                lambdas.erase(to);
            }

        //if (lambda->pi()->size() != lambda->params().size())
            //q.push(lambda);
    LambdaSet lambdas;

    for_all (def, defs_)
        if (const Lambda* lambda = def->isa<Lambda>())
            if (const Lambda* to = lambda->to()->isa<Lambda>())
                if (to->uses().size() == 1)
                    if (lambda == to->uses().begin()->def())
                        lambdas.insert(lambda);

//#endif

    while (!lambdas.empty()) {
        const Lambda* lambda = *lambdas.begin();

        if (const Lambda* to = lambda->to()->isa<Lambda>())
            if (to->uses().size() == 1)
                if (!to->isExtern()) // HACK
                    if (lambda == to->uses().begin()->def()) {
                        Lambda* newl = new Lambda(lambda->pi());
                        newl->debug = lambda->debug + "+" + to->debug;
                        jump(newl, to->to(), to->args());

                        Params::const_iterator i = to->params().begin();
                        for_all (arg, lambda->args())
                            replace(*i++, arg);

                        replace(lambda, newl);

                        lambdas.erase(to);
                        lambdas.insert(newl);
                    }

        lambdas.erase(lambda);
    }
#endif
}

void World::param_opt() {
    std::queue<const Lambda*> q;
    for_all_lambdas (lambda)
        if (lambda->pi()->size() != lambda->params().size())
            q.push(lambda);

    while (!q.empty()) {
        const Lambda* lambda = q.back();
        q.pop();

        size_t i = 0;
        for_all (param, lambda->params())
            if (param->index() == i)
                ++i;
            else
                for (; i < param->index(); ++i)
                    for_all (use, lambda->uses())
                        if (const Lambda* caller = use.def()->isa<Lambda>())
                            update(caller, i+1, bottom(lambda->pi()->elem(i)));
    }

#if 0
    for_all (def, defs_)
        if (const Lambda* lambda = def->isa<Lambda>()) {
            const Pi* pi = lambda->pi();

            if (!pi->empty() && reachable_.find(lambda) == reachable_.end()) { // HACK
                std::vector<size_t> keep;

                for_all (param, lambda->params()) {
                    const Def* same = 0;
                    // find Horspool-like phis
                    for_all (op, param->phiOps()) {
                        const Def* def = op.def();

                        if (def->isa<Undef>() || def == param || same == def)
                            continue;

                        if (same) {
                            keep.push_back(param->index());
                            break;
                        }

                        same = def;
                    }
                }

                if (keep.size() != pi->size()) {
                    std::cout << "superfluous in: " << lambda->debug << std::endl;
                    Array<const Type*> elems(keep.size());

                    size_t i = 0;
                    for_all (param, lambda->params())
                        if (param->index() == keep[i])
                            elems[i++] = param->type();

                    assert(i == keep.size());
                    const Pi* newpi = this->pi(elems);
                    Lambda* newl = new Lambda(newpi);
                    newl->debug = lambda->debug;
                    jump(newl, lambda->to(), lambda->args());
                    //replace(lambda, newl);
                }
            }
        }
//#if 0
        in_.erase(index);
        for_all (pred, preds_)
            pred->out_.erase(index);

        Params::const_iterator i = newl->params().begin();
        for_all (p, topLambda_->params())
            if (p->index() != index)
                world().replace(p, *i++);
    }
#endif
}


void World::unmark() {
    for_all (def, defs_)
        def->marker_ = false;
}

const Def* World::find(const Def* def) {
    DefSet::iterator i = defs_.find(def);
    if (i != defs_.end()) {
        anydsl_assert(!def->isa<Lambda>(), "must not be a lambda");
        delete def;
        anydsl_assert(defs_.find(*i) != defs_.end(), "hash/equal function of def class incorrect");
        return *i;
    }

    defs_.insert(def);
    anydsl_assert(defs_.find(def) != defs_.end(), "hash/equal function of def class incorrect");

    return def;
}

/*
 * other
 */

void World::dump(bool fancy) {
    unmark();

    for_all_lambdas (lambda) {
        if (!lambda->isExtern() || lambda->marker_)
            continue;

        std::queue<const Lambda*> queue;
        queue.push(lambda);
        lambda->marker_ = true;

        while (!queue.empty()) {
            const Lambda* cur = queue.front();
            queue.pop();

            cur->dump(fancy);
            std::cout << std::endl;

            for_all (succ, cur->succ()) {
                if (!succ->marker_ && !succ->isExtern()) {
                    succ->marker_ = true;
                    queue.push(succ);
                }
            }
        }
    }
}

Def* World::release(const Def* def) {
    DefSet::iterator i = defs_.find(def);
    anydsl_assert(i != defs_.end(), "must be found");
    assert(def == *i);
    defs_.erase(i);

    return const_cast<Def*>(def);
}

void World::replace(const Def* what, const Def* with) {
    if (what == with)
        return;

    Def* def = release(what);
    Lambda* lambda = def->isa<Lambda>();

    // unregister all uses of def's operands
    for (size_t i = 0, e = def->ops().size(); i != e; ++i) {
        def->ops_[i]->unregisterUse(i, def);
        def->ops_[i] = 0;
    }

    Array<Use> olduses = def->copyUses();

    // unregister all uses of def
    def->uses_.clear();

    // update all operands of old uses to point to new node instead 
    // and erase these nodes from world
    for_all (use, olduses) {
        Def* udef = release(use.def());
        udef->update(use.index(), with);
    }

    // update all operands of old uses into world
    // don't fuse this loop with the loop above
    for_all (use, olduses) {
        const Def* udef = use.def();

        DefSet::iterator i = defs_.find(udef);
        if (i != defs_.end()) {
            std::cout << "NOT YET TESTED" << std::endl;
            const Def* ndef = *i;
            assert(udef != ndef);
            replace(udef, ndef);
            delete udef;
            continue;
        } else
            defs_.insert(udef);
    }

    if (lambda) {
        Params::const_iterator i = with->as<Lambda>()->params().begin();

        Array<const Param*> params(lambda->params().size());
        std::copy(lambda->params().begin(), lambda->params().end(), params.begin());

        for_all (param, params) {
            while ((*i)->index() < param->index())
                ++i;

            const Param* newparam = *i;
            newparam->debug = param->debug;
            replace(param, newparam);
        }
    }

    delete def;
}

const Def* World::update(const Def* cdef, size_t i, const Def* op) {
    Def* def = release(cdef);
    def->update(i, op);

    return find(def);
}

const Def* World::update(const Def* cdef, ArrayRef<size_t> x, ArrayRef<const Def*> ops) {
    Def* def = release(cdef);
    def->update(x, ops);

    return find(def);
}

const Lambda* World::drop(const Lambda* lambda, ArrayRef<size_t> args, ArrayRef<const Def*> with) {
    // build new type
    const Pi* pi = lambda->pi();
    Array<const Type*> elems(pi->size() - args.size());

    // r -> read, w -> write, a -> args
    for (size_t r = 0, w = 0, a = 0, e = pi->size(); r != e; ++r)
        if (a < args.size() && args[a] == r)
            ++a;
        else
            elems[w++] = pi->elem(r);

    const Pi* npi = this->pi(elems);
    Lambda* dropped = new Lambda(npi, lambda->flags());
    dropped->alloc(lambda->size());

    // old2new[def] = not found     -> not yet examined
    // old2new[def] = def           -> must not be dropped
    // old2new[def] = ndef          -> must be dropped with ndef
    Old2New old2new;
    old2new[lambda] = lambda;       // don't drop

    size_t a = 0;
    Params::const_iterator d = dropped->params().begin();
    for_all (param, lambda->params()) {
        if (param->index() == args[a])
            old2new[param] = with[a++]; // map old param to replacement
        else {
            const Param* dparam = *d;
            while (dparam->index() < param->index())
                dparam = *d++;          // skip all dead params

            old2new[param] = dparam;    // map old param to new param
        }
    }

    for (size_t i = 0, e = dropped->size(); i != e; ++i)
        dropped->setOp(i, drop(old2new, lambda->op(i)));

    return find(dropped)->as<Lambda>();
}

const Def* World::drop(Old2New& old2new, const Def* def) {
#if 0
    Old2New::iterator it = old2new.find(def);

    if (res != old2new.end())
        return res;

    if (it == old2new.end()) {
        mustDrop(old2new, def);
        it = old2new.find(def);
    }

    if (const Def* def = it->second)
        return def;

    if (const Lambda* lambda = def->isa<Lambda>()) {
        Lambda* dropped = new Lambda(lambda->pi(), lambda->flags());
        dropped->alloc(lambda->size());

        for (size_t i = 0, e = dropped->size(); i != e; ++i)
            dropped->setOp(i, drop(old2new, lambda->op(i)));

        return it->second = find(lambda);
    }

    Def* clone = def->clone();
    for (size_t i = 0, e = clone->size(); i != e; ++i)
        clone->setOp(i, drop(old2new, def->op(i)));

    return it->second = clone;
#endif
    return 0;
}

World::Old2New::iterator World::mustDrop(Old2New& old2new, const Def* def) {
#if 0
    Old2New::iterator res = old2new.find(def);

    if (res != old2new.end())
        return res;

    // optimistically assume that we don't have to drop
    res = old2new.insert(std::make_pair(def, def)).first;

    if (const Param* param = def->isa<Param>()) {
        if (mustDrop(old2new, param->lambda()))
            return true;
    }

    for_all (op, def->ops()) {
        if (musDrop(old2new, op)->) {
            // after all, we must drop
            old2new[def] = 0;
            return true;
        }
    }

    return false;
#endif
    return old2new.begin();
}

static void depends_simple(const Def* def, LambdaSet* dep) {
    if (const Param* param = def->isa<Param>())
        dep->insert(param->lambda());
    else if (!def->isa<Lambda>()) {
        for_all (op, def->ops())
            depends_simple(op, dep);
    }
}

void World::group() {
    std::queue<const Lambda*> queue;
    LambdaSet inqueue;

    typedef boost::unordered_map<const Lambda*, LambdaSet*> DepMap;
    DepMap depmap;

    for_all_lambdas (lambda) {
        LambdaSet* dep = new LambdaSet();

        for_all (op, lambda->ops())
            depends_simple(op, dep);

        depmap[lambda] = dep;
        queue.push(lambda);
        inqueue.insert(lambda);
    }

    while (!queue.empty()) {
        const Lambda* lambda = queue.front();
        queue.pop();
        inqueue.erase(lambda);
        LambdaSet* dep = depmap[lambda];
        size_t old = dep->size();

        for_all (succ, lambda->succ()) {
            LambdaSet* succ_dep = depmap[succ];

            for_all (d, *succ_dep) {
                if (d != succ)
                    dep->insert(d);
            }
        }

        if (dep->size() != old) {
            for_all (caller, lambda->callers()) {
                if (inqueue.find(caller) == inqueue.end()) {
                    inqueue.insert(caller);
                    queue.push(caller);
                }
            }
        }
    }

#if 0
    for_all (p, depmap) {
        std::cout << p.first->debug << std::endl;

        for_all (l, *p.second)
            std::cout << "\t" << l->debug << std::endl;
    }
#endif

    for_all (p, depmap)
        delete p.second;
}

/*
 * debug printing
 */

void World::printPostOrder() {
    PostOrder order(*defs_.begin());
    for(PostOrder::iterator it = order.begin(), e = order.end();
        it != e; ++it) {
        const Def* d = *it;
        if(d->isa<Lambda>()) {
            d->dump(false);
        }
    }
}

void World::printReversePostOrder() {
    PostOrder order(*defs_.begin());
    for(PostOrder::reverse_iterator it = order.rbegin(), e = order.rend();
        it != e; ++it) {
        const Def* d = *it;
        if(d->isa<Lambda>()) {
            d->dump(false);
        }
    }
}

void World::printDominators() {
    Dominators doms(*defs_.begin());
    for(Dominators::const_iterator it = doms.begin(), e = doms.end();
        it != e; ++it) {
        const Def* d = it->first;
        const Def* t = it->second;
        if(d->isa<Lambda>()) {
            Printer p(std::cout, false);
            t->vdump(p);
            std::cout << " --> ";
            d->vdump(p);
            std::cout << std::endl;
        }
    }
}

void World::hack() {
    group();
#if 0
    const Lambda* l;
    const Def* with;

    for_all (def, defs_)
        if (def->debug == "fac_tail")
            l = def->as<Lambda>();
        else if (def->debug == "fac")
            with = def->as<Lambda>()->arg(2);

    size_t args[] = {2};
    const Def* withs[] = {with};
    drop(l, args, withs);
#endif
}

} // namespace anydsl
