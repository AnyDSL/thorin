#include "anydsl/world.h"

#include <cmath>
#include <algorithm>
#include <queue>

#include "anydsl/def.h"
#include "anydsl/primop.h"
#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/type.h"
#include "anydsl/analyses/domtree.h"
#include "anydsl/analyses/rootlambdas.h"
#include "anydsl/analyses/scope.h"
#include "anydsl/util/array.h"
#include "anydsl/util/for_all.h"

#define ANYDSL_NO_U_TYPE \
    case PrimType_u1: \
    case PrimType_u8: \
    case PrimType_u16: \
    case PrimType_u32: \
    case PrimType_u64: ANYDSL_UNREACHABLE;

#define ANYDSL_NO_F_TYPE \
    case PrimType_f32: \
    case PrimType_f64: ANYDSL_UNREACHABLE;

namespace anydsl {

/*
 * constructor and destructor
 */

World::World()
    : defs_(1031)
    , gid_counter_(0)
    , sigma0_ (consume(new Sigma(*this, ArrayRef<const Type*>(0, 0)))->as<Sigma>())
    , pi0_  (consume(new Pi   (*this, ArrayRef<const Type*>(0, 0)))->as<Pi>())
#define ANYDSL_UF_TYPE(T) ,T##_(consume(new PrimType(*this, PrimType_##T))->as<PrimType>())
#include "anydsl/tables/primtypetable.h"
{}

World::~World() {
    for_all (def, defs_)
        delete def;
}

/*
 * types
 */

Sigma* World::named_sigma(size_t num, const std::string& name) {
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
    return consume(new PrimLit(type(kind), box))->as<PrimLit>();
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
    return consume(new Any(type))->as<Any>();
}

const Bottom* World::bottom(const Type* type) {
    return consume(new Bottom(type))->as<Bottom>();
}

/*
 * create
 */

const Def* World::binop(int kind, const Def* lhs, const Def* rhs) {
    if (is_arithop(kind))
        return arithop((ArithOpKind) kind, lhs, rhs);

    anydsl_assert(is_relop(kind), "must be a RelOp");
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

    return consume(new Tuple(*this, args));
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
                        return rlit->is_zero() \
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
    if (ArithOp::is_commutative(kind))
        if ((rlit || a > b) && (!llit))
            std::swap(a, b);

    return consume(new ArithOp(kind, a, b));
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

    return consume(new RelOp(kind, a, b));
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

    return consume(new ConvOp(kind, from, to));
}

const Def* World::extract(const Def* agg, u32 index) {
    if (agg->isa<Bottom>())
        return bottom(agg->type()->as<Sigma>()->elem(index));

    if (const Tuple* tuple = agg->isa<Tuple>())
        return tuple->op(index);

    if (const Insert* insert = agg->isa<Insert>()) {
        if (index == insert->index())
            return insert->value();
        else
            return extract(insert->tuple(), index);
    }

    return consume(new Extract(agg, index));
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

    return consume(new Insert(agg, index, value));
}

const Def* World::select(const Def* cond, const Def* a, const Def* b) {
    if (cond->isa<Bottom>() || a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(a->type());

    if (const PrimLit* lit = cond->isa<PrimLit>())
        return lit->box().get_u1().get() ? a : b;

    return consume(new Select(cond, a, b));
}

const Param* World::param(const Type* type, Lambda* parent, u32 i) {
    return consume(new Param(type, parent, i))->as<Param>();
}

Lambda* World::lambda(const Pi* pi, uint32_t flags) {
    Lambda* l = new Lambda(gid_counter_++, pi, flags);
    defs_.insert(l);

    size_t i = 0;
    for_all (elem, pi->elems())
        l->params_.push_back(param(elem, l, i++));

    return l;
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

    for_all (lambda, lambdas()) {
        if (lambda->is_extern()) {
            for_all (param, lambda->ho_params()) {
                for_all (use, param->uses())
                    dce_insert(use.def());
            }
        }
    }

    // kill the living dead
    DefSet::iterator i = defs_.begin();
    while (i != defs_.end()) {
        const Def* def = *i;
        if (!def->is_marked()) {
            delete def;
            i = defs_.erase(i);
        } else
            ++i;
    }
}

void World::dce_insert(const Def* def) {
    anydsl_assert(defs_.find(def) != defs_.end(), "not in map");

    if (def->is_marked())
        return;

    def->mark();

    if (const Type* type = def->type())
        dce_insert(type);

    for_all (op, def->ops())
        dce_insert(op);

    if (const Lambda* lambda = def->isa<Lambda>()) {
        // insert control-dependent lambdas
        for_all (pred, lambda->preds())
            dce_insert(pred);
    } else if (const Param* param = def->isa<Param>()) {
        for_all (op, param->phi()) {
            // look through "phi-args"
            dce_insert(op.def());
            dce_insert(op.from());
        }

        // always consider all params in the same lambda as live
        for_all (other, param->lambda()->params())
            dce_insert(other);
    }
}

void World::uce() {
    // mark all as unreachable
    unmark();

    // find all reachable lambdas
    for_all (lambda, lambdas())
        if (lambda->is_extern())
            uce_insert(lambda);

    // destroy all unreachable lambdas
    DefSet::iterator i = defs_.begin();
    while (i != defs_.end()) {
        if (const Lambda* lambda = (*i)->isa<Lambda>()) {
            if (!lambda->is_marked()) {
                delete lambda;
                i = defs_.erase(i);
                continue;
            }
        }
        ++i;
    }
}

void World::uce_insert(const Lambda* lambda) {
    anydsl_assert(defs_.find(lambda) != defs_.end(), "not in map");

    if (lambda->is_marked())
        return;

    lambda->mark();

    if (const Type* type = lambda->type())
        dce_insert(type);

    for_all (succ, lambda->succs())
        uce_insert(succ);
}

void World::cleanup() {
    dce();
    uce();
}

void World::opt() {
    cleanup();

    const Lambda* helper;
    const Lambda* fac;
    for_all (lambda, lambdas()) {
        if (lambda->debug == "helper")
            helper = lambda;
        else if (lambda->debug == "fac")
            fac = lambda;
    }

    const Lambda* dropped = drop(helper, fac->args());
    //return;
    Lambda* fac2 = (Lambda*) fac;
    //Lambda* fac2 = release(fac)->as<Lambda>();

    for (size_t i = 1, e = fac->size(); i != e; ++i)
        fac2->unset_op(i);

    fac2->shrink(1);
    //((Lambda*) helper)->flags_ = 0;
    fac2->update(0, dropped);
    //fac2->reclose();
    //consume(fac2);

    cleanup();
}

void World::unmark() {
    for_all (def, defs_)
        def->unmark();
}

const Def* World::consume(const Def* def) {
    DefSet::iterator i = defs_.find(def);
    if (i != defs_.end()) {
        anydsl_assert(!def->isa<Lambda>(), "must not be a lambda");
        delete def;
        anydsl_assert(defs_.find(*i) != defs_.end(), "hash/equal function of def class incorrect");
        return *i;
    }

    defs_.insert(def);
    anydsl_assert(defs_.find(def) != defs_.end(), "hash/equal function of def class incorrect");

    if (const Lambda* lambda = def->isa<Lambda>()) {
        for_all (param, lambda->params())
            consume(param);
    }

    return def;
}

/*
 * other
 */

LambdaSet World::lambdas() const {
    LambdaSet result;

    for_all (def, defs())
        if (const Lambda* lambda = def->isa<Lambda>())
            result.insert(lambda);

    return result;
}

void World::dump(bool fancy) {
    //for_all (lambda, lambdas())
        //lambda->dump(fancy);
    //return;
    LambdaSet roots = find_root_lambdas(lambdas());

    for_all (root, roots) {
        ScopeTree scope(root);

        for_all (lambda, scope.rpo()) {
            int indent = scope.depth(lambda);
            lambda->dump(fancy, indent);
        }
    }

    std::cout << std::endl;
}

Def* World::release(const Def* def) {
    DefSet::iterator i = defs_.find(def);
    anydsl_assert(i != defs_.end(), "must be found");
    assert(def == *i);
    defs_.erase(i);

    if (const Lambda* lambda = def->isa<Lambda>()) {
        for_all (param, lambda->params())
            release(param);
    }

    return const_cast<Def*>(def);
}

void World::replace(const Def* what, const Def* with) {
    assert(!what->isa<Lambda>());

    if (what == with)
        return;

    Def* def = release(what);

    // unregister all uses of def's operands
    for (size_t i = 0, e = def->ops().size(); i != e; ++i) {
        def->unregister_use(i);
        def->ops_[i] = 0;
    }

    Array<Use> olduses = def->copy_uses();

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

    delete def;
}

const Def* World::update(const Def* cdef, size_t i, const Def* op) {
    Def* def = release(cdef);
    def->update(i, op);
    return consume(def);
}

const Def* World::update(const Def* cdef, ArrayRef<const Def*> ops) {
    Def* def = release(cdef);
    def->update(ops);
    return consume(def);
}

const Def* World::update(const Def* cdef, ArrayRef<size_t> x, ArrayRef<const Def*> ops) {
    Def* def = release(cdef);
    def->update(x, ops);
    return consume(def);
}

const Lambda* World::drop(const Lambda* olambda, ArrayRef<const Def*> with) {
    if (with.empty())
        return olambda;

    Array<size_t> args(with.size());
    for (size_t i = 0, e = args.size(); i < e; ++i)
        args[i] = i;

    return drop(olambda, args, with);
}

const Lambda* World::drop(const Lambda* olambda, ArrayRef<size_t> args, ArrayRef<const Def*> with) {
    if (with.empty())
        return olambda;

    Scope scope(olambda);

    const Pi* o_pi = olambda->pi();

    size_t o_numargs = o_pi->size();
    size_t numdrop = args.size();
    size_t n_numargs = o_numargs - numdrop;

    Array<const Type*> elems(n_numargs);

    for (size_t i = 0, a = 0, e = 0; i < o_numargs; ++i) {
        if (a < o_numargs && args[a] == i)
            ++a;
        else
            elems[e++] = o_pi->elem(i);
    }

    const Pi* n_pi = pi(elems);
    Lambda* nlambda = this->lambda(n_pi);
    nlambda->debug = olambda->debug + ".dropped";

    Old2New old2new;

    // put in params for entry (olambda)
    for (size_t i = 0, a = 0, n = 0; i < o_numargs; ++i) {
        const Param* oparam = olambda->param(i);
        if (a < o_numargs && args[a] == i) {
            const Def* w = with[a++];
            old2new[oparam] = w; //with[a++];
        } else {
            const Param* nparam = nlambda->param(n++);
            nparam->debug = oparam->debug + ".dropped";
            old2new[oparam] = nparam;
        }
    }

    old2new[olambda] = nlambda;

    // create stubs for all other lambdas and put their params into the map
    for_all (olambda, scope.rpo().slice_back(1)) {
        Lambda* nlambda = olambda->stub();
        nlambda->debug += ".dropped";
        old2new[olambda] = nlambda;

        for (size_t i = 0, e = nlambda->params().size(); i != e; ++i) {
            old2new[olambda->param(i)] = nlambda->param(i);
            nlambda->param(i)->debug = "xxx";
        }
    }

    for_all (cur, scope.rpo())
        drop_body(old2new, cur, (Lambda*) old2new[cur]);

    return nlambda;
}

void World::drop_body(Old2New& old2new, const Lambda* olambda, Lambda* nlambda) {
    for_all (param, olambda->params())
        for_all (use, param->uses())
            if (const PrimOp* primop = use.def()->isa<PrimOp>())
                drop(old2new, primop);

    Array<const Def*> args(olambda->args().size());
    for (size_t i = 0; i < args.size(); ++i) {
        const Def* odef = olambda->arg(i);
        Old2New::iterator iter = old2new.find(odef);
        if (iter != old2new.end())
            args[i] = iter->second;
        else
            args[i] = odef;
    }

    nlambda->jump(drop_to(old2new, olambda->to()), args);
}

void World::drop(Old2New& old2new, const PrimOp* primop) {
    Array<const Def*> ops(primop->size());

    size_t i = 0; 
    for_all (op, primop->ops()) {
        Old2New::iterator iter = old2new.find(op);
        if (iter != old2new.end())
            ops[i++] = iter->second;
        else if (op->is_const())
            ops[i++] = op;
        else 
            return;
    }

    old2new[primop] = consume(primop->clone()->update(ops));

    for_all (use, primop->uses())
        if (const PrimOp* primop = use.def()->isa<PrimOp>())
            drop(old2new, primop);
}

const Def* World::drop_to(Old2New& old2new, const Def* to) {
    Old2New::iterator iter = old2new.find(to);
    if (iter != old2new.end())
        return iter->second;

    if (const PrimOp* primop = to->isa<PrimOp>()) {
        Array<const Def*> ops(primop->size());

        size_t i = 0; 
        for_all (op, primop->ops()) {
            Old2New::iterator iter = old2new.find(op);
            if (iter != old2new.end())
                ops[i++] = iter->second;
            else
                ops[i++] = drop_to(old2new, op);

        }

        return old2new[primop] = consume(primop->clone()->update(ops));
    }

    return to;
}

} // namespace anydsl
