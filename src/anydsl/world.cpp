#include "anydsl/world.h"

#include <cmath>
#include <algorithm>
#include <queue>

#include "anydsl/def.h"
#include "anydsl/primop.h"
#include "anydsl/lambda.h"
#include "anydsl/literal.h"
#include "anydsl/memop.h"
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

namespace anydsl2 {

/*
 * constructor and destructor
 */

World::World()
    : primops_(1031)
    , lambdas_(1031)
    , types_(1031)
    , gid_counter_(0)
    , sigma0_ (consume(new Sigma(*this, ArrayRef<const Type*>(0, 0)))->as<Sigma>())
    , pi0_    (consume(new Pi   (*this, ArrayRef<const Type*>(0, 0)))->as<Pi>())
    , mem_    (consume(new Mem  (*this))->as<Mem>())
    , frame_  (consume(new Frame(*this))->as<Frame>())
#define ANYDSL_UF_TYPE(T) ,T##_(consume(new PrimType(*this, PrimType_##T))->as<PrimType>())
#include "anydsl/tables/primtypetable.h"
{
    typekeeper(sigma0_);
    typekeeper(pi0_);
    typekeeper(mem_);
    typekeeper(frame_);
    for (size_t i = 0; i < Num_PrimTypes; ++i)
        typekeeper(primTypes_[i]);
}

World::~World() {
    for_all (primop, primops_)
        delete primop;
    for_all (type, types_)
        delete type;
    for_all (lambda, lambdas_)
        delete lambda;
}

/*
 * types
 */

Sigma* World::named_sigma(size_t num, const std::string& name) {
    Sigma* s = new Sigma(*this, num);
    s->debug = name;

    anydsl_assert(types_.find(s) == types_.end(), "must not be inside");
    types_.insert(s);

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

const Def* World::convop(ConvOpKind kind, const Type* to, const Def* from) {
    if (from->isa<Bottom>())
        return bottom(to);

#if 0
    if (const PrimLit* lit = from->isa<PrimLit>())
        Box box = lit->box();
        PrimTypeKind type = lit->primtype_kind();

        // TODO folding
    }
#endif

    return consume(new ConvOp(kind, to, from));
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

const Def* World::load(const Def* mem, const Def* ptr) {
    return consume(new Load(mem, ptr));
}

const Def* World::store(const Def* mem, const Def* ptr, const Def* val) {
    return consume(new Store(mem, ptr, val));
}

const Enter* World::enter(const Def* mem) {
    return consume(new Enter(mem))->as<Enter>();
}

const Leave* World::leave(const Def* mem, const Def* frame) {
    return consume(new Leave(mem, frame))->as<Leave>();
}

const Slot* World::slot(const Enter* enter, const Type* type) {
    return consume(new Slot(enter, type))->as<Slot>();
}

const Def* World::select(const Def* cond, const Def* a, const Def* b) {
    if (cond->isa<Bottom>() || a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(a->type());

    if (const PrimLit* lit = cond->isa<PrimLit>())
        return lit->box().get_u1().get() ? a : b;

    return consume(new Select(cond, a, b));
}

const Def* World::typekeeper(const Type* type) { 
    return consume(new TypeKeeper(type)); 
}

Lambda* World::lambda(const Pi* pi, uint32_t flags) {
    Lambda* l = new Lambda(gid_counter_++, pi, flags);
    lambdas_.insert(l);

    size_t i = 0;
    for_all (elem, pi->elems())
        l->params_.push_back(new Param(elem, l, i++));

    return l;
}

/*
 * optimizations
 */

void World::dead_code_elimination() {
    for_all (lambda, lambdas()) {
        lambda->unmark(); 
        for_all (param, lambda->params())
            param->unmark();
    }

    for_all (primop, primops()) 
        primop->unmark(); 

    for_all (primop, primops()) {
        if (const TypeKeeper* tk = primop->isa<TypeKeeper>())
            dce_insert(tk);
    }

    for_all (lambda, lambdas()) {
        if (lambda->is_extern()) {
            for_all (param, lambda->ho_params()) {
                for_all (use, param->uses())
                    dce_insert(use.def());
            }
        }
    }

    for (PrimOpSet::iterator i = primops_.begin(); i != primops_.end();) {
        const PrimOp* primop = *i;
        if (primop->is_marked()) 
            ++i;
        else {
            delete primop;
            i = primops_.erase(i);
        }
    }

    for (LambdaSet::iterator i = lambdas_.begin(); i != lambdas_.end();) {
        Lambda* lambda = *i;
        if (lambda->is_marked()) 
            ++i;
        else {
            delete lambda;
            i = lambdas_.erase(i);
        }
    }
}

void World::dce_insert(const Def* def) {
#ifndef NDEBUG
    if (const PrimOp* primop = def->isa<PrimOp>()) assert(primops_.find(primop)          != primops_.end());
    if (      Lambda* lambda = def->isa_lambda() ) assert(lambdas_.find(lambda)          != lambdas_.end());
    if (const Param*  param  = def->isa<Param>() ) assert(lambdas_.find(param->lambda()) != lambdas_.end());
#endif

    if (def->is_marked()) return;
    def->mark();

    for_all (op, def->ops())
        dce_insert(op);

    if (Lambda* lambda = def->isa_lambda()) {
        // insert control-dependent lambdas
        for_all (pred, lambda->preds())
            dce_insert(pred);
    } else if (const Param* param = def->isa<Param>()) {
        for_all (op, param->phi()) {
            dce_insert(op.def());
            dce_insert(op.from());
        }

        // always consider all params in the same lambda as live
        for_all (other, param->lambda()->params())
            dce_insert(other);
    }
}

void World::unused_type_elimination() {
    for_all (type, types()) 
        type->unmark(); 

    for_all (primop, primops())
        ute_insert(primop->type());

    for_all (lambda, lambdas()) {
        ute_insert(lambda->type());
        for_all (param, lambda->params())
            ute_insert(param->type());
    }

    for (TypeSet::iterator i = types_.begin(); i != types_.end();) {
        const Type* type = *i;

        if (type->is_marked())
            ++i;
        else {
            delete type;
            i = types_.erase(i);
        }
    }
}

void World::ute_insert(const Type* type) {
    anydsl_assert(types_.find(type) != types_.end(), "not in map");

    if (type->is_marked()) return;
    type->mark();

    for_all (elem, type->elems())
        ute_insert(elem);
}


void World::unreachable_code_elimination() {
    for_all (lambda, lambdas()) 
        lambda->unmark(); 

    for_all (lambda, lambdas())
        if (lambda->is_extern())
            uce_insert(lambda);

    for (LambdaSet::iterator i = lambdas_.begin(); i != lambdas_.end();) {
        Lambda* lambda = *i;

        if (lambda->is_marked()) 
            ++i;
        else {
            delete lambda;
            i = lambdas_.erase(i);
        }
    }
}

void World::uce_insert(Lambda* lambda) {
    anydsl_assert(lambdas_.find(lambda) != lambdas_.end(), "not in map");

    if (lambda->is_marked()) return;
    lambda->mark();

    for_all (succ, lambda->succs())
        uce_insert(succ);
}

void World::cleanup() {
    dead_code_elimination();
    unreachable_code_elimination();
    unused_type_elimination();
}

void World::opt() {
    cleanup();

    Lambda* helper;
    Lambda* fac;
    Lambda* ifelse;
    for_all (lambda, lambdas()) {
        if (lambda->debug == "helper")
            helper = lambda;
        else if (lambda->debug == "fac")
            fac = lambda;
        else if (lambda->debug == "<if-else-01>")
            ifelse = lambda;
    }

    Lambda* dropped = helper->drop(3, fac->param(1));
    ifelse->unset_op(4);
    ifelse->shrink(4);
    ifelse->update(0, dropped);

    cleanup();
}

const PrimOp* World::consume(const PrimOp* primop) {
    PrimOpSet::iterator i = primops_.find(primop);
    if (i != primops_.end()) {
        delete primop;
        anydsl_assert(primops_.find(*i) != primops_.end(), "hash/equal function of primop class incorrect");
        return *i;
    }

    primops_.insert(primop);
    anydsl_assert(primops_.find(primop) != primops_.end(), "hash/equal function of def class incorrect");

    return primop;
}

const Type* World::consume(const Type* type) {
    TypeSet::iterator i = types_.find(type);
    if (i != types_.end()) {
        delete type;
        anydsl_assert(types_.find(*i) != types_.end(), "hash/equal function of type class incorrect");
        return *i;
    }

    types_.insert(type);
    anydsl_assert(types_.find(type) != types_.end(), "hash/equal function of def class incorrect");

    return type;
}

PrimOp* World::release(const PrimOp* primop) {
    PrimOpSet::iterator i = primops_.find(primop);
    anydsl_assert(i != primops_.end(), "must be found");
    assert(primop == *i);
    primops_.erase(i);

    return const_cast<PrimOp*>(primop);
}

/*
 * other
 */

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

#if 0
void World::replace(const PrimOp* what, const PrimOp* with) {
    assert(!what->isa<Lambda>());

    if (what == with)
        return;

    PrimOp* primop = release(what);

    // unregister all uses of primop's operands
    for (size_t i = 0, e = primop->ops().size(); i != e; ++i) {
        primop->unregister_use(i);
        primop->ops_[i] = 0;
    }

    Array<Use> olduses = primop->copy_uses();

    // unregister all uses of primop
    primop->uses_.clear();

    // update all operands of old uses to point to new node instead 
    // and erase these nodes from world
    for_all (use, olduses) {
        PrimOp* uprimop = release(use.def());
        uprimop->update(use.index(), with);
    }

    // update all operands of old uses into world
    // don't fuse this loop with the loop above
    for_all (use, olduses) {
        const PrimOp* uprimop = use.def();
        PrimOpSet::iterator i = primops_.find(uprimop);

        if (i != primops_.end()) {
            std::cout << "NOT YET TESTED" << std::endl;
            const PrimOp* nprimop = *i;
            assert(uprimop != nprimop);
            replace(uprimop, nprimop);
            delete uprimop;
        } else
            primops_.insert(udef);
    }

    delete def;
}
#endif

const Def* World::update(const Def* what, size_t i, const Def* op) {
    if (Lambda* lambda = what->isa_lambda()) {
        lambda->update(i, op);
        return lambda;
    }

    PrimOp* primop = release(what->as<PrimOp>());
    primop->update(i, op);
    return consume(primop);
}

const Def* World::update(const Def* what, Array<const Def*> ops) {
    if (Lambda* lambda = what->isa_lambda()) {
        lambda->update(ops);
        return lambda;
    }

    PrimOp* primop = release(what->as<PrimOp>());
    primop->update(ops);
    return consume(primop);
}

} // namespace anydsl2
