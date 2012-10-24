#include "anydsl2/world.h"

#include <cmath>
#include <algorithm>
#include <queue>
#include <iostream>

#include "anydsl2/def.h"
#include "anydsl2/primop.h"
#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"
#include "anydsl2/type.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/rootlambdas.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/for_all.h"

#define ANYDSL2_NO_U_TYPE \
    case PrimType_u1: \
    case PrimType_u8: \
    case PrimType_u16: \
    case PrimType_u32: \
    case PrimType_u64: ANYDSL2_UNREACHABLE;

#define ANYDSL2_NO_F_TYPE \
    case PrimType_f32: \
    case PrimType_f64: ANYDSL2_UNREACHABLE;

namespace anydsl2 {

/*
 * constructor and destructor
 */

World::World()
    : primops_(1031)
    , lambdas_(1031)
    , types_(1031)
    , gid_counter_(0)
    , sigma0_ (consume(new Sigma(*this, ArrayRef<const Generic*>(), ArrayRef<const Type*>()))->as<Sigma>())
    , pi0_    (consume(new Pi   (*this, ArrayRef<const Generic*>(), ArrayRef<const Type*>()))->as<Pi>())
    , mem_    (consume(new Mem  (*this))->as<Mem>())
    , frame_  (consume(new Frame(*this))->as<Frame>())
#define ANYDSL2_UF_TYPE(T) ,T##_(consume(new PrimType(*this, PrimType_##T))->as<PrimType>())
#include "anydsl2/tables/primtypetable.h"
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

Sigma* World::named_sigma(size_t num_elems, size_t num_generics, const std::string& name) {
    Sigma* s = new Sigma(*this, num_elems, num_generics);
    s->debug = name;

    assert(types_.find(s) == types_.end() && "must not be inside");
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
#define ANYDSL2_U_TYPE(T) case PrimType_##T: return literal(T(value));
#define ANYDSL2_F_TYPE(T) ANYDSL2_U_TYPE(T)
#include "anydsl2/tables/primtypetable.h"
        default: ANYDSL2_UNREACHABLE;
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

    assert(is_relop(kind) && "must be a RelOp");
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
#define ANYDSL2_UF_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                }
            case ArithOp_sub:
                switch (type) {
#define ANYDSL2_UF_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                }
            case ArithOp_mul:
                switch (type) {
#define ANYDSL2_UF_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                }
            case ArithOp_udiv:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) \
                    case PrimType_##T: \
                        return rlit->is_zero() \
                             ? (const Def*) bottom(rtype) \
                             : (const Def*) literal(type, Box(T(l.get_##T() / r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_sdiv:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed<T>::type S; \
                        return literal(type, Box(bcast<T , S>(bcast<S, T >(l.get_##T()) / bcast<S, T >(r.get_##T())))); \
                    }
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case ArithOp_fadd:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() + r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case ArithOp_fsub:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() - r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case ArithOp_fmul:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() * r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case ArithOp_fdiv:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(T(l.get_##T() / r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case ArithOp_frem:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal(type, Box(std::fmod(l.get_##T(), r.get_##T())));
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            default: 
                ANYDSL2_UNREACHABLE;
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
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() == r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_cmp_ne:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() != r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_cmp_ult:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <  r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_cmp_ule:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <= r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_cmp_slt:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed< T >::type S; \
                        return literal_u1(bcast<S, T>(l.get_##T()) <  bcast<S, T>(r.get_##T())); \
                    }
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_cmp_sle:
                switch (type) {
#define ANYDSL2_JUST_U_TYPE(T) \
                    case PrimType_##T: { \
                        typedef make_signed< T >::type S; \
                        return literal_u1(bcast<S, T>(l.get_##T()) <= bcast<S, T>(r.get_##T())); \
                    }
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_F_TYPE;
                }
            case RelOp_fcmp_oeq:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() == r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case RelOp_fcmp_one:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() != r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case RelOp_fcmp_olt:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <  r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            case RelOp_fcmp_ole:
                switch (type) {
#define ANYDSL2_JUST_F_TYPE(T) case PrimType_##T: return literal_u1(l.get_##T() <= r.get_##T());
#include "anydsl2/tables/primtypetable.h"
                    ANYDSL2_NO_U_TYPE;
                }
            default: 
                ANYDSL2_UNREACHABLE;
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

const Def* World::extract(const Def* agg, const Def* index) {
    if (agg->isa<Bottom>())
        return bottom(agg->type()->as<Sigma>()->elem_via_lit(index));

    if (const Tuple* tuple = agg->isa<Tuple>())
        return tuple->op_via_lit(index);

    if (const Insert* insert = agg->isa<Insert>()) {
        if (index == insert->index())
            return insert->value();
        else
            return extract(insert->tuple(), index);
    }

    return consume(new Extract(agg, index));
}

const Def* World::insert(const Def* agg, const Def* index, const Def* value) {
    if (agg->isa<Bottom>() || value->isa<Bottom>())
        return bottom(agg->type());

    if (const Tuple* tup = agg->isa<Tuple>()) {
        Array<const Def*> args(tup->size());
        std::copy(agg->ops().begin(), agg->ops().end(), args.begin());
        args[index->primlit_value<size_t>()] = value;

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

const Def* World::primop(int kind, const Type* type, ArrayRef<const Def*> ops) {
    if (is_arithop(kind)) { assert(ops.size() == 2); return arithop((ArithOpKind) kind, ops[0], ops[1]); }
    if (is_relop  (kind)) { assert(ops.size() == 2); return relop(  (RelOpKind  ) kind, ops[0], ops[1]); }
    if (is_convop (kind)) { assert(ops.size() == 1); return convop( (ConvOpKind ) kind, type,   ops[0]); }

    switch (kind) {
        case Node_Enter:   assert(ops.size() == 1); return enter(  ops[0]);
        case Node_Extract: assert(ops.size() == 2); return extract(ops[0], ops[1]);
        case Node_Insert:  assert(ops.size() == 3); return insert( ops[0], ops[1], ops[2]);
        case Node_Leave:   assert(ops.size() == 2); return leave(  ops[0], ops[1]);
        case Node_Load:    assert(ops.size() == 2); return load(   ops[0], ops[1]);
        case Node_Select:  assert(ops.size() == 3); return select( ops[0], ops[1], ops[2]);
        case Node_Slot:    assert(ops.size() == 1); return slot(   ops[0]->as<Enter>(), type);
        case Node_Store:   assert(ops.size() == 3); return store(  ops[0], ops[1], ops[2]);
        case Node_Tuple:                            return tuple(  ops);
        default: ANYDSL2_UNREACHABLE;
    }
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
        for_all (peek, param->peek()) {
            dce_insert(peek.def());
            dce_insert(peek.from());
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
    assert(types_.find(type) != types_.end() && "not in map");

    if (type->is_marked()) return;
    type->mark();

    for_all (arg, type->args())
        ute_insert(arg);
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
    assert(lambdas_.find(lambda) != lambdas_.end() && "not in map");

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

    Lambda* helper = 0;
    Lambda* fac = 0;
    Lambda* ifelse = 0;
    for_all (lambda, lambdas()) {
        if (lambda->debug == "helper")
            helper = lambda;
        else if (lambda->debug == "fac")
            fac = lambda;
        else if (lambda->debug == "<if-else-01>")
            ifelse = lambda;
    }

    Lambda* dropped = helper->drop(3, fac->param(1), true);
    ifelse->unset_op(4);
    ifelse->shrink(4);
    ifelse->update(0, dropped);

    cleanup();
}

const PrimOp* World::consume(const PrimOp* primop) {
    PrimOpSet::iterator i = primops_.find(primop);
    if (i != primops_.end()) {
        delete primop;
        assert(primops_.find(*i) != primops_.end() && "hash/equal function of primop class incorrect");
        return *i;
    }

    primops_.insert(primop);
    assert(primops_.find(primop) != primops_.end() && "hash/equal function of def class incorrect");

    return primop;
}

const Type* World::consume(const Type* type) {
    TypeSet::iterator i = types_.find(type);
    if (i != types_.end()) {
        delete type;
        assert(types_.find(*i) != types_.end() && "hash/equal function of type class incorrect");
        return *i;
    }

    types_.insert(type);
    assert(types_.find(type) != types_.end() && "hash/equal function of def class incorrect");

    return type;
}

PrimOp* World::release(const PrimOp* primop) {
    PrimOpSet::iterator i = primops_.find(primop);
    assert(i != primops_.end() && "must be found");
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
