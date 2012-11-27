#include "anydsl2/world.h"

#include <cmath>
#include <algorithm>
#include <queue>
#include <iostream>
#include <boost/unordered_map.hpp>

#include "anydsl2/def.h"
#include "anydsl2/primop.h"
#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/memop.h"
#include "anydsl2/type.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/rootlambdas.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/transform/cfg_builder.h"
#include "anydsl2/transform/merge_lambdas.h"
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
    , types_(1031)
    , gid_counter_(0)
    , pass_counter_(1)
    , sigma0_ (consume(new Sigma(*this, ArrayRef<const Type*>()))->as<Sigma>())
    , pi0_    (consume(new Pi   (*this, ArrayRef<const Type*>()))->as<Pi>())
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


template<class T>
inline const Def* World::find(const T& tuple) {
    PrimOpSet::iterator i = primops_.find(tuple, 
            std::ptr_fun<const T&, size_t>(hash_def),
            std::ptr_fun<const T&, const Def*, bool>(equal_def));
    return i == primops_.end() ? 0 : *i;
}

template<class T>
inline const T* World::new_consume(const T* def) {
    std::pair<PrimOpSet::iterator, bool> p = primops_.insert(def);
    assert(p.second && "hash/equal broken");
    return (*p.first)->as<T>();
}

/*
 * types
 */

Sigma* World::named_sigma(size_t size, const std::string& name) {
    Sigma* s = new Sigma(*this, size);
    s->name = name;

    assert(types_.find(s) == types_.end() && "must not be inside");
    types_.insert(s);

    return s;
}

/*
 * literals
 */

const PrimLit* World::literal(PrimTypeKind kind, Box box) {
    const Type* ptype = type(kind);
    if (const Def* def = find(PrimLitTuple(Node_PrimLit, ptype, box)))
        return def->as<PrimLit>();
    return new_consume(new PrimLit(ptype, box));
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
    if (const Def* def = find(DefTuple0(Node_Any, type)))
        return def->as<Any>();
    return new_consume(new Any(type));
}

const Bottom* World::bottom(const Type* type) {
    if (const Def* def = find(DefTuple0(Node_Bottom, type)))
        return def->as<Bottom>();
    return new_consume(new Bottom(type));
}

/*
 * create
 */

const Def* World::binop(int kind, const Def* lhs, const Def* rhs, const std::string& name) {
    if (is_arithop(kind))
        return arithop((ArithOpKind) kind, lhs, rhs);

    assert(is_relop(kind) && "must be a RelOp");
    return relop((RelOpKind) kind, lhs, rhs);
}

const Def* World::tuple(ArrayRef<const Def*> args, const std::string& name) {
    Array<const Type*> elems(args.size());

    bool bot = false;
    for_all2 (&elem, elems, arg, args) {
        elem = arg->type();

        if (arg->isa<Bottom>())
            bot = true;
    }

    if (bot)
        return bottom(sigma(elems));

    return consume(new Tuple(*this, args, name));
}

const Def* World::arithop(ArithOpKind kind, const Def* a, const Def* b, const std::string& name) {
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

    if (const Def* def = find(DefTuple2(kind, a->type(), a, b)))
        return def;

    return new_consume(new ArithOp(kind, a, b, name));
}

const Def* World::relop(RelOpKind kind, const Def* a, const Def* b, const std::string& name) {
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

    return consume(new RelOp(kind, a, b, name));
}

const Def* World::convop(ConvOpKind kind, const Type* to, const Def* from, const std::string& name) {
    if (from->isa<Bottom>())
        return bottom(to);

#if 0
    if (const PrimLit* lit = from->isa<PrimLit>())
        Box box = lit->box();
        PrimTypeKind type = lit->primtype_kind();

        // TODO folding
    }
#endif

    return consume(new ConvOp(kind, to, from, name));
}

const Def* World::extract(const Def* agg, const Def* index, const std::string& name) {
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

    return consume(new Extract(agg, index, name));
}

const Def* World::insert(const Def* agg, const Def* index, const Def* value, const std::string& name) {
    if (agg->isa<Bottom>() || value->isa<Bottom>())
        return bottom(agg->type());

    if (const Tuple* tup = agg->isa<Tuple>()) {
        Array<const Def*> args(tup->size());
        std::copy(agg->ops().begin(), agg->ops().end(), args.begin());
        args[index->primlit_value<size_t>()] = value;

        return tuple(args);
    }

    return consume(new Insert(agg, index, value, name));
}

const Def* World::load(const Def* mem, const Def* ptr, const std::string& name) {
    return consume(new Load(mem, ptr, name));
}

const Def* World::store(const Def* mem, const Def* ptr, const Def* val, const std::string& name) {
    return consume(new Store(mem, ptr, val, name));
}

const Enter* World::enter(const Def* mem, const std::string& name) {
    return consume(new Enter(mem, name))->as<Enter>();
}

const Leave* World::leave(const Def* mem, const Def* frame, const std::string& name) {
    return consume(new Leave(mem, frame, name))->as<Leave>();
}

const Slot* World::slot(const Enter* enter, const Type* type, const std::string& name) {
    return consume(new Slot(enter, type, name))->as<Slot>();
}

const CCall* World::ccall(const Def* mem, const std::string& callee, 
                          ArrayRef<const Def*> args, const Type* rettype, bool vararg, const std::string& name) {
    return consume(new CCall(mem, callee, args, rettype, vararg, name))->as<CCall>();
}

const Def* World::select(const Def* cond, const Def* a, const Def* b, const std::string& name) {
    if (cond->isa<Bottom>() || a->isa<Bottom>() || b->isa<Bottom>())
        return bottom(a->type());

    if (const PrimLit* lit = cond->isa<PrimLit>())
        return lit->box().get_u1().get() ? a : b;

    return consume(new Select(cond, a, b, name));
}

const Def* World::typekeeper(const Type* type, const std::string& name) { 
    return consume(new TypeKeeper(type, name)); 
}

Lambda* World::lambda(const Pi* pi, LambdaAttr attr, const std::string& name) {
    Lambda* l = new Lambda(gid_counter_++, pi, attr, name);
    lambdas_.insert(l);

    size_t i = 0;
    for_all (elem, pi->elems())
        l->params_.push_back(new Param(elem, l, i++, ""));

    return l;
}

const Def* World::primop(const PrimOp* in, ArrayRef<const Def*> ops) { return primop(in, ops, in->name); }

const Def* World::primop(const PrimOp* in, ArrayRef<const Def*> ops, const std::string& name) {
    int kind = in->kind();
    const Type* type = in->type();
    if (is_arithop(kind)) { assert(ops.size() == 2); return arithop((ArithOpKind) kind, ops[0], ops[1], name); }
    if (is_relop  (kind)) { assert(ops.size() == 2); return relop(  (RelOpKind  ) kind, ops[0], ops[1], name); }
    if (is_convop (kind)) { assert(ops.size() == 1); return convop( (ConvOpKind ) kind, type,   ops[0], name); }

    switch (kind) {
        case Node_Enter:   assert(ops.size() == 1); return enter(  ops[0], name);
        case Node_Extract: assert(ops.size() == 2); return extract(ops[0], ops[1], name);
        case Node_Insert:  assert(ops.size() == 3); return insert( ops[0], ops[1], ops[2], name);
        case Node_Leave:   assert(ops.size() == 2); return leave(  ops[0], ops[1], name);
        case Node_Load:    assert(ops.size() == 2); return load(   ops[0], ops[1], name);
        case Node_Select:  assert(ops.size() == 3); return select( ops[0], ops[1], ops[2], name);
        case Node_Slot:    assert(ops.size() == 1); return slot(   ops[0]->as<Enter>(), type, name);
        case Node_Store:   assert(ops.size() == 3); return store(  ops[0], ops[1], ops[2], name);
        case Node_Tuple:                            return tuple(  ops, name);
        case Node_Bottom:  assert(ops.empty());     return bottom(type);
        case Node_Any:     assert(ops.empty());     return any(type);
        case Node_PrimLit: assert(ops.empty());     return literal((PrimTypeKind) kind, in->as<PrimLit>()->box());
        default: ANYDSL2_UNREACHABLE;
    }
}

/*
 * optimizations
 */

void World::dead_code_elimination() {
    size_t pass = new_pass();

    for_all (primop, primops()) {
        if (const TypeKeeper* tk = primop->isa<TypeKeeper>())
            dce_insert(pass, tk);
    }

    for_all (lambda, lambdas()) {
        if (lambda->attr().is_extern()) {
            for_all (param, lambda->ho_params()) {
                for_all (use, param->uses())
                    dce_insert(pass, use.def());
            }
        }
    }

    for (PrimOpSet::iterator i = primops_.begin(); i != primops_.end();) {
        const PrimOp* primop = *i;
        if (primop->is_visited(pass))
            ++i;
        else {
            delete primop;
            i = primops_.erase(i);
        }
    }

    for (LambdaSet::iterator i = lambdas_.begin(); i != lambdas_.end();) {
        LambdaSet::iterator j = i++;
        Lambda* lambda = *j;
        if (!lambda->is_visited(pass)) {
            delete lambda;
            lambdas_.erase(j);
        }
    }
}

void World::dce_insert(size_t pass, const Def* def) {
#ifndef NDEBUG
    if (const PrimOp* primop = def->isa<PrimOp>()) assert(primops_.find(primop)          != primops_.end());
    if (      Lambda* lambda = def->isa_lambda() ) assert(lambdas_.find(lambda)          != lambdas_.end());
    if (const Param*  param  = def->isa<Param>() ) assert(lambdas_.find(param->lambda()) != lambdas_.end());
#endif

    if (def->visit(pass)) return;

    for_all (op, def->ops())
        dce_insert(pass, op);

    if (Lambda* lambda = def->isa_lambda()) {
        // insert control-dependent lambdas
        for_all (pred, lambda->preds())
            dce_insert(pass, pred);
    } else if (const Param* param = def->isa<Param>()) {
        for_all (peek, param->peek()) {
            dce_insert(pass, peek.def());
            dce_insert(pass, peek.from());
        }

        // always consider all params in the same lambda as live
        for_all (other, param->lambda()->params())
            dce_insert(pass, other);
    }
}

void World::unused_type_elimination() {
    size_t pass = new_pass();

    for_all (primop, primops())
        ute_insert(pass, primop->type());

    for_all (lambda, lambdas()) {
        ute_insert(pass, lambda->type());
        for_all (param, lambda->params())
            ute_insert(pass, param->type());
    }

    for (TypeSet::iterator i = types_.begin(); i != types_.end();) {
        const Type* type = *i;

        if (type->is_visited(pass))
            ++i;
        else {
            delete type;
            i = types_.erase(i);
        }
    }
}

void World::ute_insert(size_t pass, const Type* type) {
    assert(types_.find(type) != types_.end() && "not in map");

    if (type->visit(pass)) return;

    for_all (elem, type->elems())
        ute_insert(pass, elem);
}

void World::unreachable_code_elimination() {
    size_t pass = new_pass();

    for_all (lambda, lambdas())
        if (lambda->attr().is_extern())
            uce_insert(pass, lambda);

    for (LambdaSet::iterator i = lambdas_.begin(); i != lambdas_.end();) {
        LambdaSet::iterator j = i++;
        Lambda* lambda = *j;
        if (!lambda->is_visited(pass)) {
            delete lambda;
            lambdas_.erase(j);
        }
    }
}

void World::uce_insert(size_t pass, Lambda* lambda) {
    assert(lambdas_.find(lambda) != lambdas_.end() && "not in map");

    if (lambda->visit(pass)) return;

    for_all (succ, lambda->succs())
        uce_insert(pass, succ);
}

void World::cleanup() {
    unreachable_code_elimination();
    dead_code_elimination();
    unused_type_elimination();
}

void World::opt() {
    cfg_transform(*this);
    merge_lambdas(*this);
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
    if (fancy) {
        LambdaSet roots = find_root_lambdas(lambdas());

        for_all (root, roots) {
            Scope scope(root);
            for_all (lambda, scope.rpo())
                lambda->dump(fancy, scope.domtree().depth(lambda));
        }
    } else {
        for_all (lambda, lambdas())
            lambda->dump(fancy);
    }
    std::cout << std::endl;
}

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

void World::replace(const Def* cwhat, const Def* with) {
    Def* what;
    if (const PrimOp* primop = cwhat->isa<PrimOp>())
        what = release(primop);
    else {
        Lambda* lambda = cwhat->as_lambda();
        what = lambda;
        lambdas_.erase(lambda);
    }

    replace(what, with);
    delete what;
}

void World::replace(Def* what, const Def* with) {
    assert(!what->isa<Param>()  || primops_.find(what->as<PrimOp>()) == primops_.end());
    assert(!with->isa<PrimOp>() || primops_.find(with->as<PrimOp>()) != primops_.end());
    assert(what != with);

    // unregister all uses of what's operands
    for (size_t i = 0, e = what->size(); i != e; ++i)
        what->unset_op(i);

    for_all (use, what->copy_uses()) {
        if (Lambda* lambda = use.def()->isa_lambda())
            lambda->update(use.index(), with);
        else {
            PrimOp* uprimop = release(use.def()->as<PrimOp>());
            uprimop->update(use.index(), with);

            PrimOpSet::iterator i = primops_.find(uprimop);
            if (i != primops_.end())
                replace(uprimop, *i);
            else {
                PrimOpSet::iterator i = primops_.insert(uprimop).first;
                const Def* new_def = primop(uprimop, uprimop->ops());
                if (uprimop != new_def) {
                    primops_.erase(i);
                    replace(uprimop, new_def);
                }
            }
        }
    }
}

} // namespace anydsl2
