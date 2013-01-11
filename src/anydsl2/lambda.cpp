#include "anydsl2/lambda.h"

#include <algorithm>

#include "anydsl2/type.h"
#include "anydsl2/primop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

Lambda::Lambda(size_t gid, const Pi* pi, LambdaAttr attr, const std::string& name)
    : Def(Node_Lambda, pi, name)
    , gid_(gid)
    , sid_(size_t(-1))
    , scope_(0)
    , attr_(attr)
{
    params_.reserve(pi->size());
}

Lambda::~Lambda() {
    for_all (param, params())
        delete param;
}

Lambda* Lambda::stub(const GenericMap& generic_map, const std::string& name) const { 
    Lambda* result = world().lambda(pi()->specialize(generic_map)->as<Pi>(), attr(), name);

    for (size_t i = 0, e = params().size(); i != e; ++i)
        result->param(i)->name = param(i)->name;

    return result;
}

Lambda* Lambda::update(size_t i, const Def* def) {
    unset_op(i);
    set_op(i, def);
    return this;
}

const Pi* Lambda::pi() const { return type()->as<Pi>(); }
const Pi* Lambda::to_pi() const { return to()->type()->as<Pi>(); }

const Pi* Lambda::arg_pi() const {
    Array<const Type*> elems(num_args());
    for_all2 (&elem, elems, arg, args())
        elem = arg->type();

    return world().pi(elems);
}

const Param* Lambda::append_param(const Type* type, const std::string& name) {
    size_t size = pi()->size();

    Array<const Type*> elems(size + 1);
    *std::copy(pi()->elems().begin(), pi()->elems().end(), elems.begin()) = type;

    // update type
    set_type(world().pi(elems));

    // append new param
    const Param* param = new Param(type, this, size, name);
    params_.push_back(param);

    return param;
}

static void find_lambdas(const Def* def, LambdaSet& result) {
    if (Lambda* lambda = def->isa_lambda()) {
        result.insert(lambda);
        return;
    }

    for_all (op, def->ops())
        find_lambdas(op, result);
}

template<bool direct>
inline static void find_preds(Use use, LambdaSet& result) {
    const Def* def = use.def();
    if (Lambda* lambda = def->isa_lambda()) {
        if (!direct || use.index() == 0)
            result.insert(lambda);
    } else {
        assert(def->isa<PrimOp>() && "not a PrimOp");

        for_all (use, def->uses())
            find_preds<direct>(use, result);
    }
}

LambdaSet Lambda::preds() const {
    LambdaSet result;

    for_all (use, uses())
        find_preds<false>(use, result);

    return result;
}

LambdaSet Lambda::direct_preds() const {
    LambdaSet result;

    for_all (use, uses())
        find_preds<true>(use, result);

    return result;
}

Array<Lambda*> Lambda::direct_succs() const {
    if (Lambda* succ = to()->isa_lambda()) {
        Array<Lambda*> result(1);
        result[0] = succ;
        return result;
    } else if (to()->isa<Param>())
        return Array<Lambda*>(0);

    const Select* select = to()->as<Select>();
    Array<Lambda*> result(2);
    result[0] = select->tval()->as_lambda();
    result[1] = select->fval()->as_lambda();
    return result;
}

Lambdas Lambda::succs() const {
    Lambdas result;

    for_all (succ, direct_succs())
        result.push_back(succ);

    for_all (arg, args())
        if (Lambda* succ = arg->isa_lambda())
            result.push_back(succ);

    return result;
}

bool Lambda::is_cascading() const {
    if (uses().size() != 1)
        return false;

    Use use = *uses().begin();
    return use.def()->isa<Lambda>() && use.index() > 0;
}

bool Lambda::is_bb() const { return order() == 1; }

bool Lambda::is_returning() const {
    bool ret = false;
    for_all (param, params()) {
        switch (param->order()) {
            case 0: continue;
            case 1: 
                if (!ret) {
                    ret = true;
                    continue;
                }
            default:
                return false;
        }
    }
    return true;
}

void Lambda::jump(const Def* to, ArrayRef<const Def*> args) {
    if (valid()) {
        for (size_t i = 0, e = size(); i != e; ++i)
            unset_op(i);
        realloc(args.size() + 1);
    } else
        alloc(args.size() + 1);

    set_op(0, to);

    size_t x = 1;
    for_all (arg, args)
        set_op(x++, arg);
}

void Lambda::branch(const Def* cond, const Def* tto, const Def*  fto) {
    return jump(world().select(cond, tto, fto), ArrayRef<const Def*>(0, 0));
}

} // namespace anydsl2
