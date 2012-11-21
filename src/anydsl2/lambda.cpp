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

bool Lambda::equal(const Node* other) const { return this == other; }
size_t Lambda::hash() const { return boost::hash_value(this); }

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

LambdaSet Lambda::targets() const {
    LambdaSet result;
    find_lambdas(to(), result);

    return result;
}

LambdaSet Lambda::hos() const {
    LambdaSet result;
    for_all (def, args())
        find_lambdas(def, result);

    return result;
}

LambdaSet Lambda::succs() const {
    LambdaSet result;
    for_all (def, ops())
        find_lambdas(def, result);

    return result;
}

template<bool fo>
Array<const Param*> Lambda::classify_params() const {
    Array<const Param*> res(params().size());

    size_t size = 0;
    for_all (param, params())
        if (fo ^ (param->type()->isa<Pi>() != 0))
            res[size++] = param;

    res.shrink(size);

    return res;
}

// TODO buggy
template<bool fo>
Array<const Def*> Lambda::classify_args() const {
    Array<const Def*> res(args().size());

    size_t size = 0;
    for_all (arg, args())
        if (fo ^ (arg->type()->isa<Pi>() != 0))
            res[size++] = arg;

    res.shrink(size);

    return res;
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
        switch (param->type()->order()) {
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

Array<const Param*> Lambda::fo_params() const { return classify_params<true>(); }
Array<const Param*> Lambda::ho_params() const { return classify_params<false>(); }
Array<const Def*> Lambda::fo_args() const { return classify_args<true>(); }
Array<const Def*> Lambda::ho_args() const { return classify_args<false>(); }

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

Lambda* Lambda::drop(ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, bool self, const GenericMap& generic_map) {
    return mangle(to_drop, drop_with, Array<const Def*>(), self, generic_map);
}

Lambda* Lambda::lift(ArrayRef<const Def*> to_lift, bool self, const GenericMap& generic_map) {
    return mangle(Array<size_t>(), Array<const Def*>(), to_lift, self, generic_map);
}

Lambda* Lambda::mangle(ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, 
                       ArrayRef<const Def*> to_lift, bool self, const GenericMap& generic_map) {
    return Scope(this).mangle(to_drop, drop_with, to_lift, self, generic_map);
}

} // namespace anydsl2
