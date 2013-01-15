#include "anydsl2/lambda.h"

#include <algorithm>

#include "anydsl2/type.h"
#include "anydsl2/literal.h"
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

template<bool direct>
static Lambdas find_preds(const Lambda* lambda) {
    Lambdas result;

    for_all (use, lambda->uses()) {
        const Def* udef = use.def();
        if (const Select* select = udef->isa<Select>()) {
            for_all (use, select->uses()) {
                assert(use.index() == 0);
                result.push_back(use.def()->as_lambda());
            }
        } else {
            if (!direct || use.index() == 0)
                result.push_back(udef->as_lambda());
        }
    }

    return result;
}

Lambdas Lambda::preds() const { return find_preds<false>(this); }
Lambdas Lambda::direct_preds() const { return find_preds<true>(this); }

Lambdas Lambda::direct_succs() const {
    Lambdas result;
    result.reserve(2);
    if (Lambda* succ = to()->isa_lambda()) {
        result.push_back(succ);
        return result;
    } else if (to()->isa<Param>())
        return result;

    const Select* select = to()->as<Select>();
    result.resize(2);
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

/*
 * terminate
 */

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

/*
 * CPS construction
 */

const Def* Lambda::get_value(size_t handle, const Type* type, const char* name) {
    if (const Def* def = defs_.find(handle))
        return def;

    if (parent())
        return parent()->get_value(handle, type, name);

    if (/*is function*/ false) {
        // value is undefined
        // TODO provide hook instead of fixed functionality
        std::cerr << "'" << name << "'" << " may be undefined" << std::endl;
        return set_value(handle, world().bottom(type));
    }

#if 0
    // insert a 'phi', i.e., create a param and remember to fix the callers
    if (!sealed_ || preds_.size() > 1) {
        const Param* param = top_->append_param(type, name);
        size_t index = in_.size();
        in_.push_back(param);
        set_value(handle, param);

        Todo todo(handle, index, type, name);
        if (sealed_)
            fix(todo);
        else
            todos_.push_back(todo);

        return param;
    }

    // unreachable code
    if (preds().empty())
        return set_value(handle, world().bottom(type));
    
    assert(preds().size() == 1 && "there can only be one");
#endif
    Lambda* pred = *preds().begin();
    const Def* def = pred->get_value(handle, type, name);

    // create copy of lvar in this Lambda
    return set_value(handle, def);
}

void Lambda::seal() {
    assert(!sealed() && "already sealed");
    sealed_ = true;

#if 0
    if (preds().size() >= 2) {
        for_all (pred, preds_)
            assert(pred->succs().size() <= 1 && "critical edge");
    }
#endif

    for_all (todo, todos_)
        fix(todo);
}

void Lambda::fix(Todo todo) {
    assert(sealed() && "must be sealed");

    size_t handle = todo.handle();
    size_t index = todo.index();
    const Type* type = todo.type();
    const char* name = todo.name();
    const Param* p = param(index);
    const Def* same = 0;

    // find Horspool-like phis
    for_all (pred, preds()) {
        const Def* def = pred->get_value(handle, type, name);

        if (def->isa<Undef>() || def == p || same == def)
            continue;

        if (same) {
            same = 0;
            goto fix_preds;
        }
        same = def;
    }
    
goto fix_preds; // HACK fix cond_
    if (!same || same == p)
        same = world().bottom(p->type());

    for_all (use, p->uses())
        world().update(use.def(), use.index(), same);

fix_preds:
#if 0
    for_all (pred, preds()) {
        assert(pred->succs().size() == 1 && "critical edge");
        Out& out = pred->out_;

        // make potentially room for the new arg
        if (index >= out.size())
            out.resize(index + 1);

        assert(!pred->out_[index] && "already set");
        out[index] = same ? same : pred->get_value(handle, type, name);
    }
#endif

    if (same)
        set_value(handle, same);
}


} // namespace anydsl2
