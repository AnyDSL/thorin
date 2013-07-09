#include "anydsl2/lambda.h"

#include "anydsl2/literal.h"
#include "anydsl2/symbol.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/printer.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

Lambda::Lambda(size_t gid, const Pi* pi, LambdaAttr attr, bool is_sealed, const std::string& name)
    : Def(gid, Node_Lambda, 0, pi, true, name)
    , sid_(size_t(-1))
    , backwards_sid_(size_t(-1))
    , scope_(0)
    , attr_(attr)
    , parent_(this)
    , is_sealed_(is_sealed)
    , is_visited_(false)
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

Lambda* Lambda::update_op(size_t i, const Def* def) {
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
    const Param* param = world().param(type, this, size, name);
    params_.push_back(param);

    return param;
}

const Param* Lambda::mem_param() const {
    for_all (param, params())
        if (param->type()->isa<Mem>())
            return param;

    return 0;
}

const Def* Lambda::append_arg(const Def* arg) {
    ops_.push_back(arg);
    return arg;
}

template<bool direct>
static Lambdas find_preds(const Lambda* lambda) {
    Lambdas result;
    for_all (use, lambda->uses()) {
        if (const Select* select = use->isa<Select>()) {
            for_all (select_user, select->uses()) {
                assert(select_user.index() == 0);
                result.push_back(select_user->as_lambda());
            }
        } else {
            if (!direct || use.index() == 0)
                result.push_back(use->as_lambda());
        }
    }

    return result;
}

Lambdas Lambda::preds() const { return find_preds<false>(this); }
Lambdas Lambda::direct_preds() const { return find_preds<true>(this); }

Lambdas Lambda::direct_succs() const {
    Lambdas result;
    if (empty())
        return result;

    result.reserve(2);
    if (Lambda* succ = to()->isa_lambda()) {
        result.push_back(succ);
        return result;
    } else if (to()->isa<Param>() || to()->isa<Undef>())
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
    return use->isa<Lambda>() && use.index() > 0;
}

bool Lambda::is_passed() const {
    for_all (use, this->uses()) {
        if (use->isa_lambda()) {
            if (use.index() != 0)
                return true;
        }
    }

    return false;
}

bool Lambda::is_basicblock() const { return pi()->is_basicblock(); }
bool Lambda::is_returning() const { return pi()->is_returning(); }
void Lambda::dump_jump() const { Printer p(std::cout, false); print_jump(p); }
void Lambda::dump_head() const { Printer p(std::cout, false); print_head(p); }

/*
 * terminate
 */

void Lambda::jump(const Def* to, ArrayRef<const Def*> args) {
    unset_ops();
    resize(args.size()+1);
    set_op(0, to);

    size_t x = 1;
    for_all (arg, args)
        set_op(x++, arg);
}

void Lambda::branch(const Def* cond, const Def* tto, const Def*  fto) {
    return jump(world().select(cond, tto, fto), ArrayRef<const Def*>(0, 0));
}

Lambda* Lambda::call(const Def* to, ArrayRef<const Def*> args, const Type* ret_type) {
    // create next continuation in cascade
    Lambda* next = world().lambda(world().pi1(ret_type), name + "_" + to->name);
    const Param* result = next->param(0);
    result->name = to->name;

    // create jump to this new continuation
    size_t csize = args.size() + 1;
    Array<const Def*> cargs(csize);
    *std::copy(args.begin(), args.end(), cargs.begin()) = next;
    jump(to, cargs);

    return next;
}

Lambda* Lambda::mem_call(const Def* to, ArrayRef<const Def*> args, const Type* ret_type) {
    // create next continuation in cascade
    const Pi* pi = ret_type ? world().pi2(world().mem(), ret_type) : world().pi1(world().mem());
    Lambda* next = world().lambda(pi, name + "_" + to->name);
    next->param(0)->name = "mem";

    if (ret_type)
        next->param(1)->name = to->name;

    // create jump to this new continuation
    size_t csize = args.size() + 1;
    Array<const Def*> cargs(csize);
    *std::copy(args.begin(), args.end(), cargs.begin()) = next;
    jump(to, cargs);

    return next;
}

/*
 * CPS construction
 */

void Lambda::clear() { 
    for_all (tracker, tracked_values_)
        delete tracker;
    tracked_values_.clear(); 
}

const Tracker* Lambda::find_tracker(size_t handle) {
    if (handle >= tracked_values_.size())
        tracked_values_.resize(handle+1);
    return tracked_values_[handle];
}

const Def* Lambda::set_value(size_t handle, const Def* def) { 
    if (const Tracker* tracker = find_tracker(handle))
        delete tracker;

    return (tracked_values_[handle] = new Tracker(def))->def(); 
}

const Def* Lambda::get_value(size_t handle, const Type* type, const char* name) {
    if (const Tracker* tracker = find_tracker(handle))
        return tracker->def();

    if (parent() != this) { // is a function head?
        if (parent())
            return parent()->get_value(handle, type, name);
        goto return_bottom;
    } else {
        Lambdas preds = this->preds();

        if (!is_sealed_) {
            const Param* param = append_param(type, name);
            todos_.push_back(Todo(handle, param->index(), type, name));
            return set_value(handle, param);
        } else {
            switch (preds.size()) {
                case 0: goto return_bottom;
                case 1: return set_value(handle, preds.front()->get_value(handle, type, name));
                default: {
                    if (!is_visited_) {
                        is_visited_ = true;
                        const Def* same = 0;
                        for_all (pred, preds) {
                            const Def* def = pred->get_value(handle, type, name);
                            if (same && same != def) {
                                same = (const Def*)-1; // defs from preds are different
                                break;
                            }
                            same = def;
                        }

                        is_visited_ = false;
                        if (same != (const Def*)-1)
                            return same;
                    }

                    const Param* param = append_param(type, name);
                    set_value(handle, param); // break cycle by first letting handle point to param
                    return set_value(handle, fix(Todo(handle, param->index(), type, name)));
                }
            }
        }
    }

return_bottom:
    // TODO provide hook instead of fixed functionality
    std::cerr << "'" << name << "'" << " may be undefined" << std::endl;
    return set_value(handle, world().bottom(type));
}

void Lambda::seal() {
    assert(!is_sealed() && "already sealed");
    is_sealed_ = true;

    for_all (todo, todos_)
        fix(todo);
    todos_.clear();
}

const Def* Lambda::fix(const Todo& todo) {
    assert(is_sealed() && "must be sealed");

    size_t index = todo.index();
    const Param* param = this->param(index);
    assert(todo.index() == param->index());

    for_all (pred, preds()) {
        assert(!pred->empty());
        assert(pred->succs().size() == 1 && "critical edge");

        // make potentially room for the new arg
        if (index >= pred->num_args())
            pred->resize(index+2);

        assert(!pred->arg(index) && "already set");
        pred->set_op(index + 1, pred->get_value(todo));
    }

    return try_remove_trivial_param(param);
}

const Def* Lambda::try_remove_trivial_param(const Param* param) {
    assert(param->lambda() == this);
    assert(is_sealed() && "must be sealed");

    Lambdas preds = this->preds();
    size_t index = param->index();

    // find Horspool-like phis
    const Def* same = 0;
    for_all (pred, preds) {
        const Def* def = pred->arg(index);
        if (def == param || same == def)
            continue;
        if (same)
            return param;
        same = def;
    }

    same = same ? same : world().bottom(param->type());
    AutoVector<const Tracker*> uses = param->tracked_uses();
    param->replace(same);

    for_all (peek, param->peek())
        peek.from()->update_arg(index, world().bottom(param->type()));

    for_all (tracker, uses) {
        if (Lambda* lambda = tracker->def()->isa_lambda()) {
            for_all (succ, lambda->succs())
                if (param != succ->param(index))
                    succ->try_remove_trivial_param(succ->param(index));
        }
    }

    return same;
}

} // namespace anydsl2
