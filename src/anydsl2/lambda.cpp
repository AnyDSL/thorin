#include "anydsl2/lambda.h"

#include "anydsl2/literal.h"
#include "anydsl2/symbol.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/printer.h"
#include "anydsl2/util/array.h"

namespace anydsl2 {

Lambda::Lambda(size_t gid, const Pi* pi, LambdaAttr attr, bool is_sealed, const std::string& name)
    : Def(gid, Node_Lambda, 0, pi, true, name)
    , sid_(size_t(-1))
    , backwards_sid_(size_t(-1))
    , scope_(nullptr)
    , attr_(attr)
    , parent_(this)
    , is_sealed_(is_sealed)
    , is_visited_(false)
{
    params_.reserve(pi->size());
}

Lambda::~Lambda() {
    for (auto param : params())
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
    for (size_t i = 0, e = num_args(); i != e; ++i)
        elems[i] = arg(i)->type();

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

template<bool direct>
static Lambdas find_preds(const Lambda* lambda) {
    Lambdas result;
    for (auto use : lambda->uses()) {
        if (const Select* select = use->isa<Select>()) {
            for (auto select_user : select->uses()) {
                assert(select_user.index() == 0);
                result.push_back(select_user->as_lambda());
            }
        } else {
            if (!direct || use.index() == 0)
                if (Lambda* ulambda = use->isa_lambda())
                    result.push_back(ulambda);
        }
    }

    return result;
}

Lambdas& Lambda::succs() const {
    if (ops() == former_ops_)
        return succs_;

    former_ops_.resize(ops().size());
    std::copy(ops().begin(), ops().end(), former_ops_.begin());
    succs_.clear();
    std::queue<const Def*> queue;
    std::unordered_set<const Def*> done;
    const Def* def = this;
    goto start;

    while (!queue.empty()) {
        def = queue.front();
        queue.pop();

        if (Lambda* lambda = def->isa_lambda()) {
            succs_.push_back(lambda);
            continue;
        } 
start:
        for (auto op : def->ops()) {
            if (done.find(op) == done.end()) {
                queue.push(op);
                done.insert(op);
            }
        }
    }

    return succs_;
}

Lambdas Lambda::preds() const {
    // TODO cache the preds like in Lambda::succs -- but this not so obvious as it seems!!!
    std::vector<Lambda*> preds;
    std::queue<const Def*> queue;
    std::unordered_set<const Def*> done;
    const Def* def = this;
    goto start;

    while (!queue.empty()) {
        def = queue.front();
        queue.pop();

        if (Lambda* lambda = def->isa_lambda()) {
            preds.push_back(lambda);
            continue;
        } 
start:
        for (auto use : def->uses()) {
            if (done.find(use) == done.end()) {
                queue.push(use);
                done.insert(use);
            }
        }
    }

    return preds;
}

Lambdas Lambda::direct_preds() const { return find_preds<true>(this); }

bool Lambda::is_cascading() const {
    if (uses().size() != 1)
        return false;

    Use use = *uses().begin();
    return use->isa<Lambda>() && use.index() > 0;
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
    for (auto arg : args)
        set_op(x++, arg);
}

void Lambda::branch(const Def* cond, const Def* tto, const Def*  fto) {
    return jump(world().select(cond, tto, fto), ArrayRef<const Def*>(nullptr, 0));
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
    const Pi* pi = ret_type != nullptr ? world().pi2(world().mem(), ret_type) : world().pi1(world().mem());
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
    for (auto tracker : tracked_values_)
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
        if (!is_sealed_) {
            const Param* param = append_param(type, name);
            todos_.push_back(Todo(handle, param->index(), type, name));
            return set_value(handle, param);
        }

        Lambdas preds = this->preds();
        switch (preds.size()) {
            case 0: goto return_bottom;
            case 1: return set_value(handle, preds.front()->get_value(handle, type, name));
            default: {
                if (is_visited_)
                    return set_value(handle, append_param(type, name)); // create param to break cycle

                is_visited_ = true;
                const Def* same = nullptr;
                for (auto pred : preds) {
                    const Def* def = pred->get_value(handle, type, name);
                    if (same && same != def) {
                        same = (const Def*)-1; // defs from preds are different
                        break;
                    }
                    same = def;
                }
                assert(same != nullptr);
                is_visited_ = false;

                // fix any params which may have been introduced to break the cycle above
                const Def* def = nullptr;
                if (const Tracker* tracker = find_tracker(handle))
                    def = fix(Todo(handle, tracker->def()->as<Param>()->index(), type, name));

                if (same != (const Def*)-1)
                    return same;

                def = def ? def : fix(Todo(handle, append_param(type, name)->index(), type, name));
                return set_value(handle, def);
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

    for (auto todo : todos_)
        fix(todo);
    todos_.clear();
}

const Def* Lambda::fix(const Todo& todo) {
    size_t index = todo.index();
    const Param* param = this->param(index);

    assert(is_sealed() && "must be sealed");
    assert(todo.index() == param->index());

    for (auto pred : preds()) {
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
    const Def* same = nullptr;
    for (auto pred : preds) {
        const Def* def = pred->arg(index);
        if (def == param || same == def)
            continue;
        if (same)
            return param;
        same = def;
    }
    assert(same != nullptr);

    AutoVector<const Tracker*> uses = param->tracked_uses();
    param->replace(same);

    for (auto peek : param->peek())
        peek.from()->update_arg(index, world().bottom(param->type()));

    for (auto tracker : uses) {
        if (Lambda* lambda = tracker->def()->isa_lambda()) {
            for (auto succ : lambda->succs()) {
                size_t index = -1;
                for (size_t i = 0, e = succ->num_args(); i != e; ++i) {
                    if (succ->arg(i) == tracker->def()) {
                        index = i;
                        break;
                    }
                }
                if (index != size_t(-1) && param != succ->param(index))
                    succ->try_remove_trivial_param(succ->param(index));
            }
        }
    }

    return same;
}

} // namespace anydsl2
