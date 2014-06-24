#include "thorin/lambda.h"

#include <iostream>

#include "thorin/literal.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/be/thorin.h"
#include "thorin/util/queue.h"

namespace thorin {

Lambda* Lambda::stub(Type2Type& type2type, const std::string& name) const {
    auto result = world().lambda(type()->specialize(type2type).as<FnType>(), attribute(), intrinsic(), name);
    for (size_t i = 0, e = num_params(); i != e; ++i)
        result->param(i)->name = param(i)->name;
    return result;
}

Lambda* Lambda::update_op(size_t i, Def def) {
    unset_op(i);
    set_op(i, def);
    return this;
}

FnType Lambda::arg_fn_type() const {
    Array<Type> elems(num_args());
    for (size_t i = 0, e = num_args(); i != e; ++i)
        elems[i] = arg(i)->type();

    return world().fn_type(elems);
}

const Param* Lambda::append_param(Type param_type, const std::string& name) {
    size_t size = type()->num_args();
    Array<Type> elems(size + 1);
    *std::copy(type()->elems().begin(), type()->elems().end(), elems.begin()) = param_type;
    clear_type();
    set_type(param_type->world().fn_type(elems));             // update type
    auto param = world().param(param_type, this, size, name); // append new param
    params_.push_back(param);

    return param;
}

template<bool direct, bool indirect>
static Lambdas preds(const Lambda* lambda) {
    std::vector<Lambda*> preds;
    std::queue<Use> queue;
    DefSet done;

    auto enqueue = [&] (Def def) {
        for (auto use : def->uses()) {
            if (done.find(use) == done.end()) {
                queue.push(use);
                done.insert(use);
            }
        }
    };

    done.insert(lambda);
    enqueue(lambda);

    while (!queue.empty()) {
        auto use = pop(queue);
        if (auto lambda = use->isa_lambda()) {
            if ((use.index() == 0 && direct) || (use.index() != 0 && indirect))
                preds.push_back(lambda);
            continue;
        } 

        enqueue(use);
    }

    return preds;
}

template<bool direct, bool indirect>
static Lambdas succs(const Lambda* lambda) {
    std::vector<Lambda*> succs;
    std::queue<Def> queue;
    DefSet done;

    auto enqueue = [&] (Def def) {
        if (done.find(def) == done.end()) {
            queue.push(def);
            done.insert(def);
        }
    };

    done.insert(lambda);
    if (direct && !lambda->empty())
        enqueue(lambda->to());
    if (indirect) {
        for (auto arg : lambda->args())
            enqueue(arg);
    }

    while (!queue.empty()) {
        auto def = pop(queue);
        if (auto lambda = def->isa_lambda()) {
            succs.push_back(lambda);
            continue;
        } 
        for (auto op : def->ops())
            enqueue(op);
    }

    return succs;
}

Lambdas Lambda::preds() const { return thorin::preds<true, true>(this); }
Lambdas Lambda::succs() const { return thorin::succs<true, true>(this); }
Lambdas Lambda::direct_preds() const { return thorin::preds<true, false>(this); }
Lambdas Lambda::direct_succs() const { return thorin::succs<true, false>(this); }
Lambdas Lambda::indirect_preds() const { return thorin::preds<false, true>(this); }
Lambdas Lambda::indirect_succs() const { return thorin::succs<false, true>(this); }

bool Lambda::is_builtin() const { return intrinsic().is(Lambda::Builtin); }
void Lambda::set_intrinsic() {
    attribute().set(Lambda::Thorin);
    if (name=="cuda") intrinsic().set(Lambda::CUDA);
    else if (name=="nvvm") intrinsic().set(Lambda::NVVM);
    else if (name=="spir") intrinsic().set(Lambda::SPIR);
    else if (name=="opencl") intrinsic().set(Lambda::OPENCL);
    else if (name=="parallel") intrinsic().set(Lambda::Parallel);
    else if (name=="vectorized") intrinsic().set(Lambda::Vectorize);
    else if (name=="mmap") intrinsic().set(Lambda::Mmap);
    else if (name=="munmap") intrinsic().set(Lambda::Munmap);
    else assert(false && "unsupported thorin intrinsic");
}

template<typename T>
static bool aggregate_connected_builtins(const Lambda* lambda, T value, std::function<T(T, Lambda*)> func) {
    if (!lambda->is_builtin()) {
        for (auto use : lambda->uses()) {
            if (auto lambda = (use->isa<Global>() ? *use->uses().begin() : use)->isa<Lambda>())
                if (auto to_lambda = lambda->to()->isa_lambda())
                    if (to_lambda->is_builtin())
                        value = func(value, to_lambda);
        }
    }
    return value;
}

bool Lambda::is_connected_to_builtin() const {
    return aggregate_connected_builtins<bool>(this, false, [&](bool v, Lambda* lambda) { return true; });
}

bool Lambda::is_connected_to_builtin(uint32_t flags) const {
    return aggregate_connected_builtins<bool>(this, false, [&](bool v, Lambda* lambda) { return v || lambda->intrinsic().is(flags); });
}

std::vector<Lambda*> Lambda::connected_to_builtin_lambdas() const {
    std::vector<Lambda*> result;
    aggregate_connected_builtins<bool>(this, false, [&](bool v, Lambda* lambda) { result.push_back(lambda); return true; });
    return result;
}

bool Lambda::is_cascading() const {
    if (uses().size() != 1)
        return false;

    Use use = *uses().begin();
    return use->isa<Lambda>() && use.index() > 0;
}

bool Lambda::is_basicblock() const { return type()->is_basicblock(); }
bool Lambda::is_returning() const { return type()->is_returning(); }
void Lambda::dump_head() const { emit_head(this); }
void Lambda::dump_jump() const { emit_jump(this); }

/*
 * terminate
 */

void Lambda::jump(Def to, ArrayRef<Def> args) {
    unset_ops();
    resize(args.size()+1);
    set_op(0, to);

    size_t x = 1;
    for (auto arg : args)
        set_op(x++, arg);
}

void Lambda::branch(Def cond, Def tto, Def fto) {
    return jump(world().select(cond, tto, fto), ArrayRef<Def>(nullptr, 0));
}

std::pair<Lambda*, Def> Lambda::call(Def to, ArrayRef<Def> args, Type ret_type) {
    if (ret_type.empty()) {
        jump(to, args);
        return std::make_pair(nullptr, Def());
    }

    std::vector<Type> cont_elems;
    cont_elems.push_back(world().mem_type());
    bool pack = false;
    if (auto tuple = ret_type.isa<TupleType>()) {
        pack = true;
        for (auto elem : tuple->elems())
            cont_elems.push_back(elem);
    } else
        cont_elems.push_back(ret_type);

    auto next = world().lambda(world().fn_type(cont_elems), name);
    next->param(0)->name = "mem";

    // create jump to next
    size_t csize = args.size() + 1;
    Array<Def> cargs(csize);
    *std::copy(args.begin(), args.end(), cargs.begin()) = next;
    jump(to, cargs);

    // determine return value
    Def ret;
    if (pack) {
        Array<Def> defs(next->num_params()-1);
        auto p = next->params().slice_from_begin(1);
        std::copy(p.begin(), p.end(), defs.begin());
        ret = world().tuple(defs);

    } else 
        ret = next->param(1);
    ret->name = to->name;

    return std::make_pair(next, ret);
}

/*
 * CPS construction
 */

Def Lambda::find_def(size_t handle) {
    increase_values(handle);
    return values_[handle];
}

Def Lambda::set_mem(Def def) { return set_value(0, def); }
Def Lambda::get_mem() { return get_value(0, world().mem_type(), "mem"); }

Def Lambda::set_value(size_t handle, Def def) { 
    increase_values(handle);
    return values_[handle] = def;
}

Def Lambda::get_value(size_t handle, Type type, const char* name) {
    if (auto def = find_def(handle))
        return def;

    if (parent() != this) { // is a function head?
        if (parent())
            return parent()->get_value(handle, type, name);
        goto return_bottom;
    } else {
        if (!is_sealed_) {
            const Param* param = append_param(type, name);
            todos_.emplace_back(handle, param->index(), type, name);
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
                const DefNode* same = nullptr;
                for (auto pred : preds) {
                    const DefNode* def = pred->get_value(handle, type, name);
                    if (same && same != def) {
                        same = (const DefNode*)-1; // defs from preds are different
                        break;
                    }
                    same = def;
                }
                assert(same != nullptr);
                is_visited_ = false;

                // fix any params which may have been introduced to break the cycle above
                const DefNode* def = nullptr;
                if (auto found = find_def(handle))
                    def = fix(Todo(handle, found->as<Param>()->index(), type, name));

                if (same != (const DefNode*)-1)
                    return same;

                Def result = def ? Def(def) : fix(Todo(handle, append_param(type, name)->index(), type, name));
                return set_value(handle, result);
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

Def Lambda::fix(const Todo& todo) {
    size_t index = todo.index();
    const Param* param = this->param(index);

    assert(is_sealed() && "must be sealed");
    assert(todo.index() == param->index());

    for (auto pred : preds()) {
        assert(!pred->empty());
        assert(pred->direct_succs().size() == 1 && "critical edge");
        auto def = pred->get_value(todo);

        // make potentially room for the new arg
        if (index >= pred->num_args())
            pred->resize(index+2);

        assert(!pred->arg(index) && "already set");
        pred->set_op(index + 1, def);
    }

    return try_remove_trivial_param(param);
}

Def Lambda::try_remove_trivial_param(const Param* param) {
    assert(param->lambda() == this);
    assert(is_sealed() && "must be sealed");

    Lambdas preds = this->preds();
    size_t index = param->index();

    // find Horspool-like phis
    const DefNode* same = nullptr;
    for (auto pred : preds) {
        Def def = pred->arg(index);
        if (def.deref() == param || same == def)
            continue;
        if (same)
            return param;
        same = def;
    }
    assert(same != nullptr);
    param->replace(same);

    for (auto peek : param->peek())
        peek.from()->update_arg(index, world().bottom(param->type()));

    for (auto use : same->uses()) {
        if (Lambda* lambda = use->isa_lambda()) {
            for (auto succ : lambda->succs()) {
                size_t index = -1;
                for (size_t i = 0, e = succ->num_args(); i != e; ++i) {
                    if (succ->arg(i) == use.def()) {
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

}
