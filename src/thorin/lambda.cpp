#include "thorin/lambda.h"

#include <iostream>

#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/thorin.h"
#include "thorin/util/queue.h"

namespace thorin {

//------------------------------------------------------------------------------

std::vector<Param::Peek> Param::peek() const {
    std::vector<Peek> peeks;
    for (auto use : lambda()->uses()) {
        if (auto pred = use->isa_lambda()) {
            if (use.index() == 0)
                peeks.emplace_back(pred->arg(index()), pred);
        } else if (auto evalop = use->isa<EvalOp>()) {
            for (auto use : evalop->uses()) {
                if (auto pred = use->isa_lambda()) {
                    if (use.index() == 0)
                        peeks.emplace_back(pred->arg(index()), pred);
                }
            }
        }
    }

    return peeks;
}

//------------------------------------------------------------------------------

Def Lambda::to() const { 
    return empty() ? world().bottom(world().fn_type()) : op(0);
}

Lambda* Lambda::stub(Type2Type& type2type, const std::string& name) const {
    auto result = world().lambda(type()->specialize(type2type).as<FnType>(), cc(), intrinsic(), name);
    for (size_t i = 0, e = num_params(); i != e; ++i)
        result->param(i)->name = param(i)->name;
    return result;
}

Array<Def> Lambda::params_as_defs() const {
    Array<Def> params(num_params());
    for (size_t i = 0, e = num_params(); i != e; ++i)
        params[i] = param(i);
    return params;
}

const Param* Lambda::mem_param() const {
    for (auto param : params()) {
        if (param->type().isa<MemType>())
            return param;
    }
    return nullptr;
}

Lambda* Lambda::update_op(size_t i, Def def) {
    unset_op(i);
    set_op(i, def);
    return this;
}

void Lambda::refresh() {
    for (size_t i = 0, e = size(); i != e; ++i)
        update_op(i, op(i)->rebuild());
}

void Lambda::destroy_body() {
    unset_ops();
    resize(0);
}

FnType Lambda::arg_fn_type() const {
    Array<Type> args(num_args());
    for (size_t i = 0, e = num_args(); i != e; ++i)
        args[i] = arg(i)->type();

    return world().fn_type(args);
}

const Param* Lambda::append_param(Type param_type, const std::string& name) {
    size_t size = type()->num_args();
    Array<Type> args(size + 1);
    *std::copy(type()->args().begin(), type()->args().end(), args.begin()) = param_type;
    clear_type();
    set_type(param_type->world().fn_type(args));              // update type
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
        for (auto op : def->ops()) {
            if (op->order() >= 1)
                enqueue(op);
        }
    }

    return succs;
}

Lambdas Lambda::preds() const { return thorin::preds<true, true>(this); }
Lambdas Lambda::succs() const { return thorin::succs<true, true>(this); }
Lambdas Lambda::direct_preds() const { return thorin::preds<true, false>(this); }
Lambdas Lambda::direct_succs() const { return thorin::succs<true, false>(this); }
Lambdas Lambda::indirect_preds() const { return thorin::preds<false, true>(this); }
Lambdas Lambda::indirect_succs() const { return thorin::succs<false, true>(this); }

void Lambda::make_external() { return world().add_external(this); }
void Lambda::make_internal() { return world().remove_external(this); }
bool Lambda::is_external() const { return world().is_external(this); }
bool Lambda::is_intrinsic() const { return intrinsic_ != Intrinsic::None; }
bool Lambda::is_accelerator() const { return Intrinsic::_Accelerator_Begin <= intrinsic_ && intrinsic_ < Intrinsic::_Accelerator_End; }
void Lambda::set_intrinsic() {
    if      (name == "cuda")      intrinsic_ = Intrinsic::CUDA;
    else if (name == "nvvm")      intrinsic_ = Intrinsic::NVVM;
    else if (name == "spir")      intrinsic_ = Intrinsic::SPIR;
    else if (name == "opencl")    intrinsic_ = Intrinsic::OpenCL;
    else if (name == "parallel")  intrinsic_ = Intrinsic::Parallel;
    else if (name == "spawn")     intrinsic_ = Intrinsic::Spawn;
    else if (name == "sync")      intrinsic_ = Intrinsic::Sync;
    else if (name == "vectorize") intrinsic_ = Intrinsic::Vectorize;
    else if (name == "mmap")      intrinsic_ = Intrinsic::Mmap;
    else if (name == "munmap")    intrinsic_ = Intrinsic::Munmap;
    else if (name == "atomic")    intrinsic_ = Intrinsic::Atomic;
    else if (name == "bitcast")   intrinsic_ = Intrinsic::Reinterpret;
    else if (name == "select")    intrinsic_ = Intrinsic::Select;
    else if (name == "shuffle")   intrinsic_ = Intrinsic::Shuffle;
    else assert(false && "unsupported thorin intrinsic");
}

bool Lambda::visit_capturing_intrinsics(std::function<bool(Lambda*)> func) const {
    if (!is_intrinsic()) {
        for (auto use : uses()) {
            if (auto lambda = (use->isa<Global>() ? use->uses().front() : use)->isa<Lambda>()) // TODO make more robust
                if (auto to_lambda = lambda->to()->isa_lambda())
                    if (to_lambda->is_intrinsic() && func(to_lambda))
                        return true;
        }
    }
    return false;
}

bool Lambda::is_cascading() const {
    if (uses().size() == 1) {
        Use use = uses().front();
        return use->isa<Lambda>() && use.index() > 0;
    }
    return false;
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

void Lambda::branch(Def cond, Def t, Def f) {
    if (auto lit = cond->isa<PrimLit>())
        return jump(lit->value().get_bool() ? t : f, {});
    if (t == f)
        return jump(t, {});
    if (cond->is_not())
        return branch(cond->as<ArithOp>()->rhs(), f, t);
    return jump(world().branch(), {cond, t, f});
}

std::pair<Lambda*, Def> Lambda::call(Def to, ArrayRef<Def> args, Type ret_type) {
    if (ret_type.empty()) {
        jump(to, args);
        return std::make_pair(nullptr, Def());
    }

    std::vector<Type> cont_args;
    cont_args.push_back(world().mem_type());
    bool pack = false;
    if (auto tuple = ret_type.isa<TupleType>()) {
        pack = true;
        for (auto arg : tuple->args())
            cont_args.push_back(arg);
    } else
        cont_args.push_back(ret_type);

    auto next = world().lambda(world().fn_type(cont_args), name);
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
        auto p = next->params().skip_front();
        std::copy(p.begin(), p.end(), defs.begin());
        ret = world().tuple(defs);

    } else
        ret = next->param(1);
    ret->name = to->name;

    return std::make_pair(next, ret);
}

std::list<Lambda::ScopeInfo>::iterator Lambda::list_iter(const Scope* scope) {
    return std::find_if(scopes_.begin(), scopes_.end(), [&] (const ScopeInfo& info) {
        return info.scope->id() == scope->id();
    });
}

Lambda::ScopeInfo* Lambda::find_scope(const Scope* scope) {
    auto i = list_iter(scope);
    if (i != scopes_.end()) {
        // heuristic: swap found node to front so current scope will be found as first element in list
        if (i != scopes_.begin())
            scopes_.splice(scopes_.begin(), scopes_, i);
        return &scopes_.front();
    } else
        return nullptr;
}

/*
 * value numbering
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
            auto param = append_param(type, name);
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
                    def = fix(handle, found->as<Param>()->index(), type, name);

                if (same != (const DefNode*)-1)
                    return same;

                if (def)
                    return set_value(handle, def);

                auto param = append_param(type, name);
                set_value(handle, param);
                fix(handle, param->index(), type, name);
                return param;
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

    for (const auto& todo : todos_)
        fix(todo.handle(), todo.index(), todo.type(), todo.name());
    todos_.clear();
}

Def Lambda::fix(size_t handle, size_t index, Type type, const char* name) {
    auto param = this->param(index);

    assert(is_sealed() && "must be sealed");
    assert(index == param->index());

    for (auto pred : preds()) {
        assert(!pred->empty());
        assert(pred->direct_succs().size() == 1 && "critical edge");
        auto def = pred->get_value(handle, type, name);

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

//------------------------------------------------------------------------------

}
