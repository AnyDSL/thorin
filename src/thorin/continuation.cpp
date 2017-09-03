#include "thorin/continuation.h"

#include <iostream>

#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"

namespace thorin {

//------------------------------------------------------------------------------

std::vector<Param::Peek> Param::peek() const {
    std::vector<Peek> peeks;
    for (auto use : continuation()->uses()) {
        if (auto pred = use->isa_continuation()) {
            if (use.index() == 0)
                peeks.emplace_back(pred->arg(index()), pred);
        }
    }

    return peeks;
}

//------------------------------------------------------------------------------

const Def* Continuation::callee() const {
    return empty() ? world().bottom(world().fn_type(), debug()) : op(0);
}

Continuation* Continuation::stub() const {
    auto result = world().continuation(type(), cc(), intrinsic(), debug_history());
    for (size_t i = 0, e = num_params(); i != e; ++i)
        result->param(i)->debug() = param(i)->debug_history();

    return result;
}

Array<const Def*> Continuation::params_as_defs() const {
    Array<const Def*> params(num_params());
    for (size_t i = 0, e = num_params(); i != e; ++i)
        params[i] = param(i);
    return params;
}

const Param* Continuation::mem_param() const {
    for (auto param : params()) {
        if (is_mem(param))
            return param;
    }
    return nullptr;
}

const Param* Continuation::ret_param() const {
    const Param* result = nullptr;
    for (auto param : params()) {
        if (param->order() >= 1) {
            assertf(result == nullptr, "only one ret_param allowed");
            result = param;
        }
    }
    return result;
}

void Continuation::destroy_body() {
    unset_ops();
    resize(0);
}

const FnType* Continuation::arg_fn_type() const {
    Array<const Type*> args(num_args());
    for (size_t i = 0, e = num_args(); i != e; ++i)
        args[i] = arg(i)->type();

    return world().fn_type(args);
}

const Param* Continuation::append_param(const Type* param_type, Debug dbg) {
    size_t size = type()->num_ops();
    Array<const Type*> ops(size + 1);
    *std::copy(type()->ops().begin(), type()->ops().end(), ops.begin()) = param_type;
    clear_type();
    set_type(param_type->world().fn_type(ops));              // update type
    auto param = world().param(param_type, this, size, dbg); // append new param
    params_.push_back(param);

    return param;
}

template<bool direct, bool indirect>
static Continuations preds(const Continuation* continuation) {
    std::vector<Continuation*> preds;
    std::queue<Use> queue;
    DefSet done;

    auto enqueue = [&] (const Def* def) {
        for (auto use : def->uses()) {
            if (done.find(use) == done.end()) {
                queue.push(use);
                done.insert(use);
            }
        }
    };

    done.insert(continuation);
    enqueue(continuation);

    while (!queue.empty()) {
        auto use = pop(queue);
        if (auto continuation = use->isa_continuation()) {
            if ((use.index() == 0 && direct) || (use.index() != 0 && indirect))
                preds.push_back(continuation);
            continue;
        }

        enqueue(use);
    }

    return preds;
}

template<bool direct, bool indirect>
static Continuations succs(const Continuation* continuation) {
    std::vector<Continuation*> succs;
    std::queue<const Def*> queue;
    DefSet done;

    auto enqueue = [&] (const Def* def) {
        if (done.find(def) == done.end()) {
            queue.push(def);
            done.insert(def);
        }
    };

    done.insert(continuation);
    if (direct && !continuation->empty())
        enqueue(continuation->callee());
    if (indirect) {
        for (auto arg : continuation->args())
            enqueue(arg);
    }

    while (!queue.empty()) {
        auto def = pop(queue);
        if (auto continuation = def->isa_continuation()) {
            succs.push_back(continuation);
            continue;
        }

        for (auto op : def->ops()) {
            if (op->order() >= 1)
                enqueue(op);
        }
    }

    return succs;
}

Continuations Continuation::preds() const { return thorin::preds<true, true>(this); }
Continuations Continuation::succs() const { return thorin::succs<true, true>(this); }
Continuations Continuation::direct_preds() const { return thorin::preds<true, false>(this); }
Continuations Continuation::direct_succs() const { return thorin::succs<true, false>(this); }
Continuations Continuation::indirect_preds() const { return thorin::preds<false, true>(this); }
Continuations Continuation::indirect_succs() const { return thorin::succs<false, true>(this); }

void Continuation::make_external() { return world().add_external(this); }
void Continuation::make_internal() { return world().remove_external(this); }
bool Continuation::is_external() const { return world().is_external(this); }
bool Continuation::is_intrinsic() const { return intrinsic_ != Intrinsic::None; }
bool Continuation::is_accelerator() const { return Intrinsic::_Accelerator_Begin <= intrinsic_ && intrinsic_ < Intrinsic::_Accelerator_End; }
void Continuation::set_intrinsic() {
    if      (name() == "cuda")           intrinsic_ = Intrinsic::CUDA;
    else if (name() == "nvvm")           intrinsic_ = Intrinsic::NVVM;
    else if (name() == "opencl")         intrinsic_ = Intrinsic::OpenCL;
    else if (name() == "amdgpu")         intrinsic_ = Intrinsic::AMDGPU;
    else if (name() == "parallel")       intrinsic_ = Intrinsic::Parallel;
    else if (name() == "spawn")          intrinsic_ = Intrinsic::Spawn;
    else if (name() == "sync")           intrinsic_ = Intrinsic::Sync;
    else if (name() == "vectorize")      intrinsic_ = Intrinsic::Vectorize;
    else if (name() == "pe_info")        intrinsic_ = Intrinsic::PeInfo;
    else if (name() == "pe_known")       intrinsic_ = Intrinsic::PeKnown;
    else if (name() == "reserve_shared") intrinsic_ = Intrinsic::Reserve;
    else if (name() == "atomic")         intrinsic_ = Intrinsic::Atomic;
    else if (name() == "cmpxchg")        intrinsic_ = Intrinsic::CmpXchg;
    else if (name() == "undef")          intrinsic_ = Intrinsic::Undef;
    else ELOG(this, "unsupported thorin intrinsic");
}

bool Continuation::is_basicblock() const { return type()->is_basicblock(); }
bool Continuation::is_returning() const { return type()->is_returning(); }

/*
 * terminate
 */

void Continuation::jump(const Def* callee, Defs args, Debug dbg) {
    jump_debug_ = dbg;
    if (auto continuation = callee->isa<Continuation>()) {
        switch (continuation->intrinsic()) {
            case Intrinsic::Branch: {
                assert(args.size() == 3);
                auto cond = args[0], t = args[1], f = args[2];
                if (auto lit = cond->isa<PrimLit>())
                    return jump(lit->value().get_bool() ? t : f, {}, dbg);
                if (t == f)
                    return jump(t, {}, dbg);
                if (is_not(cond))
                    return branch(cond->as<ArithOp>()->rhs(), f, t, dbg);
                break;
            }
            case Intrinsic::Match:
                if (args.size() == 2) return jump(args[1], {}, dbg);
                if (auto lit = args[0]->isa<PrimLit>()) {
                    for (size_t i = 2; i < args.size(); i++) {
                        if (world().extract(args[i], 0_s)->as<PrimLit>() == lit)
                            return jump(world().extract(args[i], 1), {}, dbg);
                    }
                    return jump(args[1], {}, dbg);
                }
                break;
            default:
                break;
        }
    }

    unset_ops();
    resize(args.size()+1);
    set_op(0, callee);

    size_t x = 1;
    for (auto arg : args)
        set_op(x++, arg);

    verify();
}

void Continuation::branch(const Def* cond, const Def* t, const Def* f, Debug dbg) {
    return jump(world().branch(), {cond, t, f}, dbg);
}

void Continuation::match(const Def* val, Continuation* otherwise, Defs patterns, ArrayRef<Continuation*> continuations, Debug dbg) {
    Array<const Def*> args(patterns.size() + 2);

    args[0] = val;
    args[1] = otherwise;
    assert(patterns.size() == continuations.size());
    for (size_t i = 0; i < patterns.size(); i++)
        args[i + 2] = world().tuple({patterns[i], continuations[i]}, dbg);

    return jump(world().match(val->type(), patterns.size()), args, dbg);
}

std::pair<Continuation*, const Def*> Continuation::call(const Def* callee, Defs args, const Type* ret_type, Debug dbg) {
    if (ret_type == nullptr) {
        jump(callee, args, dbg);
        return std::make_pair(nullptr, nullptr);
    }

    std::vector<const Type*> cont_args;
    cont_args.push_back(world().mem_type());
    cont_args.push_back(ret_type);

    auto next = world().continuation(world().fn_type(cont_args), dbg);
    next->param(0)->debug().set("mem");

    // create jump to next
    size_t csize = args.size() + 1;
    Array<const Def*> cargs(csize);
    *std::copy(args.begin(), args.end(), cargs.begin()) = next;
    jump(callee, cargs, dbg);

    // determine return value
    const Def* ret = nullptr;
    ret = next->param(1);
    ret->debug().set(callee->name());

    return std::make_pair(next, ret);
}

void jump_to_dropped_call(Continuation* src, Continuation* dst, const Call& call) {
    std::vector<const Def*> nargs;
    for (size_t i = 0, e = src->num_args(); i != e; ++i) {
        if (!call.arg(i))
            nargs.push_back(src->arg(i));
    }

    src->jump(dst, nargs, src->jump_debug());
}

Continuation* Continuation::update_op(size_t i, const Def* def) {
    Array<const Def*> new_ops(ops());
    new_ops[i] = def;
    jump(new_ops.front(), new_ops.skip_front(), jump_location());
    return this;
}

/*
 * value numbering
 */

const Def* Continuation::find_def(size_t handle) {
    increase_values(handle);
    return values_[handle];
}

const Def* Continuation::set_mem(const Def* def) { return set_value(0, def); }
const Def* Continuation::get_mem() { return get_value(0, world().mem_type(), { "mem" }); }

const Def* Continuation::set_value(size_t handle, const Def* def) {
    increase_values(handle);
    return values_[handle] = def;
}

const Def* Continuation::get_value(size_t handle, const Type* type, Debug dbg) {
    auto result = find_def(handle);
    if (result)
        goto return_result;

    if (parent() != this) { // is a function head?
        if (parent()) {
            result = parent()->get_value(handle, type, dbg);
            goto return_result;
        }
    } else {
        if (!is_sealed_) {
            auto param = append_param(type, dbg);
            todos_.emplace_back(handle, param->index(), type, dbg);
            result = set_value(handle, param);
            goto return_result;
        }

        Continuations preds = this->preds();
        switch (preds.size()) {
            case 0:
                goto return_bottom;
            case 1:
                result = set_value(handle, preds.front()->get_value(handle, type, dbg));
                goto return_result;
            default: {
                if (is_visited_) {
                    result = set_value(handle, append_param(type, dbg)); // create param to break cycle
                    goto return_result;
                }

                is_visited_ = true;
                const Def* same = nullptr;
                for (auto pred : preds) {
                    auto def = pred->get_value(handle, type, dbg);
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
                if (auto found = find_def(handle))
                    def = fix(handle, found->as<Param>()->index(), type, dbg);

                if (same != (const Def*)-1) {
                    result = same;
                    goto return_result;
                }

                if (def) {
                    result = set_value(handle, def);
                    goto return_result;
                }

                auto param = append_param(type, dbg);
                set_value(handle, param);
                fix(handle, param->index(), type, dbg);
                result = param;
                goto return_result;
            }
        }
    }

return_bottom:
    WLOG(&dbg, "'{}' may be undefined", dbg.name());
    return set_value(handle, world().bottom(type));

return_result:
    assert(result->type() == type);
    return result;
}

void Continuation::seal() {
    assert(!is_sealed() && "already sealed");
    is_sealed_ = true;

    for (const auto& todo : todos_)
        fix(todo.handle(), todo.index(), todo.type(), todo.debug());
    todos_.clear();
}

const Def* Continuation::fix(size_t handle, size_t index, const Type* type, Debug dbg) {
    auto param = this->param(index);

    assert(is_sealed() && "must be sealed");
    assert(index == param->index());

    for (auto pred : preds()) {
        assert(!pred->empty());
        assert(pred->direct_succs().size() == 1 && "critical edge");
        auto def = pred->get_value(handle, type, dbg);

        // make potentially room for the new arg
        if (index >= pred->num_args())
            pred->resize(index+2);

        assert(!pred->arg(index) && "already set");
        pred->set_op(index + 1, def);
    }

    return try_remove_trivial_param(param);
}

const Def* Continuation::try_remove_trivial_param(const Param* param) {
    assert(param->continuation() == this);
    assert(is_sealed() && "must be sealed");

    Continuations preds = this->preds();
    size_t index = param->index();

    // find Horspool-like phis
    const Def* same = nullptr;
    for (auto pred : preds) {
        auto def = pred->arg(index);
        if (def == param || same == def)
            continue;
        if (same)
            return param;
        same = def;
    }
    assert(same != nullptr);
    param->replace(same);

    for (auto peek : param->peek()) {
        auto continuation = peek.from();
        continuation->unset_op(index+1);
        continuation->set_op(index+1, world().bottom(param->type(), param->debug()));
    }

    for (auto use : same->uses()) {
        if (Continuation* continuation = use->isa_continuation()) {
            for (auto succ : continuation->succs()) {
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

std::ostream& Continuation::stream_head(std::ostream& os) const {
    os << unique_name();
    //stream_type_params(os, type());
    stream_list(os, params(), [&](const Param* param) { streamf(os, "{} {}", param->type(), param); }, "(", ")");
    if (is_external())
        os << " extern ";
    if (cc() == CC::Device)
        os << " device ";
    return os;
}

std::ostream& Continuation::stream_jump(std::ostream& os) const {
    if (!empty()) {
        os << callee();
        os << '(' << stream_list(args(), [&](const Def* def) { os << def; }) << ')';
    }
    return os;
}

void Continuation::dump_head() const { stream_head(std::cout) << endl; }
void Continuation::dump_jump() const { stream_jump(std::cout) << endl; }

//------------------------------------------------------------------------------

bool visit_uses(Continuation* cont, std::function<bool(Continuation*)> func) {
    if (!cont->is_intrinsic()) {
        for (auto use : cont->uses()) {
            if (auto continuation = (use->isa<Global>() ? *use->uses().begin() : use)->isa_continuation()) // TODO make more robust
                if (func(continuation))
                        return true;
        }
    }
    return false;
}

bool visit_capturing_intrinsics(Continuation* cont, std::function<bool(Continuation*)> func) {
    if (!cont->is_intrinsic()) {
        for (auto use : cont->uses()) {
            if (auto continuation = (use->isa<Global>() ? *use->uses().begin() : use)->isa_continuation()) // TODO make more robust
                if (auto callee = continuation->callee()->isa_continuation())
                    if (callee->is_intrinsic() && func(callee))
                        return true;
        }
    }
    return false;
}

bool is_passed_to_accelerator(Continuation* cont) {
    return visit_capturing_intrinsics(cont, [&] (Continuation* continuation) { return continuation->is_accelerator(); });
}

bool is_passed_to_intrinsic(Continuation* cont, Intrinsic intrinsic) {
    return visit_capturing_intrinsics(cont, [&] (Continuation* continuation) { return continuation->intrinsic() == intrinsic; });
}

void clear_value_numbering_table(World& world) {
    for (auto continuation : world.continuations())
        continuation->clear_value_numbering_table();
}

}
