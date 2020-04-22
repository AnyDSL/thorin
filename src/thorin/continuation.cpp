#include "thorin/continuation.h"

#include <iostream>

#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"
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
    Rewriter rewriter;

    auto result = world().continuation(type(), cc(), intrinsic(), debug_history());
    for (size_t i = 0, e = num_params(); i != e; ++i) {
        result->param(i)->debug() = param(i)->debug_history();
        rewriter.old2new[param(i)] = result->param(i);
    }

    if (!filter().empty()) {
        Array<const Def*> new_filter(num_params());
        for (size_t i = 0, e = num_params(); i != e; ++i)
            new_filter[i] = rewriter.instantiate(filter(i));

        result->set_filter(new_filter);
    }

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

    return callee()->type()->isa<ClosureType>()
        ? world().closure_type(args)->as<FnType>()
        : world().fn_type(args);
}

const Param* Continuation::append_param(const Type* param_type, Debug dbg) {
    size_t size = type()->num_ops();
    Array<const Type*> ops(size + 1);
    *std::copy(type()->ops().begin(), type()->ops().end(), ops.begin()) = param_type;
    clear_type();
    set_type(param_type->table().fn_type(ops));              // update type
    auto param = world().param(param_type, this, size, dbg); // append new param
    params_.push_back(param);

    return param;
}

Continuations Continuation::preds() const {
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

    done.insert(this);
    enqueue(this);

    while (!queue.empty()) {
        auto use = pop(queue);
        if (auto continuation = use->isa_continuation()) {
            preds.push_back(continuation);
            continue;
        }

        enqueue(use);
    }

    return preds;
}

Continuations Continuation::succs() const {
    std::vector<Continuation*> succs;
    std::queue<const Def*> queue;
    DefSet done;

    auto enqueue = [&] (const Def* def) {
        if (done.find(def) == done.end()) {
            queue.push(def);
            done.insert(def);
        }
    };

    done.insert(this);
    if (!empty())
        enqueue(callee());

    for (auto arg : args())
        enqueue(arg);

    while (!queue.empty()) {
        auto def = pop(queue);
        if (auto continuation = def->isa_continuation()) {
            succs.push_back(continuation);
            continue;
        }

        for (auto op : def->ops()) {
            if (op->contains_continuation())
                enqueue(op);
        }
    }

    return succs;
}

void Continuation::set_all_true_filter() {
    filter_ = Array<const Def*>(num_params(), [&](size_t) { return world().literal_bool(true, Debug{}); });
}

void Continuation::make_external() { return world().add_external(this); }
void Continuation::make_internal() { return world().remove_external(this); }
bool Continuation::is_external() const { return world().is_external(this); }
bool Continuation::is_intrinsic() const { return intrinsic_ != Intrinsic::None; }
bool Continuation::is_accelerator() const { return Intrinsic::_Accelerator_Begin <= intrinsic_ && intrinsic_ < Intrinsic::_Accelerator_End; }
void Continuation::set_intrinsic() {
    if      (name() == "cuda")                 intrinsic_ = Intrinsic::CUDA;
    else if (name() == "nvvm")                 intrinsic_ = Intrinsic::NVVM;
    else if (name() == "opencl")               intrinsic_ = Intrinsic::OpenCL;
    else if (name() == "amdgpu")               intrinsic_ = Intrinsic::AMDGPU;
    else if (name() == "hls")                  intrinsic_ = Intrinsic::HLS;
    else if (name() == "parallel")             intrinsic_ = Intrinsic::Parallel;
    else if (name() == "spawn")                intrinsic_ = Intrinsic::Spawn;
    else if (name() == "sync")                 intrinsic_ = Intrinsic::Sync;
    else if (name() == "anydsl_create_graph")  intrinsic_ = Intrinsic::CreateGraph;
    else if (name() == "anydsl_create_task")   intrinsic_ = Intrinsic::CreateTask;
    else if (name() == "anydsl_create_edge")   intrinsic_ = Intrinsic::CreateEdge;
    else if (name() == "anydsl_execute_graph") intrinsic_ = Intrinsic::ExecuteGraph;
    else if (name() == "vectorize")            intrinsic_ = Intrinsic::Vectorize;
    else if (name() == "pe_info")              intrinsic_ = Intrinsic::PeInfo;
    else if (name() == "pipeline")             intrinsic_ = Intrinsic::Pipeline;
    else if (name() == "reserve_shared")       intrinsic_ = Intrinsic::Reserve;
    else if (name() == "atomic")               intrinsic_ = Intrinsic::Atomic;
    else if (name() == "cmpxchg")              intrinsic_ = Intrinsic::CmpXchg;
    else if (name() == "undef")                intrinsic_ = Intrinsic::Undef;
    else ELOG("unsupported thorin intrinsic '{}'", name());
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

std::ostream& Continuation::stream_head(std::ostream& os) const {
    os << unique_name();
    //stream_type_params(os, type());
    stream_list(os, params(), [&](const Param* param) { streamf(os, "{} {}", param->type(), param); }, "(", ")");
    if (is_external())
        os << " extern ";
    if (cc() == CC::Device)
        os << " device ";
    if (!filter().empty())
        os << " @(" << stream_list(filter(), [&](const Def* def) { os << def; }) << ')';
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

bool visit_uses(Continuation* cont, std::function<bool(Continuation*)> func, bool include_globals) {
    if (!cont->is_intrinsic()) {
        for (auto use : cont->uses()) {
            auto def = include_globals && use->isa<Global>() ? use->uses().begin()->def() : use.def();
            if (auto continuation = def->isa_continuation())
                if (func(continuation))
                    return true;
        }
    }
    return false;
}

bool visit_capturing_intrinsics(Continuation* cont, std::function<bool(Continuation*)> func, bool include_globals) {
    return visit_uses(cont, [&] (auto continuation) {
        if (auto callee = continuation->callee()->isa_continuation())
            return callee->is_intrinsic() && func(callee);
        return false;
    }, include_globals);
}

bool is_passed_to_accelerator(Continuation* cont, bool include_globals) {
    return visit_capturing_intrinsics(cont, [&] (Continuation* continuation) { return continuation->is_accelerator(); }, include_globals);
}

bool is_passed_to_intrinsic(Continuation* cont, Intrinsic intrinsic, bool include_globals) {
    return visit_capturing_intrinsics(cont, [&] (Continuation* continuation) { return continuation->intrinsic() == intrinsic; }, include_globals);
}

}
