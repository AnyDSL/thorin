#include "thorin/continuation.h"

#include <iostream>

#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

//------------------------------------------------------------------------------

const Def* Continuation::callee() const {
    return empty() ? world().bottom(world().fn_type(), debug()) : op(0);
}

Continuation* Continuation::stub() const {
    Rewriter rewriter;

    auto result = world().continuation(type(), attributes(), debug_history());
    for (size_t i = 0, e = num_params(); i != e; ++i) {
        result->param(i)->set_name(debug_history().name);
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
            if (op->has_dep(Dep::Cont))
                enqueue(op);
        }
    }

    return succs;
}

void Continuation::set_all_true_filter() {
    filter_ = Array<const Def*>(num_params(), [&](size_t) { return world().literal_bool(true, Debug{}); });
}

bool Continuation::is_accelerator() const { return Intrinsic::AcceleratorBegin <= intrinsic() && intrinsic() < Intrinsic::AcceleratorEnd; }
void Continuation::set_intrinsic() {
    if      (name() == "cuda")                 attributes().intrinsic = Intrinsic::CUDA;
    else if (name() == "nvvm")                 attributes().intrinsic = Intrinsic::NVVM;
    else if (name() == "opencl")               attributes().intrinsic = Intrinsic::OpenCL;
    else if (name() == "amdgpu")               attributes().intrinsic = Intrinsic::AMDGPU;
    else if (name() == "spirv")                attributes().intrinsic = Intrinsic::SpirV;
    else if (name() == "hls")                  attributes().intrinsic = Intrinsic::HLS;
    else if (name() == "parallel")             attributes().intrinsic = Intrinsic::Parallel;
    else if (name() == "fibers")               attributes().intrinsic = Intrinsic::Fibers;
    else if (name() == "spawn")                attributes().intrinsic = Intrinsic::Spawn;
    else if (name() == "sync")                 attributes().intrinsic = Intrinsic::Sync;
    else if (name() == "vectorize")            attributes().intrinsic = Intrinsic::Vectorize;
    else if (name() == "pe_info")              attributes().intrinsic = Intrinsic::PeInfo;
    else if (name() == "pipeline")             attributes().intrinsic = Intrinsic::Pipeline;
    else if (name() == "reserve_shared")       attributes().intrinsic = Intrinsic::Reserve;
    else if (name() == "atomic")               attributes().intrinsic = Intrinsic::Atomic;
    else if (name() == "atomic_load")          attributes().intrinsic = Intrinsic::AtomicLoad;
    else if (name() == "atomic_store")         attributes().intrinsic = Intrinsic::AtomicStore;
    else if (name() == "cmpxchg")              attributes().intrinsic = Intrinsic::CmpXchg;
    else if (name() == "undef")                attributes().intrinsic = Intrinsic::Undef;
    else world().ELOG("unsupported thorin intrinsic '{}'", name());
}

bool Continuation::is_basicblock() const { return type()->is_basicblock(); }
bool Continuation::is_returning() const { return type()->is_returning(); }

/*
 * terminate
 */

void Continuation::jump(const Def* callee, Defs args, Debug dbg) {
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

void Continuation::structured_loop_merge(const Continuation* loop_header, ArrayRef<const Continuation*> targets) {
    attributes_.intrinsic = Intrinsic::SCFLoopMerge;
    attributes_.scf_metadata.loop_epilogue.loop_header = loop_header;
    resize(targets.size());
    size_t x = 0;
    for (auto target : targets)
        set_op(x++, target);
}

void Continuation::structured_loop_continue(const Continuation* loop_header) {
    attributes_.intrinsic = Intrinsic::SCFLoopContinue;
    resize(1);
    set_op(0, loop_header);
}

void Continuation::structured_loop_header(const Continuation* loop_epilogue, const Continuation* loop_continue, ArrayRef<const Continuation*> targets) {
    attributes_.intrinsic = Intrinsic::SCFLoopHeader;
    resize(targets.size());
    attributes_.scf_metadata.loop_header.continue_target = loop_continue;
    attributes_.scf_metadata.loop_header.merge_target = loop_epilogue;
    size_t x = 0;
    for (auto target : targets)
        set_op(x++, target);
}

void jump_to_dropped_call(Continuation* src, Continuation* dst, const Call& call) {
    std::vector<const Def*> nargs;
    for (size_t i = 0, e = src->num_args(); i != e; ++i) {
        if (!call.arg(i))
            nargs.push_back(src->arg(i));
    }

    src->jump(dst, nargs);
}

Continuation* Continuation::update_op(size_t i, const Def* def) {
    Array<const Def*> new_ops(ops());
    new_ops[i] = def;
    jump(new_ops.front(), new_ops.skip_front());
    return this;
}

#if 0
std::ostream& Continuation::stream_head(std::ostream& os) const {
    os << unique_name();
    //stream_type_params(os, type());
    stream_list(os, params(), [&](const Param* param) { streamf(os, "{} {}", param->type(), param); }, "(", ")");
    if (is_exported())
        os << " export ";
    else if (is_imported())
        os << " import ";
    if (is_intrinsic() && intrinsic() == Intrinsic::Match)
        os << " " << "match" << " ";
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
#endif

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
