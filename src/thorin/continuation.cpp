#include "thorin/continuation.h"

#include <iostream>

#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"

namespace thorin {

//------------------------------------------------------------------------------

Param::Param(const Type* type, Lam* continuation, size_t index, Debug dbg)
    : Def(Node_Param, type, 1, dbg)
    , index_(index)
{
    set_op(0, continuation);
}

//------------------------------------------------------------------------------

App::App(const Defs ops, Debug dbg) : Def(Node_App, ops[0]->world().bottom_type(), ops, dbg) {
#if THORIN_ENABLE_CHECKS
    verify();
#endif
}

void App::verify() const {
    auto callee_type = callee()->type()->isa<FnType>(); // works for closures too, no need for a special case
    assertf(callee_type, "callee type must be a FnType");
    assertf(callee_type->num_ops() == num_args(), "app node '{}' has fn type {} with {} parameters, but is supplied {} arguments", this, callee_type, callee_type->num_ops(), num_args());
    for (size_t i = 0; i < num_args(); i++) {
        auto pt = callee_type->op(i);
        auto at = arg(i)->type();
        assertf(pt == at, "app node argument {} has type {} but the callee was expecting {}", this, at, pt);
    }
    if (auto cont = callee()->isa_nom<Lam>()) {
        assert(!cont->dead_);
    }
}

//------------------------------------------------------------------------------

Filter::Filter(World& world, const Defs defs, Debug dbg) : Def(Node_Filter, world.bottom_type(), defs, dbg) {}

const Filter* Filter::cut(ArrayRef<size_t> indices) const {
    return world().filter(ops().cut(indices), debug());
}

//------------------------------------------------------------------------------

Lam::Lam(const FnType* fn, const Attributes& attributes, Debug dbg)
    : Def(Node_Continuation, fn, 2, dbg)
    , attributes_(attributes)
{
    params_.reserve(fn->num_ops());
    set_op(0, world().bottom(world().bottom_type()));
    set_op(1, world().filter({}, dbg));
}

Lam* Lam::stub() const {
    Rewriter rewriter;

    auto result = world().lambda(type(), attributes(), debug_history());
    for (size_t i = 0, e = num_params(); i != e; ++i) {
        result->param(i)->set_name(debug_history().name);
        rewriter.old2new[param(i)] = result->param(i);
    }

    if (!filter()->is_empty()) {
        Array<const Def*> new_conditions(num_params());
        for (size_t i = 0, e = num_params(); i != e; ++i)
            new_conditions[i] = rewriter.instantiate(filter()->condition(i));

        result->set_filter(world().filter(new_conditions, filter()->debug()));
    }

    return result;
}

Array<const Def*> Lam::params_as_defs() const {
    Array<const Def*> params(num_params());
    for (size_t i = 0, e = num_params(); i != e; ++i)
        params[i] = param(i);
    return params;
}

const Param* Lam::mem_param() const {
    for (auto param : params()) {
        if (is_mem(param))
            return param;
    }
    return nullptr;
}

const Param* Lam::ret_param() const {
    const Param* result = nullptr;
    for (auto param : params()) {
        if (param->order() >= 1) {
            assertf(result == nullptr, "only one ret_param allowed");
            result = param;
        }
    }
    return result;
}

void Lam::destroy(const char* cause) {
    world().VLOG("{} has been destroyed by {}", this, cause);
    destroy_filter();
    unset_op(0);
    set_op(0, world().bottom(world().bottom_type()));
    dead_ = true;
}

const FnType* Lam::arg_fn_type() const {
    assert(has_body());
    Array<const Type*> args(body()->num_args());
    for (size_t i = 0, e = body()->num_args(); i != e; ++i)
        args[i] = body()->arg(i)->type();

    return body()->callee()->type()->isa<ClosureType>()
           ? world().closure_type(args)->as<FnType>()
           : world().fn_type(args);
}

const Param* Lam::append_param(const Type* param_type, Debug dbg) {
    size_t size = type()->num_ops();
    Array<const Type*> ops(size + 1);
    *std::copy(type()->ops().begin(), type()->ops().end(), ops.begin()) = param_type;
    clear_type();
    set_type(param_type->table().fn_type(ops));              // update type
    auto param = world().param(param_type, this, size, dbg); // append new param
    params_.push_back(param);

    return param;
}

Lams Lam::preds() const {
    std::vector<Lam*> preds;
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
        if (auto continuation = use->isa_nom<Lam>()) {
            preds.push_back(continuation);
            continue;
        }

        enqueue(use);
    }

    return preds;
}

Lams Lam::succs() const {
    std::vector<Lam*> succs;
    std::queue<const Def*> queue;
    DefSet done;

    auto enqueue = [&] (const Def* def) {
        if (done.find(def) == done.end()) {
            queue.push(def);
            done.insert(def);
        }
    };

    done.insert(this);
    if (has_body())
        enqueue(body());

    while (!queue.empty()) {
        auto def = pop(queue);
        if (auto continuation = def->isa_nom<Lam>()) {
            succs.push_back(continuation);
            continue;
        }

        for (auto op : def->ops()) {
            if (op->has_dep(Dep::Lam))
                enqueue(op);
        }
    }

    return succs;
}

void Lam::destroy_filter() {
    set_filter(world().filter({}));
}

/// An all-true filter
const Filter* Lam::all_true_filter() const {
    auto conditions = Array<const Def*>(num_params(), [&](size_t) { return world().literal_bool(true, Debug{}); });
    return world().filter(conditions, debug());
}

bool Lam::is_accelerator() const { return Intrinsic::AcceleratorBegin <= intrinsic() && intrinsic() < Intrinsic::AcceleratorEnd; }
void Lam::set_intrinsic() {
    if      (name() == "cuda")           attributes().intrinsic = Intrinsic::CUDA;
    else if (name() == "nvvm")           attributes().intrinsic = Intrinsic::NVVM;
    else if (name() == "opencl")         attributes().intrinsic = Intrinsic::OpenCL;
    else if (name() == "amdgpu")         attributes().intrinsic = Intrinsic::AMDGPU;
    else if (name() == "hls")            attributes().intrinsic = Intrinsic::HLS;
    else if (name() == "parallel")       attributes().intrinsic = Intrinsic::Parallel;
    else if (name() == "fibers")         attributes().intrinsic = Intrinsic::Fibers;
    else if (name() == "spawn")          attributes().intrinsic = Intrinsic::Spawn;
    else if (name() == "sync")           attributes().intrinsic = Intrinsic::Sync;
    else if (name() == "vectorize")      attributes().intrinsic = Intrinsic::Vectorize;
    else if (name() == "pe_info")        attributes().intrinsic = Intrinsic::PeInfo;
    else if (name() == "pipeline")       attributes().intrinsic = Intrinsic::Pipeline;
    else if (name() == "reserve_shared") attributes().intrinsic = Intrinsic::Reserve;
    else if (name() == "atomic")         attributes().intrinsic = Intrinsic::Atomic;
    else if (name() == "atomic_load")    attributes().intrinsic = Intrinsic::AtomicLoad;
    else if (name() == "atomic_store")   attributes().intrinsic = Intrinsic::AtomicStore;
    else if (name() == "cmpxchg")        attributes().intrinsic = Intrinsic::CmpXchg;
    else if (name() == "cmpxchg_weak")   attributes().intrinsic = Intrinsic::CmpXchgWeak;
    else if (name() == "fence")          attributes().intrinsic = Intrinsic::Fence;
    else if (name() == "undef")          attributes().intrinsic = Intrinsic::Undef;
    else world().ELOG("unsupported thorin intrinsic '{}'", name());
}

bool Lam::is_basicblock() const { return type()->is_basicblock(); }
bool Lam::is_returning() const { return type()->is_returning(); }
bool Lam::is_external() const { return world().is_external(this); }

void Lam::jump(const Def* callee, Defs args, Debug dbg) {
    set_body(world().app(callee, args, dbg));
    verify();
}

void Lam::branch(const Def* cond, const Def* t, const Def* f, Debug dbg) {
    set_body(world().app(world().branch(), {cond, t, f}, dbg));
    verify();
}

void Lam::match(const Def* val, Lam* otherwise, Defs patterns, ArrayRef<Lam*> continuations, Debug dbg) {
    Array<const Def*> args(patterns.size() + 2);

    args[0] = val;
    args[1] = otherwise;
    assert(patterns.size() == continuations.size());
    for (size_t i = 0; i < patterns.size(); i++)
        args[i + 2] = world().tuple({patterns[i], continuations[i]}, dbg);

    set_body(world().app(world().match(val->type(), patterns.size()), args, dbg));
    verify();
}

void Lam::verify() const {
    if (!has_body()) {
        assertf(filter()->is_empty(), "continuations with no body should have an empty (no) filter");

        if (world().is_external(this)) {} // external (imported) continuations can of course have no body
        else if (dead_) {}
        else if (num_uses() == 0) {} // front-ends (ie Artic) may create such orphan lambda stubs currently, ideally these should only be tolerated until the first rebuild
        else if (intrinsic() != Intrinsic::None) {} // intrinsics don't have a body TODO: or do they ?
        else {
            // assertf(false, "{} has no body but does not correspond to any legitimate case where that may happen", *this);
        }
    } else {
        body()->verify();
        assert(!dead_); // destroy() should remove the body
        assert(intrinsic() == Intrinsic::None);
        assertf(filter()->is_empty() || num_params() == filter()->size(), "The filter needs to be either empty, or match the param count");
    }
}

/// Rewrites the body to only keep the non-specialized arguments
void jump_to_dropped_call(Lam* continuation, Lam* dropped, const Defs specialized_args) {
    assert(continuation->has_body());
    auto obody = continuation->body();
    std::vector<const Def*> nargs;
    for (size_t i = 0, e = obody->num_args(); i != e; ++i) {
        if (!specialized_args[i])
            nargs.push_back(obody->arg(i));
    }

    continuation->jump(dropped, nargs);
}

#if 0
std::ostream& Lam::stream_head(std::ostream& os) const {
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

std::ostream& Lam::stream_jump(std::ostream& os) const {
    if (!empty()) {
        os << callee();
        os << '(' << stream_list(args(), [&](const Def* def) { os << def; }) << ')';
    }
    return os;
}

void Lam::dump_head() const { stream_head(std::cout) << endl; }
void Lam::dump_jump() const { stream_jump(std::cout) << endl; }
#endif

//------------------------------------------------------------------------------

bool visit_uses(Lam* cont, std::function<bool(Lam*)> func, bool include_globals) {
    if (!cont->is_intrinsic()) {
        for (auto use : cont->uses()) {
            auto def = include_globals && use->isa<Global>() ? use->uses().begin()->def() : use.def();
            if (auto app = def->isa<App>()) {
                for (auto ucontinuation : app->using_continuations()) {
                    if (func(ucontinuation))
                        return true;
                }
            }
        }
    }
    return false;
}

bool visit_capturing_intrinsics(Lam* cont, std::function<bool(Lam*)> func, bool include_globals) {
    return visit_uses(cont, [&] (auto continuation) {
        if (!continuation->has_body()) return false;
        auto body = continuation->body();
        if (auto callee = body->callee()->template isa_nom<Lam>())
            return callee->is_intrinsic() && func(callee);
        return false;
    }, include_globals);
}

bool is_passed_to_accelerator(Lam* cont, bool include_globals) {
    return visit_capturing_intrinsics(cont, [&] (Lam* continuation) { return continuation->is_accelerator(); }, include_globals);
}

bool is_passed_to_intrinsic(Lam* cont, Intrinsic intrinsic, bool include_globals) {
    return visit_capturing_intrinsics(cont, [&] (Lam* continuation) { return continuation->intrinsic() == intrinsic; }, include_globals);
}

}
