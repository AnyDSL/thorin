#include "thorin/continuation.h"

#include <iostream>

#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"

namespace thorin {

//------------------------------------------------------------------------------

Param::Param(World& world, const Type* type, const Continuation* continuation, size_t index, Debug dbg)
    : Def(world, Node_Param, type, { continuation }, dbg)
    //: Def(world, Node_Param, type, 1, dbg)
    , index_(index)
{
    //set_op(0, continuation);
}

const Def* Param::rebuild(World& world, const Type* t, Defs defs) const {
    assert(defs.size() == 1);
    auto cont = defs[0]->as<Continuation>();
    return cont->param(index());
}

hash_t Param::vhash() const {
    return hash_combine(Def::vhash(), (hash_t) index());
}

bool Param::equal(const Def* other) const {
    return Def::equal(other) && other->as<Param>()->index() == index();
}

//------------------------------------------------------------------------------

App::App(World& world, const Defs ops, Debug dbg) : Def(world, Node_App, ops[0]->world().bottom_type(), ops, dbg) {
#if THORIN_ENABLE_CHECKS
    verify();
    if (auto cont = callee()->isa_nom<Continuation>())
        assert(!cont->dead_); // debugging hint, apps should not be built with dead continuations
        // this cannot live in verify because continuations are mutable and may die later
#endif
}

bool App::verify() const {
    auto callee_type = callee()->type()->isa<FnType>(); // works for closures too, no need for a special case
    assertf(callee_type, "callee type must be a FnType");
    assertf(callee_type->num_ops() == num_args(), "app node '{}' has fn type {} with {} parameters, but is supplied {} arguments", this, callee_type, callee_type->num_ops(), num_args());
    for (size_t i = 0; i < num_args(); i++) {
        auto pt = callee_type->op(i);
        auto at = arg(i)->type();
        assertf(pt == at, "app node argument {} has type {} but the callee was expecting {}", this, at, pt);
    }
    return true;
}

//------------------------------------------------------------------------------

Filter::Filter(World& world, const Defs defs, Debug dbg) : Def(world, Node_Filter, world.bottom_type(), defs, dbg) {}

const Filter* Filter::cut(ArrayRef<size_t> indices) const {
    return world().filter(ops().cut(indices), debug());
}

//------------------------------------------------------------------------------

Continuation::Continuation(World& w, const FnType* pi, const Attributes& attributes, Debug dbg)
    : Def(w, Node_Continuation, pi, 2, dbg)
    , attributes_(attributes)
{
    params_.reserve(pi->num_ops());
    set_op(0, world().bottom(world().bottom_type()));
    set_op(1, world().filter({}, dbg));

    size_t i = 0;
    for (auto op : pi->types()) {
        auto p = w.param(op, this, i++, dbg);
        params_.emplace_back(p);
    }
}

// TODO: merge with regular stub()
Continuation* Continuation::mangle_stub() const {
    Rewriter rewriter;
    return mangle_stub(rewriter);
}

Continuation* Continuation::mangle_stub(Rewriter& rewriter) const {
    auto result = world().continuation(type(), attributes(), debug_history());
    for (size_t i = 0, e = num_params(); i != e; ++i) {
        result->param(i)->set_name(debug_history().name);
        rewriter.insert(param(i), result->param(i));
    }

    if (!filter()->is_empty()) {
        Array<const Def*> new_conditions(num_params());
        for (size_t i = 0, e = num_params(); i != e; ++i)
            new_conditions[i] = rewriter.instantiate(filter()->condition(i));

        result->set_filter(world().filter(new_conditions, filter()->debug()));
    }

    return result;
}

Continuation* Continuation::stub(World& nworld, const Type* t) const {
    assert(!dead_);
    // TODO maybe we want to deal with intrinsics in a more streamlined way
    if (this == world().branch())
        return nworld.branch();
    if (this == world().end_scope())
        return nworld.end_scope();

    auto npi = t->isa<FnType>();
    assert(npi);
    Continuation* ncontinuation = nworld.continuation(npi, attributes(), debug_history());
    assert(&ncontinuation->world() == &nworld);
    assert(&npi->world() == &nworld);
    for (size_t i = 0, e = num_params(); i != e; ++i)
        ncontinuation->param(i)->set_name(param(i)->debug_history().name);

    if (is_external())
        nworld.make_external(ncontinuation);
    return ncontinuation;
}

void Continuation::rebuild_from(const Def*, Defs nops) {
    if (this == world().branch())
        return;
    if (this == world().end_scope())
        return;

    auto napp = nops[0]->isa<App>();
    if (napp)
        set_body(napp);
    set_filter(nops[1]->as<Filter>());
    verify();
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

void Continuation::destroy(const char* cause) {
    world().VLOG("{} has been destroyed by {}", this, cause);
    destroy_filter();
    unset_op(0);
    set_op(0, world().bottom(world().bottom_type()));
    dead_ = true;
}

const FnType* Continuation::arg_fn_type() const {
    assert(has_body());
    Array<const Type*> args(body()->num_args());
    for (size_t i = 0, e = body()->num_args(); i != e; ++i)
        args[i] = body()->arg(i)->type();

    return body()->callee()->type()->isa<ClosureType>()
           ? world().closure_type(args)->as<FnType>()
           : world().fn_type(args);
}

const Param* Continuation::append_param(const Type* param_type, Debug dbg) {
    size_t size = type()->num_ops();
    Array<const Type*> ops(size + 1);
    *std::copy(type()->types().begin(), type()->types().end(), ops.begin()) = param_type;
    clear_type();
    set_type(world().fn_type(ops));              // update type
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
            if (use->isa<Param>())
                continue;
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
        if (auto continuation = use->isa_nom<Continuation>()) {
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
    if (has_body())
        enqueue(body());

    while (!queue.empty()) {
        auto def = pop(queue);
        if (auto continuation = def->isa_nom<Continuation>()) {
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

void Continuation::destroy_filter() {
    set_filter(world().filter({}));
}

/// An all-true filter
const Filter* Continuation::all_true_filter() const {
    auto conditions = Array<const Def*>(num_params(), [&](size_t) { return world().literal_bool(true, Debug{}); });
    return world().filter(conditions, debug());
}

bool Continuation::is_accelerator() const { return Intrinsic::AcceleratorBegin <= intrinsic() && intrinsic() < Intrinsic::AcceleratorEnd; }
void Continuation::set_intrinsic() {
    if      (name() == "cuda")           attributes().intrinsic = Intrinsic::CUDA;
    else if (name() == "nvvm")           attributes().intrinsic = Intrinsic::NVVM;
    else if (name() == "opencl")         attributes().intrinsic = Intrinsic::OpenCL;
    else if (name() == "amdgpu")         attributes().intrinsic = Intrinsic::AMDGPU;
    else if (name() == "shady_compute")  attributes().intrinsic = Intrinsic::ShadyCompute;
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

bool Continuation::is_basicblock() const { return type()->is_basicblock(); }
bool Continuation::is_returning() const { return type()->is_returning(); }
bool Continuation::is_external() const { return world().is_external(this); }

void Continuation::jump(const Def* callee, Defs args, Debug dbg) {
    set_body(world().app(callee, args, dbg));
    verify();
}

void Continuation::branch(const Def* mem, const Def* cond, const Def* t, const Def* f, Debug dbg) {
    set_body(world().app(world().branch(), {mem, cond, t, f}, dbg));
    verify();
}

void Continuation::match(const Def* mem, const Def* val, Continuation* otherwise, Defs patterns, ArrayRef<Continuation*> continuations, Debug dbg) {
    Array<const Def*> args(patterns.size() + 3);

    args[0] = mem;
    args[1] = val;
    args[2] = otherwise;
    assert(patterns.size() == continuations.size());
    for (size_t i = 0; i < patterns.size(); i++)
        args[i + 3] = world().tuple({patterns[i], continuations[i]}, dbg);

    set_body(world().app(world().match(val->type(), patterns.size()), args, dbg));
    verify();
}

bool Continuation::verify() const {
    bool ok = true;
    if (!has_body())
        assertf(filter()->is_empty(), "continuations with no body should have an empty (no) filter");
    else {
        ok &= body()->verify();
        assert(!dead_); // destroy() should remove the body
        assert(intrinsic() == Intrinsic::None);
        assertf(filter()->is_empty() || num_params() == filter()->size(), "The filter needs to be either empty, or match the param count");
    }
    return ok;
}

/// Rewrites the body to only keep the non-specialized arguments
void jump_to_dropped_call(Continuation* continuation, Continuation* dropped, const Defs specialized_args) {
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
            auto actual_uses = include_globals && use->isa<Global>() ? use->uses() : Uses { use };
            for (auto actual_use : actual_uses) {
                auto def = actual_use.def();
                if (auto app = def->isa<App>()) {
                    for (auto ucontinuation : app->using_continuations()) {
                        if (func(ucontinuation))
                            return true;
                    }
                }
            }
        }
    }
    return false;
}

bool visit_capturing_intrinsics(Continuation* cont, std::function<bool(Continuation*)> func, bool include_globals) {
    return visit_uses(cont, [&] (auto continuation) {
        if (!continuation->has_body()) return false;
        auto callee = continuation->body()->callee()->template isa_nom<Continuation>();
        if (callee)
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
