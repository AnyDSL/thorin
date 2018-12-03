#include "thorin/continuation.h"

#include <iostream>

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

//------------------------------------------------------------------------------

Continuation* get_param_continuation(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return extract->agg()->as<Param>()->continuation();
    return def->as<Param>()->continuation();
}

size_t get_param_index(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return primlit_value<u64>(extract->index()->as<PrimLit>());
    assert(def->isa<Param>());
    return 0;
}

std::vector<Peek> peek(const Def* param) {
    std::vector<Peek> peeks;
    size_t index = get_param_index(param);
    for (auto use : get_param_continuation(param)->uses()) {
        if (auto pred = use->isa_continuation()) {
            if (use.index() == 0)
                peeks.emplace_back(pred->arg(index), pred);
        }
    }

    return peeks;
}

//------------------------------------------------------------------------------

void Continuation::set_filter(Defs filter) {
    set_filter(world().tuple(filter));
}

const Def* Continuation::callee() const {
    return empty() ? world().bottom(world().fn_type(), debug()) : op(0);
}

Continuation* Continuation::stub() const {
    Rewriter rewriter;

    auto result = world().continuation(type(), cc(), intrinsic(), debug_history());
    result->param()->debug() = param()->debug_history();
    rewriter.old2new[param()] = result->param();

    if (filter_ != nullptr) {
        auto new_filter = rewriter.instantiate(filter_);
        result->set_filter(new_filter);
    }

    return result;
}

size_t Continuation::num_params() const {
    if (auto tuple_type = param()->type()->isa<TupleType>())
        return tuple_type->num_ops();
    return 1;
}

const Def* Continuation::param(size_t i) const {
    //if (param()->type()->isa<TupleType>())
        return world().extract(param(), i);
    //return param();
}

Array<const Def*> Continuation::params() const {
    size_t n = num_params();
    Array<const Def*> params(n);
    for (size_t i = 0; i != n; ++i)
        params[i] = param(i);
    return params;
}

Array<const Def*> Continuation::args() const {
    size_t n = num_args();
    Array<const Def*> args(n);
    for (size_t i = 0; i != n; ++i)
        args[i] = arg(i);
    return args;
}

size_t Continuation::num_args() const {
    if (auto tuple_type = arg()->type()->isa<TupleType>())
        return tuple_type->num_ops();
    return 1;
}

const Def* Continuation::arg(size_t i) const {
    if (arg()->type()->isa<TupleType>())
        return world().extract(arg(), i);
    return arg();
}

const Def* Continuation::filter(size_t i) const {
    if (filter()->type()->isa<TupleType>())
        return world().extract(filter(), i);
    return filter();
}

// TODO get rid off this
size_t Call::num_args() const {
    if (auto tuple_type = arg()->type()->isa<TupleType>())
        return tuple_type->num_ops();
    return 1;
}

// TODO get rid off this
const Def* Call::arg(size_t i) const {
    if (arg()->type()->isa<TupleType>())
        return callee()->world().extract(arg(), i);
    return arg();
}

// TODO get rid off this
Array<const Def*> Call::args() const {
    size_t n = num_args();
    Array<const Def*> args(n);
    for (size_t i = 0; i != n; ++i)
        args[i] = arg(i);
    return args;
}

const Def* Continuation::mem_param() const {
    for (size_t i = 0, e = num_params(); i != e; ++i) {
        auto p = param(i);
        if (is_mem(p))
            return p;
    }
    return nullptr;
}

const Def* Continuation::ret_param() const {
    const Def* result = nullptr;
    for (size_t i = 0, e = num_params(); i != e; ++i) {
        auto p = param(i);
        if (p->order() >= 1) {
            assertf(result == nullptr, "only one ret_param allowed");
            result = p;
        }
    }
    return result;
}

void Continuation::destroy_body() {
    unset_ops();
    resize(0);
}

const FnType* Continuation::arg_fn_type() const {
    return callee()->type()->isa<ClosureType>()
        ? world().closure_type(arg()->type())->as<FnType>()
        : world().fn_type(arg()->type());
}

const Def* Continuation::append_param(const Type* param_type, Debug dbg) {
    assert(param_);
    auto old_domain = type()->domain();
    clear_type();
    auto new_domain = merge_tuple_type(old_domain, param_type);
    set_type(param_type->table().fn_type(new_domain));
    const_cast<Param*>(param_)->clear_type();         // HACK
    const_cast<Param*>(param_)->set_type(new_domain); // HACK
    auto p = params().back();
    p->debug() = dbg;
    return p;
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
    if (!empty()) {
        enqueue(callee());
        enqueue(arg());
    }

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

#if 0
void Continuation::set_all_true_filter() {
    filter_ = Array<const Def*>(num_params(), [&](size_t) { return world().literal_bool(true, Debug{}); });
}
#endif

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
    else if (name() == "reserve_shared")       intrinsic_ = Intrinsic::Reserve;
    else if (name() == "atomic")               intrinsic_ = Intrinsic::Atomic;
    else if (name() == "cmpxchg")              intrinsic_ = Intrinsic::CmpXchg;
    else if (name() == "undef")                intrinsic_ = Intrinsic::Undef;
    else ELOG("unsupported thorin intrinsic");
}

bool Continuation::is_basicblock() const { return type()->is_basicblock(); }
bool Continuation::is_returning() const { return type()->is_returning(); }

/*
 * terminate
 */

void Continuation::jump(const Def* callee, Defs args, Debug dbg) {
    jump(callee, world().tuple(args), dbg);
}

void Continuation::jump(const Def* callee, const Def* arg, Debug dbg) {
    jump_debug_ = dbg;
    if (auto continuation = callee->isa<Continuation>()) {
        switch (continuation->intrinsic()) {
            case Intrinsic::Branch: {
                auto cond = world().extract(arg, 0_s);
                auto t    = world().extract(arg, 1_s);
                auto f    = world().extract(arg, 2_s);
                if (auto lit = cond->isa<PrimLit>())
                    return jump(lit->value().get_bool() ? t : f, Defs{}, dbg);
                if (t == f)
                    return jump(t, Defs{}, dbg);
                if (is_not(cond))
                    return branch(cond->as<ArithOp>()->rhs(), f, t, dbg);
                break;
            }
            case Intrinsic::Match: {
                auto args = arg->as<Tuple>()->ops();
                if (args.size() == 2) return jump(args[1], Defs{}, dbg);
                if (auto lit = args[0]->isa<PrimLit>()) {
                    for (size_t i = 2; i < args.size(); i++) {
                        if (world().extract(args[i], 0_s)->as<PrimLit>() == lit)
                            return jump(world().extract(args[i], 1), Defs{}, dbg);
                    }
                    return jump(args[1], Defs{}, dbg);
                }
                break;
            }
            default:
                break;
        }
    }

    unset_ops();
    resize(2); // TODO remove this
    set_op(0, callee);
    set_op(1, arg);
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
        if (call.arg(i)->isa<Top>())
            nargs.push_back(src->arg(i));
    }

    src->jump(dst, nargs, src->jump_debug());
}

Continuation* Continuation::update_op(size_t i, const Def* def) {
    std::array<const Def*, 2> new_ops = {callee(), arg()};
    new_ops[i] = def;
    jump(new_ops[0], new_ops[1], jump_location());
    return this;
}

std::ostream& Continuation::stream_head(std::ostream& os) const {
    os << unique_name();
    streamf(os, "{} {}", param()->type(), param());
    if (is_external())
        os << " extern ";
    if (cc() == CC::Device)
        os << " device ";
    if (filter())
        streamf(os, " @({})", filter());
    return os;
}

std::ostream& Continuation::stream_jump(std::ostream& os) const {
    if (!empty()) {
        streamf(os, "{} {}", callee(), arg());
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
