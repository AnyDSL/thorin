#include "thorin/lam.h"

#include <iostream>

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

//------------------------------------------------------------------------------

Lam* get_param_lam(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return extract->agg()->as<Param>()->lam();
    return def->as<Param>()->lam();
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
    for (auto use : get_param_lam(param)->uses()) {
        if (auto pred = use->isa_lam()) {
            if (auto app = pred->app()) {
                if (use.index() == 1) // is it an arg of pred?
                    peeks.emplace_back(app->arg(index), pred);
            }
        }
    }

    return peeks;
}

//------------------------------------------------------------------------------

const Def* Param::vrebuild(World& to, const Type*, Defs ops) const { return to.param(ops[0]->as_lam(), debug()); }
const Def* App  ::vrebuild(World& to, const Type*, Defs ops) const { return to.app  (ops[0], ops[1], debug()); }

//------------------------------------------------------------------------------

Lam::Lam(const Pi* pi, CC cc, Intrinsic intrinsic, Debug dbg)
    : Def(Node_Lam, pi, 2, dbg)
    , cc_(cc)
    , intrinsic_(intrinsic)
{
    set_op(0, world().literal_bool(false));
    set_op(1, world().top(pi->codomain()));

    contains_lam_ = true;
}

const Param* Lam::param(Debug dbg) const {
    return world().param(this->as_lam(), dbg);
}

bool Lam::is_empty() const { return body()->isa<Top>(); }

void Lam::set_filter(Defs filter) {
    set_filter(world().tuple(filter));
}


Def* Lam::vstub(World& to, const Type* type) const {
    return to.lam(type->as<Pi>(), cc(), intrinsic(), debug_history());
}

size_t Lam::num_params() const {
    if (auto tuple_type = param()->type()->isa<TupleType>())
        return tuple_type->num_ops();
    return 1;
}

const Def* Lam::param(size_t i, Debug dbg) const {
    //if (param()->type()->isa<TupleType>())
        return world().extract(param(), i, dbg);
    //return param();
}

Array<const Def*> Lam::params() const {
    size_t n = num_params();
    Array<const Def*> params(n);
    for (size_t i = 0; i != n; ++i)
        params[i] = param(i);
    return params;
}

const Def* Lam::filter(size_t i) const {
    if (filter()->type()->isa<TupleType>())
        return world().extract(filter(), i);
    return filter();
}

size_t App::num_args() const {
    if (auto tuple_type = arg()->type()->isa<TupleType>())
        return tuple_type->num_ops();
    return 1;
}

const Def* App::arg(size_t i) const {
    if (arg()->type()->isa<TupleType>())
        return callee()->world().extract(arg(), i);
    return arg();
}

Array<const Def*> App::args() const {
    size_t n = num_args();
    Array<const Def*> args(n);
    for (size_t i = 0; i != n; ++i)
        args[i] = arg(i);
    return args;
}

const Def* Lam::mem_param() const {
    for (size_t i = 0, e = num_params(); i != e; ++i) {
        auto p = param(i);
        if (is_mem(p))
            return p;
    }
    return nullptr;
}

const Def* Lam::ret_param() const {
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

void Lam::destroy_filter() {
    update_op(0, world().literal_bool(false));
}

void Lam::destroy_body() {
    update_op(0, world().literal_bool(false));
    update_op(1, world().top(type()->codomain()));
}

#if 0
const Pi* Lam::arg_pi() const {
    return world().pi(arg()->type());
}
#endif

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
        if (auto lam = use->isa_lam()) {
            preds.push_back(lam);
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
    enqueue(body());

    while (!queue.empty()) {
        auto def = pop(queue);
        if (auto lam = def->isa_lam()) {
            succs.push_back(lam);
            continue;
        }

        for (auto op : def->ops()) {
            if (op->contains_lam())
                enqueue(op);
        }
    }

    return succs;
}

#if 0
void Lam::set_all_true_filter() {
    filter_ = Array<const Def*>(num_params(), [&](size_t) { return world().literal_bool(true, Debug{}); });
}
#endif

void Lam::make_external() { return world().add_external(this); }
void Lam::make_internal() { return world().remove_external(this); }
bool Lam::is_external() const { return world().is_external(this); }
bool Lam::is_intrinsic() const { return intrinsic_ != Intrinsic::None; }
bool Lam::is_accelerator() const { return Intrinsic::_Accelerator_Begin <= intrinsic_ && intrinsic_ < Intrinsic::_Accelerator_End; }
void Lam::set_intrinsic() {
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

bool Lam::is_basicblock() const { return type()->is_basicblock(); }
bool Lam::is_returning() const { return type()->is_returning(); }

/*
 * terminate
 */

void Lam::app(const Def* callee, Defs args, Debug dbg) {
    app(callee, world().tuple(args), dbg);
}

void Lam::app(const Def* callee, const Def* arg, Debug dbg) {
    if (auto lam = callee->isa<Lam>()) {
        switch (lam->intrinsic()) {
            case Intrinsic::Branch: {
                auto cond = world().extract(arg, 0_s);
                auto t    = world().extract(arg, 1_s);
                auto f    = world().extract(arg, 2_s);
                if (auto lit = cond->isa<PrimLit>())
                    return app(lit->value().get_bool() ? t : f, Defs{}, dbg);
                if (t == f)
                    return app(t, Defs{}, dbg);
                if (is_not(cond))
                    return branch(cond->as<ArithOp>()->rhs(), f, t, dbg);
                break;
            }
            case Intrinsic::Match: {
                auto args = arg->as<Tuple>()->ops();
                if (args.size() == 2) return app(args[1], Defs{}, dbg);
                if (auto lit = args[0]->isa<PrimLit>()) {
                    for (size_t i = 2; i < args.size(); i++) {
                        if (world().extract(args[i], 0_s)->as<PrimLit>() == lit)
                            return app(world().extract(args[i], 1), Defs{}, dbg);
                    }
                    return app(args[1], Defs{}, dbg);
                }
                break;
            }
            default:
                break;
        }
    }

    Def::update_op(1, world().app(callee, arg, dbg));
    verify();
}

void Lam::branch(const Def* cond, const Def* t, const Def* f, Debug dbg) {
    return app(world().branch(), {cond, t, f}, dbg);
}

void Lam::match(const Def* val, Lam* otherwise, Defs patterns, ArrayRef<Lam*> lams, Debug dbg) {
    Array<const Def*> args(patterns.size() + 2);

    args[0] = val;
    args[1] = otherwise;
    assert(patterns.size() == lams.size());
    for (size_t i = 0; i < patterns.size(); i++)
        args[i + 2] = world().tuple({patterns[i], lams[i]}, dbg);

    return app(world().match(val->type(), patterns.size()), args, dbg);
}

void app_to_dropped_app(Lam* src, Lam* dst, const App* app) {
    std::vector<const Def*> nargs;
    auto src_app = src->body()->as<App>();
    for (size_t i = 0, e = src_app->num_args(); i != e; ++i) {
        if (app->arg(i)->isa<Top>())
            nargs.push_back(src_app->arg(i));
    }

    src->app(dst, nargs, src_app->debug());
}

std::ostream& Lam::stream_head(std::ostream& os) const {
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

std::ostream& Lam::stream_body(std::ostream& os) const {
    return streamf(os, "{}", body());
}

void Lam::dump_head() const { stream_head(std::cout) << endl; }
void Lam::dump_body() const { stream_body(std::cout) << endl; }

//------------------------------------------------------------------------------

bool visit_uses(Lam* lam, std::function<bool(Lam*)> func, bool include_globals) {
    if (!lam->is_intrinsic()) {
        for (auto use : lam->uses()) {
            auto def = include_globals && use->isa<Global>() ? use->uses().begin()->def() : use.def();
            if (auto lam = def->isa_lam())
                if (func(lam))
                    return true;
        }
    }
    return false;
}

bool visit_capturing_intrinsics(Lam* lam, std::function<bool(Lam*)> func, bool include_globals) {
    return visit_uses(lam, [&] (auto lam) {
        if (auto callee = lam->app()->callee()->isa_lam())
            return callee->is_intrinsic() && func(callee);
        return false;
    }, include_globals);
}

bool is_passed_to_accelerator(Lam* lam, bool include_globals) {
    return visit_capturing_intrinsics(lam, [&] (Lam* lam) { return lam->is_accelerator(); }, include_globals);
}

bool is_passed_to_intrinsic(Lam* lam, Intrinsic intrinsic, bool include_globals) {
    return visit_capturing_intrinsics(lam, [&] (Lam* lam) { return lam->intrinsic() == intrinsic; }, include_globals);
}

}
