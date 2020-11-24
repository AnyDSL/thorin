#include "thorin/lam.h"

#include "thorin/world.h"

namespace thorin {

const Def* Lam::mem_param(const Def* dbg) {
    return thorin::isa<Tag::Mem>(param(0_s)->type()) ? param(0, dbg) : nullptr;
}

const Def* Lam::ret_param(const Def* dbg) {
    if (num_params() > 0) {
        auto p = param(num_params() - 1, dbg);
        if (auto pi = p->type()->isa<thorin::Pi>(); pi != nullptr && pi->is_cn()) return p;
    }
    return nullptr;
}

bool Lam::is_intrinsic() const { return intrinsic() != Intrinsic::None; }
bool Lam::is_accelerator() const { return Intrinsic::_Accelerator_Begin <= intrinsic() && intrinsic() < Intrinsic::_Accelerator_End; }

void Lam::set_intrinsic() {
    // TODO this is slow and inelegant - but we want to remove this code anyway
    auto n = debug().name;
    auto intrin = Intrinsic::None;
    if      (n == "cuda")                 intrin = Intrinsic::CUDA;
    else if (n == "nvvm")                 intrin = Intrinsic::NVVM;
    else if (n == "opencl")               intrin = Intrinsic::OpenCL;
    else if (n == "amdgpu")               intrin = Intrinsic::AMDGPU;
    else if (n == "hls")                  intrin = Intrinsic::HLS;
    else if (n == "parallel")             intrin = Intrinsic::Parallel;
    else if (n == "spawn")                intrin = Intrinsic::Spawn;
    else if (n == "sync")                 intrin = Intrinsic::Sync;
    else if (n == "anydsl_create_graph")  intrin = Intrinsic::CreateGraph;
    else if (n == "anydsl_create_task")   intrin = Intrinsic::CreateTask;
    else if (n == "anydsl_create_edge")   intrin = Intrinsic::CreateEdge;
    else if (n == "anydsl_execute_graph") intrin = Intrinsic::ExecuteGraph;
    else if (n == "vectorize")            intrin = Intrinsic::Vectorize;
    else if (n == "pe_info")              intrin = Intrinsic::PeInfo;
    else if (n == "reserve_shared")       intrin = Intrinsic::Reserve;
    else if (n == "atomic")               intrin = Intrinsic::Atomic;
    else if (n == "cmpxchg")              intrin = Intrinsic::CmpXchg;
    else if (n == "undef")                intrin = Intrinsic::Undef;
    else world().ELOG("unsupported thorin intrinsic");

    set_intrinsic(intrin);
}

bool Lam::is_basicblock() const { return type()->is_basicblock(); }
bool Lam::is_returning() const { return type()->is_returning(); }

void Lam::app(const Def* callee, const Def* arg, const Def* dbg) {
    assert(isa_nominal());
    auto filter = world().lit_false();
    set(filter, world().app(callee, arg, dbg));
}
void Lam::app(const Def* callee, Defs args, const Def* dbg) { app(callee, world().tuple(args), dbg); }
void Lam::branch(const Def* cond, const Def* t, const Def* f, const Def* mem, const Def* dbg) {
    return app(world().extract(world().tuple({f, t}), cond, dbg), mem, dbg);
}

void Lam::match(const Def* val, Defs cases, const Def* mem, const Def* dbg) {
    return app(world().match(val, cases, dbg), mem, dbg);
}

/*
 * Pi
 */

Pi* Pi::set_domain(Defs domains) { return Def::set(0, world().sigma(domains))->as<Pi>(); }

bool Pi::is_returning() const {
    bool ret = false;
    for (auto op : ops()) {
        switch (op->order()) {
            case 1:
                if (!ret) {
                    ret = true;
                    continue;
                }
                return false;
            default: continue;
        }
    }
    return ret;
}

// TODO remove
Lam* get_param_lam(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return extract->tuple()->as<Param>()->nominal()->as<Lam>();
    return def->as<Param>()->nominal()->as<Lam>();
}

// TODO remove
size_t get_param_index(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return as_lit<size_t>(extract->index());
    assert(def->isa<Param>());
    return 0;
}

std::vector<Peek> peek(const Def* param) {
    std::vector<Peek> peeks;
    size_t index = get_param_index(param);
    for (auto use : get_param_lam(param)->uses()) {
        if (auto app = use->isa<App>()) {
            for (auto use : app->uses()) {
                if (auto pred = use->isa_nominal<Lam>()) {
                    if (pred->body() == app)
                        peeks.emplace_back(app->arg(index), pred);
                }
            }
        }
    }

    return peeks;
}

}
