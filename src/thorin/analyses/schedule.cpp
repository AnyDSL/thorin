#include "thorin/analyses/schedule.h"

#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"
#include "thorin/util/queue.h"

namespace thorin {

typedef DefMap<const CFNode*> Def2CFNode;
typedef std::set<Use, UseLT> Uses;

//------------------------------------------------------------------------------

class Scheduler {
public:
    Scheduler(const Scope& scope, Schedule& schedule)
        : scope_(scope)
        , cfg_(scope.f_cfg())
        , domtree_(cfg_.domtree())
        , looptree_(cfg_.looptree())
        , schedule_(schedule)
    {
        compute_def2uses();
        schedule_early();
        schedule_late();
        schedule_smart();
        topo_sort();
    }

    const Scope& scope() const { return scope_; }
    const F_CFG& cfg() const { return cfg_; }
    const DomTree& domtree() const { return domtree_; }
    const LoopTree<true>& looptree() const { return looptree_; }
    const DefMap<Uses>& def2uses() const { return def2uses_; }
    const Uses& uses(Def def) const { return def2uses_.find(def)->second; }

    Range<filter_iterator<const Def*, std::function<bool(Def)>>> ops(Def def) {
        std::function<bool(Def)> pred = [&](Def op) -> bool {
            return !op->isa_lambda() && def2uses().find(op) != def2uses().end();
        };
        return range(def->ops().begin(), def->ops().end(), pred);
    }

    void compute_def2uses();
    void schedule_early();
    void schedule_early(Def);
    void schedule_late();
    void schedule_late(Def);
    void schedule_smart();
    void schedule_smart(const PrimOp*);
    void topo_sort();

private:
    const Scope& scope_;
    const F_CFG& cfg_;
    const DomTree& domtree_;
    const LoopTree<true>& looptree_;
    DefMap<Uses> def2uses_;
    Def2CFNode def2early_;
    Def2CFNode def2late_;
    Def2CFNode def2smart_;
    DefMap<int> def2num;
    Schedule& schedule_;
};

void Scheduler::compute_def2uses() {
    std::queue<Def> queue;
    DefSet done;

    auto enqueue = [&](Def def, size_t i, Def op) {
        if (scope()._contains(op)) {
            auto p1 = def2uses_[op].emplace(i, def);
            assert_unused(p1.second);
            auto p2 = done.insert(op);
            if (p2.second)
                queue.push(op);
        }
    };

    for (auto n : cfg().reverse_post_order()) {
        queue.push(n->lambda());
        auto p = done.insert(n->lambda());
        assert_unused(p.second);
    }

    while (!queue.empty()) {
        auto def = pop(queue);
        for (size_t i = 0, e = def->size(); i != e; ++i)
            enqueue(def, i, def->op(i));
    }
}

void Scheduler::schedule_early(Def def) {
    if (!def2early_.contains(def)) {
        if (auto param = def->isa<Param>()) {
            def2early_[param] = cfg()[param->lambda()];
            return;
        }

        auto primop = def->as<PrimOp>();
        auto n = cfg().entry();
        for (auto op : ops(primop)) {
            schedule_early(op);
            auto m = def2early_[op];
            if (domtree().depth(m) > domtree().depth(n))
                n = m;
        }

        def2early_[primop] = n;
    }
}

void Scheduler::schedule_early() {
    for (const auto& p : def2uses()) {
        if (auto primop = p.first->isa<PrimOp>())
            schedule_early(primop);
    }
}

void Scheduler::schedule_late(Def def) {
    if (!def2late_.contains(def)) {
        if (auto lambda = def->isa_lambda()) {
            def2late_[lambda] = cfg()[lambda];
            return;
        }

        auto primop = def->as<PrimOp>();
        const CFNode* n = nullptr;
        for (auto use : uses(primop)) {
            schedule_late(use);
            auto m = def2late_[use];
            n = n ? domtree().lca(n, m) : m;
        }

        def2late_[primop] = n;
    }
}

void Scheduler::schedule_late() {
    for (const auto& p : def2uses()) {
        if (auto primop = p.first->isa<PrimOp>())
            schedule_late(primop);
    }
}

void Scheduler::schedule_smart(const PrimOp* primop) {
    auto early = def2early_[primop];
    auto late  = def2late_ [primop];
    auto smart = late;
    int depth = looptree()[late]->depth();


    if (primop->isa<Enter>() || primop->isa<Slot>() || Enter::is_out_mem(primop) || Enter::is_out_frame(primop))
        smart = early;
    else {
        for (auto i = late; i != early;) {
            i = domtree().idom(i);
            int cur_depth = looptree()[i]->depth();
            if (cur_depth < depth) {
                smart = i;
                depth = cur_depth;
            }
        }
    }

    schedule_[smart].primops_.push_back(primop);
    def2smart_[primop] = smart;
}

void Scheduler::schedule_smart() {
    for (const auto& p : def2uses()) {
        if (auto primop = p.first->isa<PrimOp>())
            schedule_smart(primop);
    }
}

void Scheduler::topo_sort() {
    for (auto& block : schedule_.blocks_) {
        std::vector<const PrimOp*> primops;

        std::queue<const PrimOp*> queue;
        DefSet done;

        auto inside = [&](Def def) {
            auto i = def2smart_.find(def);
            return i != def2smart_.end() && i->second == block.node();
        };

        auto enqueue = [&](const PrimOp* primop) {
            if (!done.contains(primop)) {
                for (auto op : primop->ops()) {
                    if (inside(op) && !done.contains(op))
                        return;
                }

                queue.push(primop);
                done.insert(primop);
                primops.push_back(primop);
            }
        };

        for (auto primop : block)
            enqueue(primop);

        while (!queue.empty()) {
            auto primop = pop(queue);

            for (auto use : uses(primop)) {
                if (auto primop = use->isa<PrimOp>()) {
                    if (inside(primop))
                        enqueue(primop);
                }
            }
        }

        assert(block.primops_.size() == primops.size());
        swap(block.primops_, primops);
    }
}

//------------------------------------------------------------------------------

Schedule::Schedule(const Scope& scope)
    : scope_(scope)
    , indices_(cfg())
    , blocks_(cfa().size())
{
    block_schedule();
    Scheduler(scope, *this);
    verify();
}

void Schedule::block_schedule() {
    // until we have sth better simply use the RPO of the CFG
    size_t i = 0;
    for (auto n : cfg().reverse_post_order()) {
        auto& block = blocks_[i];
        block.node_ = n;
        block.index_ = i;
        indices_[n] = i++;
    }
    assert(blocks_.size() == i);
}

void Schedule::verify() {
#ifndef NDEBUG
    auto& domtree = cfg().domtree();
    Schedule::Map<Def> block2mem(*this);

    for (auto& block : *this) {
        Def mem = block.lambda()->mem_param();
        mem = mem ? mem : block2mem[(*this)[domtree.idom(block.node())]];
        for (auto primop : block) {
            if (auto memop = primop->isa<MemOp>()) {
                if (memop->mem() != mem)
                    WLOG("incorrect schedule: % (current mem is %)", memop, mem);
                mem = memop->out_mem();
            }
        }
        block2mem[block] = mem;
    }
#endif
}

//------------------------------------------------------------------------------

}
