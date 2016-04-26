#include "thorin/analyses/schedule.h"

#include "thorin/continuation.h"
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
        Def2CFNode* def2node = nullptr;

        switch (schedule.kind()) {
            case Schedule::Early: schedule_early(); def2node = &def2early_; break;
            case Schedule::Late:  schedule_late();  def2node = &def2late_;  break;
            case Schedule::Smart: schedule_early(); schedule_late(); schedule_smart(); def2node = &def2smart_; break;
        }

        for (const auto& p : *def2node) {
            if (auto primop = p.first->isa<PrimOp>())
                schedule[p.second].primops_.push_back(primop);
        }

        topo_sort(*def2node);
    }

    void for_all_primops(std::function<void(const PrimOp*)> f) {
        for (const auto& p : def2uses_) {
            if (auto primop = p.first->isa<PrimOp>())
                f(primop);
        }
    }

    const Uses& uses(const Def* def) const { return def2uses_.find(def)->second; }
    void compute_def2uses();
    void schedule_early() { for_all_primops([&](const PrimOp* primop) { schedule_early(primop); }); }
    void schedule_late()  { for_all_primops([&](const PrimOp* primop) { schedule_late (primop); }); }
    void schedule_smart() { for_all_primops([&](const PrimOp* primop) { schedule_smart(primop); }); }
    void schedule_early(const Def*);
    void schedule_late(const Def*);
    void schedule_smart(const PrimOp*);
    void topo_sort(Def2CFNode&);

private:
    const Scope& scope_;
    const F_CFG& cfg_;
    const DomTree& domtree_;
    const LoopTree<true>& looptree_;
    DefMap<Uses> def2uses_;
    Def2CFNode def2early_;
    Def2CFNode def2late_;
    Def2CFNode def2smart_;
    Schedule& schedule_;
};

void Scheduler::compute_def2uses() {
    std::queue<const Def*> queue;
    DefSet done;

    auto enqueue = [&](const Def* def, size_t i, const Def* op) {
        if (scope_.contains(op)) {
            auto p1 = def2uses_[op].emplace(i, def);
            assert_unused(p1.second);
            auto p2 = done.insert(op);
            if (p2.second)
                queue.push(op);
        }
    };

    for (auto n : cfg_.reverse_post_order()) {
        queue.push(n->continuation());
        auto p = done.insert(n->continuation());
        assert_unused(p.second);
    }

    while (!queue.empty()) {
        auto def = pop(queue);
        for (size_t i = 0, e = def->size(); i != e; ++i)
            enqueue(def, i, def->op(i));
    }
}

void Scheduler::schedule_early(const Def* def) {
    if (!def2early_.contains(def)) {
        if (auto param = def->isa<Param>()) {
            def2early_[param] = cfg_[param->continuation()];
        } else {
            auto primop = def->as<PrimOp>();
            auto n = cfg_.entry();
            for (auto op : primop->ops()) {
                if (!op->isa_continuation() && def2uses_.find(op) != def2uses_.end()) {
                    schedule_early(op);
                    auto m = def2early_[op];
                    if (domtree_.depth(m) > domtree_.depth(n))
                        n = m;
                }
            }

            def2early_[primop] = n;
        }
    }
}

void Scheduler::schedule_late(const Def* def) {
    if (!def2late_.contains(def)) {
        if (auto continuation = def->isa_continuation()) {
            def2late_[continuation] = cfg_[continuation];
            return;
        }

        auto primop = def->as<PrimOp>();
        const CFNode* n = nullptr;
        for (auto use : uses(primop)) {
            schedule_late(use);
            auto m = def2late_[use];
            n = n ? domtree_.lca(n, m) : m;
        }

        def2late_[primop] = n;
    }
}

void Scheduler::schedule_smart(const PrimOp* primop) {
    auto early = def2early_[primop];
    auto late  = def2late_ [primop];
    auto smart = late;

    if (primop->isa<Enter>() || primop->isa<Slot>() || Enter::is_out_mem(primop) || Enter::is_out_frame(primop))
        smart = early;
    else {
        int depth = looptree_[late]->depth();
        for (auto i = late; i != early;) {
            i = domtree_.idom(i);
            int cur_depth = looptree_[i]->depth();
            if (cur_depth < depth) {
                smart = i;
                depth = cur_depth;
            }
        }
    }

    def2smart_[primop] = smart;
}

void Scheduler::topo_sort(Def2CFNode& def2node) {
    for (auto& block : schedule_.blocks_) {
        std::vector<const PrimOp*> primops;
        std::queue<const PrimOp*> queue;
        DefSet done;

        auto inside = [&](const Def* def) {
            auto i = def2node.find(def);
            return i != def2node.end() && i->second == block.node();
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

Schedule::Schedule(const Scope& scope, Kind kind)
    : scope_(scope)
    , indices_(cfg())
    , blocks_(cfa().size())
    , kind_(kind)
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
    Schedule::Map<const Def*> block2mem(*this);
    bool error = false;

    for (auto& block : *this) {
        const Def* mem = block.continuation()->mem_param();
        mem = mem ? mem : block2mem[(*this)[domtree.idom(block.node())]];
        for (auto primop : block) {
            if (auto memop = primop->isa<MemOp>()) {
                if (memop->mem() != mem) {
                    WLOG("incorrect schedule: % (current mem is %) - scope entry: %", memop, mem, scope_.entry());
                    error = true;
                }
                mem = memop->out_mem();
            }
        }
        block2mem[block] = mem;
    }

    if (error)
        thorin();
#endif
}

std::ostream& Schedule::stream(std::ostream& os) const {
    for (auto& block : *this) {
        auto continuation = block.continuation();
        if (continuation->intrinsic() != Intrinsic::EndScope) {
            bool indent = continuation != scope().entry();
            if (indent)
                os << up;
            os << endl;
            continuation->stream_head(os) << up_endl;
            for (auto primop : block)
                primop->stream_assignment(os);

            continuation->stream_jump(os) << down_endl;
            if (indent)
                os << down;
        }
    }
    return os << endl;
}

void Schedule::write_thorin(const char* filename) const { std::ofstream file(filename); stream(file); }

void Schedule::thorin() const {
    auto filename = world().name() + "_" + scope().entry()->unique_name() + ".thorin";
    write_thorin(filename.c_str());
}

//------------------------------------------------------------------------------

}
