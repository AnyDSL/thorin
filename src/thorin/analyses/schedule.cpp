#include "thorin/analyses/schedule.h"

#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/nest.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"

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

        switch (schedule.tag()) {
            case Schedule::Early: schedule_early(); def2node = &def2early_; break;
            case Schedule::Late:  schedule_late();  def2node = &def2late_;  break;
            case Schedule::Smart: schedule_smart(); def2node = &def2smart_; break;
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
    const CFNode* schedule_early(const Def*);
    const CFNode* schedule_late(const Def*);
    const CFNode* schedule_smart(const PrimOp*);
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
            auto p2 = done.emplace(op);
            if (p2.second)
                queue.push(op);
        }
    };

    for (auto n : cfg_.reverse_post_order()) {
        queue.push(n->continuation());
        auto p = done.emplace(n->continuation());
        assert_unused(p.second);
    }

    while (!queue.empty()) {
        auto def = pop(queue);
        for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
            // all reachable continuations have already been registered above
            // NOTE we might still see references to unreahable continuations in the schedule
            if (!def->op(i)->isa<Continuation>())
                enqueue(def, i, def->op(i));
        }
    }
}

const CFNode* Scheduler::schedule_early(const Def* def) {
    auto i = def2early_.find(def);
    if (i != def2early_.end())
        return i->second;

    if (auto param = def->isa<Param>())
        return def2early_[def] = cfg_[param->continuation()];

    auto result = cfg_.entry();
    for (auto op : def->as<PrimOp>()->ops()) {
        if (!op->isa_continuation() && def2uses_.find(op) != def2uses_.end()) {
            auto n = schedule_early(op);
            if (domtree_.depth(n) > domtree_.depth(result))
                result = n;
        }
    }

    return def2early_[def] = result;
}

const CFNode* Scheduler::schedule_late(const Def* def) {
    auto i = def2late_.find(def);
    if (i != def2late_.end())
        return i->second;

    if (auto continuation = def->isa_continuation())
        return def2late_[def] = cfg_[continuation];

    const CFNode* result = nullptr;
    auto primop = def->as<PrimOp>();
    for (auto use : uses(primop)) {
        auto n = schedule_late(use);
        result = result ? domtree_.lca(result, n) : n;
    }

    return def2late_[def] = result;
}

const CFNode* Scheduler::schedule_smart(const PrimOp* primop) {
    auto i = def2smart_.find(primop);
    if (i != def2smart_.end())
        return i->second;

    auto early = schedule_early(primop);
    auto late  = schedule_late (primop);

    const CFNode* result;
    if (primop->isa<Enter>() || primop->isa<Slot>() || Enter::is_out_mem(primop) || Enter::is_out_frame(primop))
        result = early;
    else {
        result = late;
        int depth = looptree_[late]->depth();
        for (auto i = late; i != early;) {
            auto idom = domtree_.idom(i);

            if (i == idom) {
                WLOG(primop, "don't know where to put {} - using late postion {}", primop, late);
                result = late;
                break;
            }

            i = idom;
            int cur_depth = looptree_[i]->depth();
            if (cur_depth < depth) {
                result = i;
                depth = cur_depth;
            }
        }
    }

    return def2smart_[primop] = result;
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
                done.emplace(primop);
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

Schedule::Schedule(const Scope& scope, Tag tag)
    : scope_(scope)
    , indices_(cfg())
    , blocks_(cfa().size())
    , tag_(tag)
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

    for (auto& block : *this) {
        const Def* mem = block.continuation()->mem_param();
        mem = mem ? mem : block2mem[(*this)[domtree.idom(block.node())]];
        for (auto primop : block) {
            if (auto memop = primop->isa<MemOp>()) {
                if (memop->mem() != mem)
                    WLOG(memop, "incorrect schedule: {} (current mem is {}) - scope entry: {}", memop, mem, scope_.entry());
                mem = memop->out_mem();
            }
        }
        block2mem[block] = mem;
    }
#endif
}

std::ostream& Schedule::stream(std::ostream& os) const {
    for (auto& block : *this) {
        auto continuation = block.continuation();
        if (continuation->intrinsic() != Intrinsic::EndScope) {
            // HACK
            auto n = scope().nest().node(continuation);
            const int depth = n ? n->depth() : 1;
            for (int i = 0; i != depth; ++i)
                os << up;

            os << endl;
            continuation->stream_head(os) << up_endl;
            for (auto primop : block)
                primop->stream_assignment(os);

            continuation->stream_jump(os) << down_endl;

            for (int i = 0; i != depth; ++i)
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
