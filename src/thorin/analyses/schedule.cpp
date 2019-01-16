#include "thorin/analyses/schedule.h"

#include "thorin/config.h"
#include "thorin/def.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
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
        Def2CFNode* def2node = nullptr;
        compute_def2uses();

        switch (schedule.tag()) {
            case Schedule::Early: this->schedule(); def2node = &def2early_; break;
            case Schedule::Late:  this->schedule(); def2node = &def2late_;  break;
            case Schedule::Smart: schedule_smart(); def2node = &def2smart_; break;
        }

        for (const auto& p : *def2node)
            schedule[p.second].defs_.push_back(p.first);

        topo_sort(*def2node);
    }

    void for_all_defs(std::function<void(const Def*)> f) {
        for (const auto& p : def2uses_) {
            if (!p.first->isa<Lam>())
                f(p.first);
        }
    }

    const Uses& uses(const Def* def) const { return def2uses_.find(def)->second; }
    void compute_def2uses();
    void schedule();
    void schedule_smart() { for_all_defs([&](const Def* def) { schedule_smart(def); }); }
    const CFNode* schedule_smart(const Def*);
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

    auto enqueue = [&](const Def* def) {
        if (done.emplace(def).second)
            queue.push(def);
    };

    enqueue(scope_.entry());

    while (!queue.empty()) {
        auto def = pop(queue);

        for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
            auto op = def->op(i);
            if (scope_.contains(op)) {
                def2uses_[op].emplace(i, def);
                enqueue(op);
            }
        }
    }
}

void Scheduler::schedule() {
    unique_queue<LamSet> lams;
    unique_stack<DefSet> defs;

    lams.push(scope_.entry());

    // visits all lambdas
    while (!lams.empty()) {
        auto cur = lams.pop();
        def2early_[cur] = cfg_[cur];
        def2late_ [cur] = cfg_[cur];
        defs.push(cur);

        // post-order walk of all ops within cur
        while (!defs.empty()) {
            auto def = defs.top();

            bool todo = false;
            for (auto op : def->ops()) {
                if (scope_.contains(op)) {
                    if (auto lam = op->isa_lam())
                        lams.push(lam); // for outer loop
                    else
                        todo |= defs.push(op);
                }
            }

            if (!todo) {
                auto result = cfg_[cur];
                def2late_[def] = result;
                for (auto op : def->ops()) {
                    if (!op->isa<Lam>() && scope_.contains(op)) {
                        auto n = def2early_[op];
                        if (domtree_.depth(n) > domtree_.depth(result))
                            result = n;
                    }
                }

                def2early_[def] = result;
                defs.pop();
            }
        }
    }
}

const CFNode* Scheduler::schedule_smart(const Def* def) {
    // TODO merge with method above
    auto i = def2smart_.find(def);
    if (i != def2smart_.end()) return i->second;

    schedule();
    auto early = def2early_[def];
    auto late  = def2late_ [def];

    const CFNode* result;
    if (def->isa<Enter>() || def->isa<Slot>() || Enter::is_out_mem(def) || Enter::is_out_frame(def)) {
        // Place allocas early for LLVM
        result = early;
    } else {
        result = late;
        int depth = looptree_[late]->depth();
        for (auto i = late; i != early;) {
            auto idom = domtree_.idom(i);
            assert(i != idom);
            i = idom;

            // HACK this should actually never occur
            if (i == nullptr) {
                WLOG("don't know where to put {}", def);
                result = late;
                break;
            }

            int cur_depth = looptree_[i]->depth();
            if (cur_depth < depth) {
                result = i;
                depth = cur_depth;
            }
        }
    }

    return def2smart_[def] = result;
}

void Scheduler::topo_sort(Def2CFNode& def2node) {
    for (auto& block : schedule_.blocks_) {
        std::vector<const Def*> defs;
        std::queue<const Def*> queue;
        DefSet done;

        auto inside = [&](const Def* def) {
            auto i = def2node.find(def);
            return i != def2node.end() && i->second == block.node();
        };

        auto enqueue = [&](const Def* def) {
            if (!done.contains(def)) {
                for (auto op : def->ops()) {
                    if (inside(op) && !done.contains(op))
                        return;
                }

                queue.push(def);
                done.emplace(def);
                defs.push_back(def);
            }
        };

        for (auto def : block)
            enqueue(def);

        while (!queue.empty()) {
            auto def = pop(queue);

            for (auto use : uses(def)) {
                if (auto def = use->isa<Def>()) {
                    if (inside(def))
                        enqueue(def);
                }
            }
        }

        assert(block.defs_.size() == defs.size());
        swap(block.defs_, defs);
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
#if 0//THORIN_ENABLE_CHECKS
    bool ok = true;
    auto& domtree = cfg().domtree();
    Schedule::Map<const Def*> block2mem(*this);

    for (auto& block : *this) {
        const Def* mem = block.lam()->mem_param();
        auto idom = block.lam() != scope().entry() ? domtree.idom(block.node()) : block.node();
        mem = mem ? mem : block2mem[(*this)[idom]];
        for (auto def : block) {
            if (auto memop = def->isa<MemOp>()) {
                if (memop->mem() != mem) {
                    WLOG("incorrect schedule: {} @ '{}'; current mem is {} @ '{}') - scope entry: {}", memop, memop->location(), mem, mem->location(), scope_.entry());
                    ok = false;
                }
                mem = memop->out_mem();
            }
        }
        block2mem[block] = mem;
    }

    assert(ok && "incorrectly wired or scheduled memory operations");
#endif
}

std::ostream& Schedule::stream(std::ostream& os) const {
    for (auto& block : *this) {
        auto lam = block.lam();
        if (lam->intrinsic() != Intrinsic::EndScope) {
            bool indent = lam != scope().entry();
            if (indent)
                os << up;
            os << endl;
            lam->stream_head(os) << up_endl;
            for (auto def : block)
                def->stream_assignment(os);

            lam->stream_body(os) << down_endl;
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

void verify_mem(World& world) {
    Scope::for_each(world, [&](const Scope& scope) { schedule(scope); });
}

}
