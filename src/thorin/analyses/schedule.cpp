#include "thorin/analyses/schedule.h"

#include <fstream>

#include "thorin/config.h"
#include "thorin/def.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"

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

        switch (schedule.mode()) {
            case Schedule::Early: schedule_early(); def2node = &def2early_; break;
            case Schedule::Late:  schedule_late();  def2node = &def2late_;  break;
            case Schedule::Smart: schedule_smart(); def2node = &def2smart_; break;
        }

        for (auto&& p : *def2node)
            schedule[p.second].defs_.push_back(p.first);

        topo_sort(*def2node);
    }

    void for_all_defs(std::function<void(const Def*)> f) {
        for (auto&& p : def2uses_) {
            if (!p.first->isa_nominal())
                f(p.first);
        }
    }

    World& world() const { return scope_.world(); }
    const Uses& uses(const Def* def) const { return def2uses_.find(def)->second; }
    void compute_def2uses();
    void schedule_early() { for_all_defs([&](const Def* def) { schedule_early(def); }); }
    void schedule_late()  { for_all_defs([&](const Def* def) { schedule_late (def); }); }
    void schedule_smart() { for_all_defs([&](const Def* def) { schedule_smart(def); }); }
    const CFNode* schedule_early(const Def*);
    const CFNode* schedule_late(const Def*);
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
    // TODO use some variant of Scope::walk instead
    std::queue<const Def*> queue;
    DefSet done;

    auto enqueue = [&](const Def* def) {
        if (done.emplace(def).second)
            queue.push(def);
    };

    for (auto n : cfg_.reverse_post_order()) {
        if (n->nominal()->is_set())
            enqueue(n->nominal());
    }

    while (!queue.empty()) {
        auto def = pop(queue);

        for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
            auto op = def->op(i);
            if (scope_.contains(op)) {
                def2uses_[op].emplace(def, i);
                enqueue(op);
            }
        }
    }
}

const CFNode* Scheduler::schedule_early(const Def* def) {
    auto i = def2early_.find(def);
    if (i != def2early_.end()) return i->second;

    const CFNode* result;

    if (auto nom = def->isa_nominal()) {
        result = cfg_[nom];
    } else if (auto param = def->isa<Param>()) {
        result = schedule_early(param->nominal());
    } else {
        result = cfg_.entry();
        for (auto op : def->ops()) {
            if (op->isa_nominal()) continue;
            if (def2uses_.contains(op)) {
                auto n = schedule_early(op);
                if (domtree_.depth(n) > domtree_.depth(result))
                    result = n;
            }
        }
    }

    return def2early_[def] = result;
}

const CFNode* Scheduler::schedule_late(const Def* def) {
    auto i = def2late_.find(def);
    if (i != def2late_.end()) return i->second;

    const CFNode* result = nullptr;

    if (auto nom = def->isa_nominal()) {
        result = cfg_[nom];
    } else if (auto param = def->isa<Param>()) {
        result = schedule_late(param->nominal());
    } else {
        for (auto use : uses(def)) {
            auto n = schedule_late(use);
            result = result ? domtree_.least_common_ancestor(result, n) : n;
        }
    }

    return def2late_[def] = result;
}

const CFNode* Scheduler::schedule_smart(const Def* def) {
    auto i = def2smart_.find(def);
    if (i != def2smart_.end())
        return i->second;

    auto early = schedule_early(def);
    auto late  = schedule_late (def);
    world().DLOG("schedule {}: {} -- {}", def, early, late);

    const CFNode* result;
    //if (def->isa<Enter>() || def->isa<Slot>() || Enter::is_out_mem(def) || Enter::is_out_frame(def)) {
    if (false) {
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
                world().WLOG("don't know where to put {}", def);
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
                if (inside(use)) enqueue(use);
            }
        }

        assert(block.defs_.size() == defs.size());
        swap(block.defs_, defs);
    }
}

//------------------------------------------------------------------------------

Schedule::Schedule(const Scope& scope, Mode mode)
    : scope_(scope)
    , indices_(cfg())
    , blocks_(cfa().size())
    , mode_(mode)
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
#if 0
#if THORIN_ENABLE_CHECKS
    bool ok = true;
    auto& domtree = cfg().domtree();
    Schedule::Map<const Def*> block2mem(*this);

    for (auto& block : *this) {
        const Def* mem = block.nominal()->mem_param();
        auto idom = block.nominal() != scope().entry() ? domtree.idom(block.node()) : block.node();
        mem = mem ? mem : block2mem[(*this)[idom]];
        for (auto def : block) {
            if (is_memop(def)) {
                const Def* m;
                if (auto app = def->isa<App>())
                    m = app->arg(0);
                else
                    m = def->op(0);
                if (m != mem) {
                    def->dump();
                    WLOG("incorrect schedule: {} @ '{}'; current mem is {} @ '{}') - scope entry: {}", def, def->loc(), mem, mem->loc(), scope_.entry());
                    ok = false;
                }
                mem = def->out(0);
            }
        }
        block2mem[block] = mem;
    }

    assert(ok && "incorrectly wired or scheduled memory operations");
#endif
#endif
}

Stream& Schedule::stream(Stream& s) const {
    for (auto& block : *this) {
        auto nom = block.nominal();
        if (nom == scope_.exit()) continue;

        bool indent = nom != scope().entry();
        if (indent) s.indent();
        s.endl().fmt("{}: {}", nom->unique_name(), nom->type()).indent();

        for (auto def : block) stream_assignment(s.endl(), def);

        s.dedent();
        if (indent) s.dedent();
        s.endl();
    }

    return s;
}

template void Streamable<Schedule>::dump() const;
template void Streamable<Schedule>::write() const;

//------------------------------------------------------------------------------

void verify_mem(World& world) {
    world.visit([&](const Scope& scope) { schedule(scope); });
}

}
