#include "thorin/analyses/schedule.h"

#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Scheduler::Scheduler(const Scope& s, ScopesForest& forest)
    : forest_(&forest)
    , scope_(&s)
    , cfg_(&scope().f_cfg())
    , domtree_(&cfg().domtree())
{
    std::queue<const Def*> queue;
    DefSet done;

    auto enqueue = [&](const Def* def, size_t i, const Def* op) {
        if (scope().contains(op)) {
            auto [_, ins] = def2uses_[op].emplace(i, def);
            assert_unused(ins);
            if (auto [_, ins] = done.emplace(op); ins) queue.push(op);
        }
    };

    assert(s.entry()->has_body());
    done.emplace(s.entry());
    enqueue(s.entry(), 0, s.entry()->body());

    while (!queue.empty()) {
        auto def = pop(queue);
        for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
            // all reachable continuations have already been registered above
            // NOTE we might still see references to unreachable continuations in the schedule
            enqueue(def, i, def->op(i));
        }
    }

    register_defs(s);
}

void Scheduler::register_defs(const Scope& s) {
    for (auto child : s.children_scopes()) {
        Scope& cs = forest_->get_scope(child);
        register_defs(cs);
    }

    for (auto def : s.defs()) {
        if (!early_.lookup(def))
            early_[def] = s.entry();
    }
}

Continuation* Scheduler::early(const Def* def) {
    if (auto cont = early_.lookup(def)) return *cont;
    if (auto param = def->isa<Param>()) return early_[def] = param->continuation();
    assert(false);
}

Continuation* Scheduler::late(const Def* def) {
    if (auto cont = late_.lookup(def)) return *cont;

    Continuation* result = nullptr;
    if (auto continuation = def->isa_nom<Continuation>()) {
        result = continuation;
    } else if (auto param = def->isa<Param>()) {
        result = param->continuation();
    } else if (def->isa_nom()) {
        // don't try to late-schedule recursive nodes for now
        result = early(def);
    } else {
        for (auto use : uses(def)) {
            auto cont = late(use);
            result = result ? forest_->least_common_ancestor(result, cont) : cont;
            assert(result);
        }
    }

    return late_[def] = result;
}

Continuation* Scheduler::smart(const Def* def) {
    if (auto cont = smart_.lookup(def)) return *cont;

    auto e = cfg(early(def));
    auto l = cfg(late (def));
    auto s = l;

    // if the 'early' or 'late' BB in the schedule is statically unreachable, instead emit early
    if (!e || !l)
        return early(def);

    int depth = cfg().looptree()[l]->depth();
    for (auto i = l; i != e;) {
        auto idom = domtree().idom(i);
        assert(i != idom);
        i = idom;

        if (i == nullptr) {
            scope_->world().WLOG("this should never occur - don't know where to put {}", def);
            s = l;
            break;
        }

        if (int cur_depth = cfg().looptree()[i]->depth(); cur_depth < depth) {
            s = i;
            depth = cur_depth;
        }
    }

    return smart_[def] = s->continuation();
}

static void add_scope_to_schedule(Schedule& sched, const Scope& s) {
    sched.push_back(s.entry());
    for (auto child : s.children_scopes()) {
        add_scope_to_schedule(sched, s.forest().get_scope(child));
    }
}

Schedule schedule(const Scope& scope) {
    // until we have sth better simply use the RPO of the CFG
    Schedule result;
    //for (auto n : scope.f_cfg().reverse_post_order())
    //    result.emplace_back(n->continuation());
    add_scope_to_schedule(result, scope);

    return result;
}

#if 0
void Schedule::verify() {
#if THORIN_ENABLE_CHECKS
    bool ok = true;
    auto& domtree = cfg().domtree();
    Schedule::Map<const Def*> block2mem(*this);

    for (auto& block : *this) {
        const Def* mem = block.continuation()->mem_param();
        auto idom = block.continuation() != scope().entry() ? domtree.idom(block.node()) : block.node();
        mem = mem ? mem : block2mem[(*this)[idom]];
        for (auto primop : block) {
            if (auto memop = primop->isa<MemOp>()) {
                if (memop->mem() != mem) {
                    world().WLOG("incorrect schedule: {} @ '{}'; current mem is {} @ '{}') - scope entry: {}", memop, memop->location(), mem, mem->location(), scope_.entry());
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

void verify_mem(World& world) {
    Scope::for_each(world, [&](const Scope& scope) { schedule(scope); });
}
#endif

}
