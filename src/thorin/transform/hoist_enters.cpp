#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"

namespace thorin {

static void find_enters(std::deque<const Enter*>& enters, const Def* def) {
    if (auto enter = def->isa<Enter>())
        enters.push_back(enter);

    if (auto memop = def->isa<MemOp>())
        def = memop->out_mem();

    for (auto use : def->uses()) {
        if (auto memop = use->isa<MemOp>())
            find_enters(enters, memop);
    }
}

static void find_enters(std::deque<const Enter*>& enters, Continuation* continuation) {
    if (auto mem_param = continuation->mem_param())
        find_enters(enters, mem_param);
}

static bool hoist_enters(const Scope& scope) {
    World& world = scope.world();
    std::deque<const Enter*> enters;

    for (auto n : scope.f_cfg().reverse_post_order())
        find_enters(enters, n->continuation());

    if (enters.empty() || enters[0]->mem() != scope.entry()->mem_param()) {
        world.VLOG("cannot optimize {} - didn't find entry enter", scope.entry());
        return false;
    }

    auto entry_enter = enters[0];
    auto frame = entry_enter->out_frame();
    enters.pop_front();

    bool todo = false;
    for (auto i = enters.rbegin(), e = enters.rend(); i != e; ++i) {
        auto old_enter = *i;
        for (auto use : old_enter->out_frame()->uses()) {
            auto slot = use->as<Slot>();
            if (slot->uses().size() > 0)
                todo = true;
            slot->replace_uses(world.slot(slot->alloced_type(), frame, slot->debug()));
            assert(slot->num_uses() == 0);
        }
    }
    return todo;
}

void hoist_enters(Thorin& thorin) {
    bool todo = false;
    do {
        todo = false;
        //Scope::for_each(thorin.world(), [&](const Scope& scope) { hoist_enters(scope); });
        //Scope::for_each(thorin.world(), [&](const Scope& scope) { if (!todo) todo = hoist_enters(scope); });
        ScopesForest forest(thorin.world());
        for (auto cont : thorin.world().copy_continuations()) {
            if (!cont->has_body())
                continue;
            assert(forest.stack_.empty());
            //forest->scopes_.clear();
            auto& scope = forest.get_scope(cont);
            //Scope scope(cont);
            assert(forest.stack_.empty());
            if(!scope.has_free_params()) {
                assert(forest.stack_.empty());
                if (!todo) todo = hoist_enters(scope);
                if (todo)
                    break;
            }
        }
    } while (todo);
    thorin.cleanup();
}

}
