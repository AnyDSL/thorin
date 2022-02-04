#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"

namespace thorin {

static void dead_load_opt(const Scope& scope) {
    auto& world = scope.world();
    for (auto n : scope.f_cfg().post_order()) {
        auto continuation = n->lambda();
        if (!continuation->has_body()) continue;

        const Def* mem;
        for (auto arg : continuation->body()->args()) {
            if (is_mem(arg)) {
                mem = arg;
                break;
            }
        }

        if (mem) {
            while (true) {
                if (auto memop = mem->isa<MemOp>()) {
                    if (memop->isa<Load>() || memop->isa<Enter>()) {
                        if (memop->out(1)->num_uses() == 0)
                            memop->replace_uses(world.tuple({ memop->mem(), world.bottom(memop->out(1)->type()) }));
                    }
                    mem = memop->mem();
                } else if (auto extract = mem->isa<Extract>()) {
                    mem = extract->agg();
                } else
                    break;
            }
        }
    }
}

void dead_load_opt(World& world) {
    Scope::for_each(world, [&] (const Scope& scope) { dead_load_opt(scope); });
}

}
