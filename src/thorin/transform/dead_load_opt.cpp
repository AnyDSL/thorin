#include "thorin/primop.h"
#include "thorin/analyses/scope.h"

namespace thorin {

static void dead_load_opt(const Scope& scope) {
    for (size_t i = scope.size(); i-- != 0;) {
        auto lambda = scope.rpo(i);
        Def mem;
        for (auto arg : lambda->args()) {
            if (arg->is_mem()) {
                mem = arg;
                break;
            }
        }

        while (true) {
            if (auto memop = mem->isa<MemOp>()) {
                if (memop->isa<Load>() || memop->isa<Enter>()) {
                    if (memop->out(1)->num_uses() == 0)
                        memop->out_mem()->replace(memop->mem());
                }
                mem = memop->mem();
            } else if (auto extract = mem->isa<Extract>()) {
                mem = extract->agg();
            } else
                break;
        }
    }
}

void dead_load_opt(World& world) {
    Scope::for_each(world, [&] (const Scope& scope) {
            dead_load_opt(scope);
    });
}

}
