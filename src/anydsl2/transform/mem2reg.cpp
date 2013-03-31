#include "anydsl2/memop.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/rootlambdas.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/transform/cfg_builder.h"
#include "anydsl2/transform/inliner.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

void mem2reg(World& world) {
    return;
    for_all (root, find_root_lambdas(world)) {
        Scope scope(root);

        size_t pass = world.new_pass();

        for_all (lambda, scope.rpo()) {
            for_all (param, lambda->params())
                param->visit_first(pass);
            const Def* def = lambda->mem_param();
            while (def) {
                if (const Store* store = def->isa<Store>()) {
                    store->dump();
                    if (const Slot* slot = store->ptr()->isa<Slot>()) {
                        std::cout << "asdf" << std::endl;
                        lambda->set_value(slot->index(), store->val());
                        store->replace(store->mem());
                    }
                } else if (const Load* load = def->isa<Load>()) {
                    def = load->extract_mem();

                    if (const Slot* slot = load->ptr()->isa<Slot>()) {
                        const Type* type = slot->type()->as<Ptr>()->ref();
                        load->extract_val()->replace(lambda->get_value(slot->index(), type, slot->name.c_str()));
                        load->extract_mem()->replace(load->mem());
                    }
                } else if (const Enter* enter = def->isa<Enter>()) {
                    def = enter->extract_mem();
                } else if (const CCall* ccall = def->isa<CCall>()) {
                    def = ccall->extract_mem();
                }

                size_t num = 0;
                for_all (use, def->uses())
                    if (use->isa<PrimOp>())
                        ++num;

                assert(num <= 1);

                for_all (use, def->uses()) {
                    if (use->isa<PrimOp>()) {
                        def = use;
                        goto out;
                    }
                }
                def = 0;
out:;
            }
        }
    }

    for_all (lambda, world.lambdas())
        lambda->clear();
}

}
