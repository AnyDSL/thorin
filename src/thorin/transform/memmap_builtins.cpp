#include "thorin/world.h"
#include "thorin/memop.h"
#include "thorin/literal.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/transform/mangle.h"
#include "thorin/be/thorin.h"

#include <iostream>

namespace thorin {

static void map_param(World& world, Lambda* lambda, std::vector<Lambda*>& todo) {
    // useful sanity checks
    assert(lambda->attribute().is(Lambda::Map) && "invalid map function");
    assert(lambda->params().size() == 4 && "invalid signature");
    assert(lambda->param(1)->type()->isa<Ptr>() && "invalid pointer type");

    for (auto use : lambda->uses()) {
        auto ulambda = use->as_lambda();
        auto space = (AddressSpace)ulambda->arg(2)->as<PrimLit>()->ps32_value().data();
        auto cont = ulambda->arg(3)->as_lambda();
        auto ptr_param = cont->param(1);

        auto mapped = world.map(ulambda->arg(0), ulambda->arg(1), space);
        cont->param(1)->replace(mapped->extract_mapped_ptr());
        std::vector<Def> nargs;
        nargs.push_back(mapped->extract_mem());
        for (size_t i = 1, e = cont->num_args(); i != e; ++i) {
            auto arg = cont->arg(i);
            if (arg != ptr_param)
                nargs.push_back(arg);
        }
        ulambda->jump(cont->to(), nargs);
        cont->destroy_body();
        for (auto use : mapped->extract_mapped_ptr()->uses())
            if (auto lambda = use->isa_lambda())
                todo.push_back(lambda);
    }
}

void memmap_builtins(World& world) {
    // 1) look for "mapped" lambdas
    std::vector<Lambda*> todo;
    for (auto lambda : world.copy_lambdas()) {
        if (lambda->attribute().is(Lambda::Map))
            map_param(world, lambda, todo);
    }
    world.cleanup();
    emit_thorin(world);

    // 2) adapt mapped address spaces
}

}
