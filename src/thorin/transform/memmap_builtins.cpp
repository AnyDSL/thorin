#include "thorin/world.h"
#include "thorin/memop.h"
#include "thorin/literal.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/transform/mangle.h"

namespace thorin {

typedef std::vector<std::pair<const Type*, Use>> ToDo;

static bool map_param(World& world, Lambda* lambda, ToDo& todo) {
    // useful sanity checks
    assert(lambda->attribute().is(Lambda::Map) && "invalid map function");
    assert(lambda->params().size() == 7 && "invalid signature");
    assert(lambda->param(1)->type()->isa<Ptr>() && "invalid pointer type");

    auto uses = lambda->uses();
    if (uses.size() < 1)
        return false;
    auto ulambda = uses.begin()->def()->as_lambda();

    auto mapped = world.map(ulambda->arg(0),  // memory
                            ulambda->arg(1),  // source ptr
                            ulambda->arg(2),  // target device (0 for host device)
                            ulambda->arg(3),  // address space
                            ulambda->arg(4),  // top_left of region
                            ulambda->arg(5)); // region size
    auto cont = ulambda->arg(6)->as_lambda(); // continuation

    Scope cont_scope(cont);
    auto ncont = drop(cont_scope, { mapped->extract_mem(), mapped->extract_mapped_ptr() });
    ulambda->jump(ncont, {});
    cont->destroy_body();
    for (auto use : mapped->extract_mapped_ptr()->uses())
        todo.push_back(std::pair<const Type*, Use>(mapped->ptr_type(), use));
    return true;
}

static void adapt_addr_space(World &world, ToDo& uses) {
    auto entry = uses.back();
    auto use = entry.second;
    uses.pop_back();
    if (auto ulambda = use->isa_lambda()) {
        // we need to specialize the next lambda if the types do not match
        auto to = ulambda->to()->isa_lambda();
        if (!to || use.index() == 0) {
            // cannot handle calls to parameters right now
            THORIN_UNREACHABLE;
        }
        assert(use.index() > 0);
        auto index = use.index() - 1;
        // -> specialize for new ptr type
        if (to->param(index)->type() != entry.first) {
            Array<const Type*> pi(to->pi()->elems());
            pi[index] = entry.first;
            auto nto = world.lambda(world.pi(pi), to->name);
            assert(nto->num_params() == to->num_params());
            nto->attribute() = to->attribute();

            Scope to_scope(to);
            Array<Def> mapping(nto->num_params());
            for (size_t i = 0, e = nto->num_params(); i != e; ++i)
                mapping[i] = nto->param(i);

            auto specialized = drop(to_scope, mapping);
            nto->jump(specialized, {});
            ulambda->update_to(nto);
        }
    } else {
        auto primop = use->as<PrimOp>();
        if (primop->isa<MemOp>())
            return;
        // search downwards
        for (auto puse : primop->uses())
            uses.push_back(std::pair<const Type*, Use>(primop->type(), puse));
    }
}

void memmap_builtins(World& world) {
    ToDo todo;
    bool has_work;
    do {
        // 1) look for "mapped" lambdas
        has_work = false;
        for (auto lambda : world.copy_lambdas()) {
            if (lambda->attribute().is(Lambda::Map) && map_param(world, lambda, todo))
                has_work = true;
        }
        // 2) adapt mapped address spaces on users
        while (!todo.empty())
            adapt_addr_space(world, todo);
    } while (has_work);
    world.cleanup();
}

}
