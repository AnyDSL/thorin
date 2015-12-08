#include "thorin/world.h"
#include "thorin/primop.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

typedef std::vector<std::pair<Type, Use>> ToDo;

static bool map_param(World& world, Lambda* lambda, ToDo& todo) {
    assert(lambda->intrinsic() == Intrinsic::Mmap && "invalid mmap function");
    assert(lambda->param(1)->type().isa<PtrType>() && "invalid pointer type");

    auto uses = lambda->uses();
    if (uses.size() < 1)
        return false;
    auto ulambda = uses.begin()->def()->as_lambda();

    auto cont = ulambda->arg(6)->as_lambda(); // continuation
    Scope cont_scope(cont);
    Lambda* ncont;

    auto map = world.map(ulambda->arg(2), // target device (-1 for host device)
                         ulambda->arg(3), // address space
                         ulambda->arg(0), // memory
                         ulambda->arg(1), // source ptr
                         ulambda->arg(4), // offset to memory
                         ulambda->arg(5), // size of memory
                         ulambda->loc());
    ncont = drop(cont_scope, {}, { map->out_mem(), map->out_ptr() });

    ulambda->jump(ncont, {}, {});
    cont->destroy_body();
    for (auto use : map->out_ptr()->uses())
        todo.emplace_back(map->out_ptr_type(), use);
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
            Array<Type> fn(to->type()->num_args());
            for (size_t i = 0, e = to->type()->num_args(); i != e; ++i) {
                if (i==index) fn[i] = entry.first;
                else fn[i] = to->type()->arg(i);
            }
            auto nto = world.lambda(world.fn_type(fn), to->loc(), to->cc(), to->intrinsic(), to->name);
            assert(nto->num_params() == to->num_params());

            if (!to->empty()) {
                Scope to_scope(to);
                Array<Def> mapping(nto->num_params());
                for (size_t i = 0, e = nto->num_params(); i != e; ++i)
                    mapping[i] = nto->param(i);

                auto specialized = drop(to_scope, {}, mapping);
                nto->jump(specialized, {}, {});
            }
            ulambda->update_to(nto);
        }
    } else {
        auto primop = use->as<PrimOp>();
        if (primop->isa<MemOp>())
            return;
        // search downwards
        for (auto puse : primop->uses())
            uses.emplace_back(primop->type(), puse);
    }
}

void memmap_builtins(World& world) {
    ToDo todo;
    bool has_work;
    do {
        // 1) look for "mapped" lambdas
        has_work = false;
        for (auto lambda : world.copy_lambdas()) {
            if (lambda->intrinsic() == Intrinsic::Mmap && map_param(world, lambda, todo))
                has_work = true;
        }
        // 2) adapt mapped address spaces on users
        while (!todo.empty())
            adapt_addr_space(world, todo);
    } while (has_work);
    world.cleanup();
}

}
