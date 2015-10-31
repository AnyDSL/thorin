#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

void lower2cff(World& world) {
    HashMap<Array<Def>, Lambda*> cache;
    LambdaSet top;

    bool local = true;
    for (bool todo = true; todo || local;) {
        todo = false;

        Scope::for_each(world, [&] (Scope& scope) {
            bool dirty = false;

            auto is_bad = [&] (Lambda* to) {
                if (to->empty())
                    return false;
                if (local)
                    return scope.inner_contains(to) && !to->is_basicblock();
                else {
                    if (top.contains(to))
                        return !to->is_returning() && !scope.outer_contains(to);
                    else
                        return !to->is_basicblock();
                }
            };

            const auto& cfg = scope.f_cfg();
            for (auto n : cfg.post_order()) {
                auto lambda = n->lambda();
                if (lambda->gid() == 25521)
                    lambda->dump();
                if (auto to = lambda->to()->isa_lambda()) {
                    if (is_bad(to)) {
                        DLOG("bad: %", to->unique_name());
                        todo = true;
                        dirty = true;
                        Type2Type map;
                        bool res = to->type()->infer_with(map, lambda->arg_fn_type());
                        assert(res);

                        Array<Def> ops(lambda->size());
                        ops[0] = to;
                        for (size_t i = 1, e = ops.size(); i != e; ++i)
                            ops[i] = to->param(i-1)->order() > 0 ? lambda->arg(i-1) : nullptr;

                        auto p = cache.emplace(ops, nullptr);
                        Lambda*& target = p.first->second;
                        if (p.second) {
                            Scope to_scope(to);
                            target = drop(to_scope, ops.skip_front(), map); // use already dropped version as target
                        }

                        std::vector<Def> nargs;
                        size_t i = 0;
                        for (auto arg : ops.skip_front()) {
                            if (arg == nullptr)
                                nargs.push_back(lambda->arg(i));
                            ++i;
                        }

                        lambda->jump(target, nargs);
                        assert(lambda->arg_fn_type() == target->type());
                    }
                }
            }

            if (dirty)
                scope.update();
            top.insert(scope.entry());
        });

        if (!todo && local) {
            DLOG("switching to global mode");
            local = false;
            todo = true;
        }
    }

    debug_verify(world);
    world.cleanup();
}

}
