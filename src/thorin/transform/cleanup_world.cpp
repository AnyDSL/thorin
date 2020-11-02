#include "thorin/config.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/importer.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/resolve_loads.h"
#include "thorin/transform/partial_evaluation.h"
#include "thorin/util/log.h"

namespace thorin {

class Cleaner {
public:
    Cleaner(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
    void cleanup();
    void eliminate_tail_rec();
    void eta_conversion();
    void eliminate_params();
    void rebuild();
    void verify_closedness();
    void within(const Def*);
    void clean_pe_infos();

private:
    void cleanup_fix_point();
    void clean_pe_info(std::queue<Lam*>, Lam*);
    World& world_;
    bool todo_ = true;
};

void Cleaner::eliminate_tail_rec() {
    Scope::for_each(world_, [&](Scope& scope) {
        auto entry = scope.entry();

        bool only_tail_calls = true;
        bool recursive = false;
        for (auto use : entry->uses()) {
            if (scope.contains(use)) {
                if (use.index() != 0 || !use->isa<Lam>()) {
                    only_tail_calls = false;
                    break;
                } else {
                    recursive = true;
                }
            }
        }

        if (recursive && only_tail_calls) {
            auto n = entry->num_params();
            Array<const Def*> args(n);

            for (size_t i = 0; i != n; ++i) {
                args[i] = entry->param(i);

                for (auto use : entry->uses()) {
                    if (scope.contains(use)) {
                        auto arg = use->as_lam()->app()->arg(i);
                        if (!arg->isa<Bottom>() && arg != args[i]) {
                            args[i] = nullptr;
                            break;
                        }
                    }
                }
            }

            std::vector<const Def*> new_args;

            for (size_t i = 0; i != n; ++i) {
                if (args[i] == nullptr) {
                    new_args.emplace_back(entry->param(i));
                    if (entry->param(i)->order() != 0) {
                        // the resulting function wouldn't be of first order so examine next scope
                        return;
                    }
                }
            }

            if (new_args.size() != n) {
                DLOG("tail recursive: {}", entry);
                auto dropped = drop(scope, args);

                entry->app(dropped, new_args);
                todo_ = true;
                scope.update();
            }
        }
    });
}

void Cleaner::eta_conversion() {
    for (bool todo = true; todo;) {
        todo = false;
        for (auto lam : world().copy_lams()) {
            if (lam->is_empty())
                continue;
            // Perform eta-conversion
            //
            // f = lam (x, y, z)
            //   app g (x, (z, y))
            //
            // if g is from a parameter or a lambda with only one use, any call to f can be inlined,
            // which results in a new app with the arguments automatically permuted. Special care has
            // to be taken when f is recursive (even if it has only one use, that use can be its own body).
            auto callee_lam = lam->app()->callee()->isa_lam();
            auto callee_param = is_from_param(lam->app()->callee());
            auto arg_param    = is_from_param(lam->app()->arg());
            if ((callee_lam && callee_lam->num_uses() == 1 && callee_lam != lam && !callee_lam->is_exported()) ||
                (callee_param && arg_param && callee_param->lam() == lam && arg_param->lam() == lam)) {
                for (auto use : lam->copy_uses()) {
                    auto ulam = use->isa_lam();
                    if (ulam && use.index() == 0) {
                        ulam->set_body(drop(ulam->app()));
                        todo_ = todo = true;
                    }
                }
            }
        }
    }
}

void Cleaner::eliminate_params() {
    for (auto olam : world().copy_lams()) {
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;

        if (!olam->is_empty() && !olam->is_exported()) {
            for (auto use : olam->uses()) {
                if (use.index() != 0 || !use->isa_lam())
                    goto next_lam;
            }

            for (size_t i = 0, e = olam->num_params(); i != e; ++i) {
                auto param = olam->param(i);
                if (param->num_uses() == 0)
                    proxy_idx.push_back(i);
                else
                    param_idx.push_back(i);
            }

            if (!proxy_idx.empty()) {
                auto cn = world().cn(olam->type()->domains().cut(proxy_idx));
                auto nlam = world().lam(cn, olam->attributes(), olam->debug_history());
                size_t j = 0;
                for (auto i : param_idx) {
                    olam->param(i)->replace(nlam->param(j));
                    nlam->param(j++)->debug() = olam->param(i)->debug_history();
                }

                if (olam->filter() != nullptr) {
                    Array<const Def*> new_filter(param_idx.size());
                    size_t i = 0;
                    for (auto j : param_idx)
                        new_filter[i++] = olam->filter(param_idx[j]);

                    nlam->set_filter(world().tuple(new_filter));
                }
                nlam->app(olam->app()->callee(), olam->app()->args(), olam->app()->debug());
                olam->destroy_body();

                for (auto use : olam->copy_uses()) {
                    auto ulam = use->as_lam();
                    assert(use.index() == 0);
                    ulam->app(nlam, ulam->app()->args().cut(proxy_idx), ulam->app()->debug());
                }

                todo_ = true;
            }
        }
next_lam:;
    }
}

void Cleaner::rebuild() {
    Importer importer(world_);
    importer.type_old2new_.rehash(world_.types_.capacity());
    importer.def_old2new_.rehash(world_.defs().capacity());

#if THORIN_ENABLE_CHECKS
    world_.swap_breakpoints(importer.world());
#endif

    for (auto continuation : world().exported_lams())
        importer.import(continuation);

    swap(importer.world(), world_);
    todo_ |= importer.todo();
}

void Cleaner::verify_closedness() {
    auto check = [&](const Def* def) {
        size_t i = 0;
        for (auto op : def->ops()) {
            within(op);
            assert_unused(op->uses_.contains(Use(i++, def)) && "can't find def in op's uses");
        }

        for (const auto& use : def->uses_) {
            within(use);
            assert(use->op(use.index()) == def && "use doesn't point to def");
        }
    };

    for (auto def : world().defs())
        check(def);
}

void Cleaner::within(const Def* def) {
    assert(world().types().contains(def->type()));
    assert_unused(world().defs().contains(def));
}

void Cleaner::clean_pe_info(std::queue<Lam*> queue, Lam* cur) {
    auto app = cur->app();
    assert(app->arg(1)->type() == world().ptr_type(world().indefinite_array_type(world().type_pu8())));
    auto next = app->arg(3);
    auto msg = app->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<DefiniteArray>();

    IDEF(app->callee(), "pe_info was not constant: {}: {}", msg->as_string(), app->arg(2));
    cur->app(next, {app->arg(0)}, app->debug());
    todo_ = true;

    // always re-insert into queue because we've changed cur's jump
    queue.push(cur);
}

void Cleaner::clean_pe_infos() {
    VLOG("cleaning remaining pe_infos");
    std::queue<Lam*> queue;
    LamSet done;
    auto enqueue = [&](Lam* lam) {
        if (done.emplace(lam).second)
            queue.push(lam);
    };

    for (auto continuation : world().exported_lams())
        enqueue(continuation);

    while (!queue.empty()) {
        auto lam = pop(queue);

        if (auto app = lam->app()) {
            if (auto callee = app->callee()->isa_lam()) {
                if (callee->intrinsic() == Intrinsic::PeInfo) {
                    clean_pe_info(queue, lam);
                    continue;
                }
            }
        }

        for (auto succ : lam->succs())
            enqueue(succ);
    }
}

void Cleaner::cleanup_fix_point() {
    int i = 0;
    for (; todo_; ++i) {
        VLOG("iteration: {}", i);
        todo_ = false;
        if (world_.is_pe_done())
            eliminate_tail_rec();
        eta_conversion();
        eliminate_params();
        rebuild(); // resolve replaced defs before going to resolve_loads
        todo_ |= resolve_loads(world());
        rebuild();
        if (!world().is_pe_done())
            todo_ |= partial_evaluation(world_);
        else
            clean_pe_infos();
    }
}

void Cleaner::cleanup() {
    VLOG("start cleanup");
    cleanup_fix_point();

    if (!world().is_pe_done()) {
        world().mark_pe_done();
        //for (auto lam : world().lams())
            //lam->destroy_filter();
        todo_ = true;
        cleanup_fix_point();
    }

    VLOG("end cleanup");
#if THORIN_ENABLE_CHECKS
    verify_closedness();
    debug_verify(world());
#endif
}

void cleanup_world(World& world) { Cleaner(world).cleanup(); }

}
