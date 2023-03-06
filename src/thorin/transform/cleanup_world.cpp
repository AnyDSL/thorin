#include "thorin/config.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/importer.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/resolve_loads.h"
#include "thorin/transform/partial_evaluation.h"

namespace thorin {

class Cleaner {
public:
    Cleaner(Thorin& thorin)
        : thorin_(thorin)
    {}

    World& world() { return thorin_.world(); }
    void cleanup();
    void eliminate_tail_rec();
    void eliminate_params();
    void rebuild();
    void verify_closedness();
    void within(const Def*);
    void clean_pe_infos();

private:
    void cleanup_fix_point();
    void clean_pe_info(std::queue<Continuation*>, Continuation*);
    Thorin& thorin_;
    bool todo_ = true;
};

void Cleaner::eliminate_tail_rec() {
    Scope::for_each(world(), [&](Scope& scope) {
        auto entry = scope.entry();

        bool only_tail_calls = true;
        bool recursive = false;
        for (auto use : entry->uses()) {
            if (scope.contains(use)) {
                if (use.index() == App::CALLEE_POSITION && use->isa<App>()) {
                    recursive = true;
                    continue;
                } else if (use->isa<Param>())
                    continue; // ignore params

                world().ELOG("non-recursive usage of {} index:{} use:{}", scope.entry()->name(), use.index(), use.def()->to_string());
                only_tail_calls = false;
                break;
            }
        }

        if (recursive && only_tail_calls) {
            auto n = entry->num_params();
            Array<const Def*> args(n);

            for (size_t i = 0; i != n; ++i) {
                args[i] = entry->param(i);

                for (auto use : entry->uses()) {
                    if (scope.contains(use.def())) {
                        auto app = use->isa<App>();
                        if (!app) continue;
                        auto arg = app->arg(i);
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
                world().DLOG("tail recursive: {}", entry);
                auto dropped = drop(scope, args);

                entry->jump(dropped, new_args);
                todo_ = true;
                scope.update();
            }
        }
    });
}

void Cleaner::eliminate_params() {
    // TODO
    for (auto ocontinuation : world().copy_continuations()) {
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;

        if (ocontinuation->has_body() && !world().is_external(ocontinuation)) {
            auto obody = ocontinuation->body();
            for (auto use : ocontinuation->uses()) {
                if (use.index() != App::CALLEE_POSITION || !use->isa_nom<Continuation>())
                    goto next_continuation;
            }

            for (size_t i = 0, e = ocontinuation->num_params(); i != e; ++i) {
                auto param = ocontinuation->param(i);
                if (param->num_uses() == 0)
                    proxy_idx.push_back(i);
                else
                    param_idx.push_back(i);
            }

            if (!proxy_idx.empty()) {
                auto ncontinuation = world().continuation(
                    world().fn_type(ocontinuation->type()->types().cut(proxy_idx)),
                    ocontinuation->attributes(), ocontinuation->debug_history());
                size_t j = 0;
                for (auto i : param_idx) {
                    ocontinuation->param(i)->replace_uses(ncontinuation->param(j));
                    ncontinuation->param(j++)->set_name(ocontinuation->param(i)->debug_history().name);
                }

                if (!ocontinuation->filter()->is_empty())
                    ncontinuation->set_filter(ocontinuation->filter()->cut(proxy_idx));
                ncontinuation->jump(obody->callee(), obody->args(), ocontinuation->debug());
                ncontinuation->verify();
                ocontinuation->destroy("cleanup: calls a parameter (permutated)");

                for (auto use : ocontinuation->copy_uses()) {
                    auto uapp = use->as<App>();
                    assert(use.index() == App::CALLEE_POSITION);
                    for (auto ucontinuation : uapp->using_continuations()) {
                        ucontinuation->jump(ncontinuation, uapp->args().cut(proxy_idx), ucontinuation->debug());
                    }
                }

                todo_ = true;
            }
        }
next_continuation:;
    }
}

void Cleaner::rebuild() {
    auto fresh_world = std::make_unique<World>(world());
    Importer importer(world(), *fresh_world);

    for (auto&& [_, def] : world().externals()) {
        if (auto cont = def->isa<Continuation>(); cont && cont->is_exported())
            importer.import(cont);
        if (auto global = def->isa<Global>(); global && global->is_external())
            importer.import(global);
    }

    std::swap(thorin_.world_container(), fresh_world);

    // verify(world());

    todo_ |= importer.todo();
}

void Cleaner::verify_closedness() {
    auto check = [&](const Def* def) {
        size_t i = 0;
        for (auto op : def->ops()) {
            within(op);
            assert_unused(op->uses().contains(Use(i++, def)) && "can't find def in op's uses");
        }

        for (const auto& use : def->uses()) {
            within(use);
            assert(use->op(use.index()) == def && "use doesn't point to def");
        }
    };

    for (auto def : world().defs())
        check(def);
}

void Cleaner::within(const Def* def) {
    assert(&def->type()->world() == &world());
    assert_unused(world().defs().contains(def));
}

void Cleaner::clean_pe_info(std::queue<Continuation*> queue, Continuation* cur) {
    assert(cur->has_body());
    auto body = cur->body();
    assert(body->arg(1)->type() == world().ptr_type(world().indefinite_array_type(world().type_pu8())));
    auto next = body->arg(3);
    auto msg = body->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<DefiniteArray>();

    world().idef(body->callee(), "pe_info was not constant: {}: {}", msg->as_string(), body->arg(2));
    cur->jump(next, {body->arg(0)}, cur->debug()); // TODO debug
    todo_ = true;

    // always re-insert into queue because we've changed cur's jump
    queue.push(cur);
}

void Cleaner::clean_pe_infos() {
    world().VLOG("cleaning remaining pe_infos");
    std::queue<Continuation*> queue;
    ContinuationSet done;
    auto enqueue = [&](Continuation* continuation) {
        if (done.emplace(continuation).second)
            queue.push(continuation);
    };

    for (auto&& [_, def] : world().externals())
        if (auto cont = def->isa<Continuation>(); cont && cont->has_body()) enqueue(cont);

    while (!queue.empty()) {
        auto continuation = pop(queue);

        if (continuation->has_body()) {
            if (auto body = continuation->body()->isa<App>()) {
                if (auto callee = body->callee()->isa_nom<Continuation>(); callee && callee->intrinsic() == Intrinsic::PeInfo) {
                    clean_pe_info(queue, continuation);
                    continue;
                }
            }
        }

        for (auto succ : continuation->succs())
            enqueue(succ);
    }
}

void Cleaner::cleanup_fix_point() {
    int i = 0;
    for (; todo_; ++i) {
        world().VLOG("iteration: {}", i);
        todo_ = false;
        //if (world().is_pe_done())
        rebuild();
            eliminate_tail_rec();
        eliminate_params();
        rebuild(); // resolve replaced defs before going to resolve_loads
        todo_ |= resolve_loads(world());
        rebuild();
        //if (!world().is_pe_done())
            todo_ |= partial_evaluation(world());
        //else
        //    clean_pe_infos();
    }
}

void Cleaner::cleanup() {
    world().VLOG("start cleanup");
    cleanup_fix_point();

    if (!world().is_pe_done()) {
        world().mark_pe_done();
        for (auto def : world().defs()) {
            if (auto cont = def->isa_nom<Continuation>())
                cont->destroy_filter();
        }

        todo_ = true;
        cleanup_fix_point();
    }

    world().VLOG("end cleanup");
#if THORIN_ENABLE_CHECKS
    verify_closedness();
    debug_verify(world());
#endif
}

void Thorin::cleanup() { Cleaner(*this).cleanup(); }

}
