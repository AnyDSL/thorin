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
    void clean_pe_info(std::queue<Continuation*>, Continuation*);
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
                if (use.index() != 0 || !use->isa<Continuation>()) {
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
                        auto arg = use->as_continuation()->arg(i);
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

                entry->jump(dropped, new_args);
                todo_ = true;
                scope.update();
            }
        }
    });
}

void Cleaner::eta_conversion() {
    for (bool todo = true; todo;) {
        todo = false;
        for (auto continuation : world().continuations()) {
            if (!continuation->empty()) {
                // eat calls to known continuations that are only used once
                while (auto callee = continuation->callee()->isa_continuation()) {
                    if (callee->num_uses() == 1 && !callee->empty() && !callee->is_external()) {
                        for (size_t i = 0, e = continuation->num_args(); i != e; ++i)
                            callee->param(i)->replace(continuation->arg(i));
                        continuation->jump(callee->callee(), callee->args(), callee->jump_debug());
                        callee->destroy_body();
                        todo_ = todo = true;
                    } else
                        break;
                }

                // try to subsume continuations which call a parameter
                // (that is free within that continuation) with that parameter
                if (auto param = continuation->callee()->isa<Param>()) {
                    if (param->continuation() == continuation || continuation->is_external())
                        continue;

                    if (continuation->arg() == continuation->param()) {
                        continuation->replace(continuation->callee());
                        continuation->destroy_body();
                        todo_ = todo = true;
                        continue;
                    }

                    // build the permutation of the arguments
                    Array<size_t> perm(continuation->num_args());
                    bool is_permutation = true;
                    for (size_t i = 0, e = continuation->num_args(); i != e; ++i)  {
                        auto param_it = std::find(continuation->params().begin(),
                                                    continuation->params().end(),
                                                    continuation->arg(i));

                        if (param_it == continuation->params().end()) {
                            is_permutation = false;
                            break;
                        }

                        perm[i] = param_it - continuation->params().begin();
                    }

                    if (!is_permutation) continue;

                    // for every use of the continuation at a call site,
                    // permute the arguments and call the parameter instead
                    for (auto use : continuation->copy_uses()) {
                        auto ucontinuation = use->isa_continuation();
                        if (ucontinuation && use.index() == 0) {
                            Array<const Def*> new_args(perm.size());
                            for (size_t i = 0, e = perm.size(); i != e; ++i) {
                                new_args[i] = ucontinuation->arg(perm[i]);
                            }
                            ucontinuation->jump(param, new_args, ucontinuation->jump_debug());
                            todo_ = todo = true;
                        }
                    }
                }
            }
        }
    }
}

void Cleaner::eliminate_params() {
    for (auto ocontinuation : world().copy_continuations()) {
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;

        if (!ocontinuation->empty() && !world().is_external(ocontinuation)) {
            for (auto use : ocontinuation->uses()) {
                if (use.index() != 0 || !use->isa_continuation())
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
                auto old_domain = ocontinuation->type()->domain();
                const Type* new_domain;
                if (auto tuple_type = old_domain->isa<TupleType>())
                    new_domain = world().tuple_type(tuple_type->ops().cut(proxy_idx));
                else {
                    assert(proxy_idx.size() == 1 && proxy_idx[0] == 0);
                    new_domain = world().tuple_type({});
                }
                auto fn_type = world().fn_type(new_domain);
                auto ncontinuation = world().continuation(fn_type, ocontinuation->cc(), ocontinuation->intrinsic(), ocontinuation->debug_history());
                size_t j = 0;
                for (auto i : param_idx) {
                    ocontinuation->param(i)->replace(ncontinuation->param(j));
                    ncontinuation->param(j++)->debug() = ocontinuation->param(i)->debug_history();
                }

                if (ocontinuation->filter() != nullptr) {
                    Array<const Def*> new_filter(param_idx.size());
                    size_t i = 0;
                    for (auto j : param_idx)
                        new_filter[i++] = ocontinuation->filter(param_idx[j]);

                    ncontinuation->set_filter(world().tuple(new_filter));
                }
                ncontinuation->jump(ocontinuation->callee(), ocontinuation->args(), ocontinuation->jump_debug());
                ocontinuation->destroy_body();

                for (auto use : ocontinuation->copy_uses()) {
                    auto ucontinuation = use->as_continuation();
                    assert(use.index() == 0);
                    ucontinuation->jump(ncontinuation, ucontinuation->args().cut(proxy_idx), ucontinuation->jump_debug());
                }

                todo_ = true;
            }
        }
next_continuation:;
    }
}

void Cleaner::rebuild() {
    Importer importer(world_);
    importer.type_old2new_.rehash(world_.types_.capacity());
    importer.def_old2new_.rehash(world_.primops().capacity());

#if THORIN_ENABLE_CHECKS
    world_.swap_breakpoints(importer.world());
#endif

    for (auto external : world().externals())
        importer.import(external);

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

    for (auto primop : world().primops())
        check(primop);
    for (auto continuation : world().continuations()) {
        check(continuation);
        check(continuation->param());
    }
}

void Cleaner::within(const Def* def) {
    assert(world().types().contains(def->type()));
    if (auto primop = def->isa<PrimOp>())
        assert_unused(world().primops().contains(primop));
    else if (auto continuation = def->isa_continuation())
        assert_unused(world().continuations().contains(continuation));
    else
        within(def->as<Param>()->continuation());
}

void Cleaner::clean_pe_info(std::queue<Continuation*> queue, Continuation* cur) {
    assert(cur->arg(1)->type() == world().ptr_type(world().indefinite_array_type(world().type_pu8())));
    auto next = cur->arg(3);
    auto msg = cur->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<DefiniteArray>();

    assert(!is_const(cur->arg(2)));
    IDEF(cur->callee(), "pe_info not constant: {}: {}", msg->as_string(), cur->arg(2));
    cur->jump(next, {cur->arg(0)}, cur->jump_debug());
    todo_ = true;

    // always re-insert into queue because we've changed cur's jump
    queue.push(cur);
}

void Cleaner::clean_pe_infos() {
    VLOG("cleaning remaining pe_infos");
    std::queue<Continuation*> queue;
    ContinuationSet done;
    auto enqueue = [&](Continuation* continuation) {
        if (done.emplace(continuation).second)
            queue.push(continuation);
    };
    for (auto external : world().externals()) {
        enqueue(external);
    }

    while (!queue.empty()) {
        auto continuation = pop(queue);

        if (auto callee = continuation->callee()->isa_continuation()) {
            if (callee->intrinsic() == Intrinsic::PeInfo) {
                clean_pe_info(queue, continuation);
                continue;
            }
        }

        for (auto succ : continuation->succs())
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
        for (auto continuation : world().continuations())
            continuation->destroy_filter();
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
