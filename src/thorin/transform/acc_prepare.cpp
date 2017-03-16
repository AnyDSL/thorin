#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/mangle.h"

namespace thorin {

static void force_inline(Scope& scope, int threshold) {
    for (bool todo = true; todo && threshold-- != 0;) {
        todo = false;
        for (auto n : scope.f_cfg().post_order()) {
            auto continuation = n->continuation();
            if (auto callee = continuation->callee()->isa_continuation()) {
                if (!callee->empty() && !scope.contains(callee)) {
                    Scope callee_scope(callee);
                    continuation->jump(drop(callee_scope, continuation->args()), {}, continuation->jump_debug());
                    todo = true;
                }
            }
        }

        if (todo)
            scope.update();
    }

    for (auto n : scope.f_cfg().reverse_post_order()) {
        auto continuation = n->continuation();
        if (auto callee = continuation->callee()->isa_continuation()) {
            if (!callee->empty() && !scope.contains(callee))
                WLOG(callee, "couldn't inline {} at {}", scope.entry(), continuation->jump_location());
        }
    }
}

void acc_prepare(World& world) {
    std::vector<Continuation*> todo;
    ContinuationSet do_force_inline;

    Scope::for_each(world, [&] (const Scope& scope) {
        for (auto n : scope.f_cfg().post_order()) {
            auto continuation = n->continuation();
            if (continuation->is_passed_to_accelerator() && !continuation->is_basicblock()) {
                todo.push_back(continuation);
                if (continuation->is_passed_to_intrinsic(Intrinsic::Vectorize))
                    do_force_inline.emplace(continuation);
            }
        }
    });

    for (auto continuation : todo) {
        Scope scope(continuation);
        bool first = true;
        for (auto use : continuation->copy_uses()) {
            if (first) {
                first = false; // re-use the initial continuation as first clone
            } else {
                auto ncontinuation = clone(scope);
                if (auto ucontinuation = use->isa_continuation()) {
                    ucontinuation->update_op(use.index(), ncontinuation);

                    if (auto callee = ucontinuation->callee()->isa_continuation()) {
                        if (callee->is_accelerator()) {
                            todo.emplace_back(ncontinuation);
                            if (do_force_inline.contains(continuation))
                                do_force_inline.emplace(ncontinuation);
                        }
                    }
                } else {
                    auto primop = use->as<PrimOp>();
                    Array<const Def*> nops(primop->num_ops());
                    std::copy(primop->ops().begin(), primop->ops().end(), nops.begin());
                    nops[use.index()] = ncontinuation;
                    primop->replace(primop->rebuild(nops));
                }
            }
        }
    }

    static const int inline_threshold = 10;
    for (auto continuation : do_force_inline) {
        Scope scope(continuation);
        force_inline(scope, inline_threshold);
    }

    for (auto cur : todo) {
        Scope scope(cur);

        // remove all continuations - they should be top-level functions and can thus be ignored
        std::vector<const Def*> defs;
        for (auto param : free_defs(scope)) {
            if (!param->isa_continuation()) {
                assert(param->order() == 0 && "creating a higher-order function");
                defs.push_back(param);
            }
        }

        auto lifted = lift(scope, defs);

        std::vector<Use> uses(cur->uses().begin(), cur->uses().end()); // TODO rewrite this
        for (auto use : uses) {
            if (auto ucontinuation = use->isa_continuation()) {
                if (auto callee = ucontinuation->callee()->isa_continuation()) {
                    if (callee->is_intrinsic()) {
                        auto old_ops = ucontinuation->ops();
                        Array<const Def*> new_ops(old_ops.size() + defs.size());
                        std::copy(defs.begin(), defs.end(), std::copy(old_ops.begin(), old_ops.end(), new_ops.begin()));    // old ops + former free defs
                        assert(old_ops[use.index()] == cur);
                        new_ops[use.index()] = world.global(lifted, false, lifted->debug());                                // update to new lifted continuation
                        ucontinuation->jump(cur, new_ops.skip_front(), ucontinuation->jump_debug());                        // set new args

                        // jump to new top-level dummy function
                        auto ncontinuation = world.continuation(ucontinuation->arg_fn_type(), callee->cc(), callee->intrinsic(), callee->debug());
                        ucontinuation->update_callee(ncontinuation);
                    }
                }
            }
        }
    }
}

}
