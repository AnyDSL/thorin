#include <unordered_map>

#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/be/air.h"
#include "anydsl2/analyses/schedule.h"
#include "anydsl2/analyses/looptree.h"
#include "anydsl2/transform/mangle.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

static Lambda* cached(World& world, const Call& call) {
    auto iter = world.cache_.find(call);
    if (iter != world.cache_.end()) {
        return iter->second;
    }
    return nullptr;
}

void partial_evaluation(World& world) {
    bool todo;

    do {
        todo = false;

        for (auto top : top_level_lambdas(world)) {
            Scope scope(top);
            for (auto lambda : scope.rpo()) {
                if (lambda->empty())
                    continue;

                bool has_run = false;
                auto to = lambda->to()->isa_lambda();
                if (!to) {
                    if (auto run = lambda->to()->isa<Run>())
                        has_run = to = run->def()->isa_lambda();
                } else if (to->isa<Halt>())
                    continue;

                if (to) {
                    Scope scope(to);
                    Call e_call(to), f_call(to);
                    Lambda* e_dropped = nullptr;
                    Lambda* f_dropped = nullptr;
                    std::vector<Lambda*> e_new, f_new;
                    bool e_cached, f_cached = false;

                    GenericMap map;
                    bool res = to->type()->infer_with(map, lambda->arg_pi());
                    assert(res);

                    {
                        for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                            if (auto run = lambda->arg(i)->isa<Run>()) {
                                has_run = true;
                                e_call.args.push_back(run);
                                e_call.idx.push_back(i);
                            }
                        }

                        if (has_run) {
                            e_cached = cached(world, e_call);
                            e_dropped = drop(scope, e_call.idx, e_call.args, map);
                            if (!e_cached) {
                                e_new.push_back(e_dropped);
                                for (auto lambda : scope.rpo()) {
                                    auto mapped = (Lambda*) lambda->ptr;
                                    if (mapped != lambda)
                                        e_new.push_back(mapped);
                                }
                            }
                        }
                    }
                    if (!has_run)
                        continue;
                    {
                        for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                            if (!lambda->arg(i)->isa<Halt>()) {
                                f_call.args.push_back(lambda->arg(i));
                                f_call.idx.push_back(i);
                            }
                        }

                        f_cached = cached(world, f_call);
                        f_dropped = drop(scope, f_call.idx, f_call.args, map);
                        if (!f_cached) {
                            f_new.push_back(f_dropped);
                            for (auto lambda : scope.rpo()) {
                                auto mapped = (Lambda*) lambda->ptr;
                                if (mapped != lambda)
                                    f_new.push_back(mapped);
                            }
                        }
                    }

                    assert(f_dropped != nullptr);
                    bool use_f = f_dropped->to()->isa<Lambda>();
                    if (!use_f) {
                        if (auto run = f_dropped->to()->isa<Run>())
                            use_f = run->def()->isa<Lambda>();
                    }
                    Lambda* dropped = use_f ? f_dropped : e_dropped;
                    auto& new_lambdas = use_f ? f_new : e_new;
                    auto& idx = use_f ? f_call.idx : e_call.idx;

                    lambda->jump(dropped, lambda->args().cut(idx));
                    todo = true;

                    if ((use_f && !f_cached) || (!use_f && !e_cached)) {
                        // propagate run
                        for (auto lambda : new_lambdas) {
                            if (auto to = lambda->to()->isa_lambda())
                                lambda->update_to(world.run(to));
                        }
                    }

                    goto next;
                }
            }
        }
next:;
        world.cleanup();
    } while (todo);
    
    for (auto lambda : world.lambdas()) {
        for (size_t i = 0, e = lambda->size(); i != e; ++i) {
            auto op = lambda->op(i);
            if (auto evalop = op->isa<EvalOp>())
                lambda->update_op(i, evalop->def());
        }
    }
}

}
