#include <unordered_map>

#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/be/air.h"
#include "anydsl2/analyses/schedule.h"
#include "anydsl2/analyses/looptree.h"
#include "anydsl2/transform/mangle.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

#if 0
struct Call {
    Lambda* to;
    std::vector<const DefNode*> args;
    std::vector<size_t> idx;
};

struct CallHash { 
    size_t operator () (const Call& call) const { 
        return hash_combine(hash_combine(hash_value(call.to), 
                ArrayRef<const DefNode*>(call.args)), 
                ArrayRef<size_t>(call.idx));
    }
};

struct CallEqual { 
    bool operator () (const Call& call1, const Call& call2) const { 
        return call1.to == call2.to 
            && ArrayRef<const DefNode*>(call1.args) == ArrayRef<const DefNode*>(call2.args) 
            && ArrayRef<size_t>(call1.idx) == ArrayRef<size_t>(call2.idx);
    }
};
#endif

void partial_evaluation(World& world) {
    bool todo;

    //for (int counter = 0; counter < 10; ++counter) {
    do {
        todo = false;

        for (auto top : top_level_lambdas(world)) {
            Scope scope(top);
            for (auto lambda : scope.rpo()) {
                if (lambda->empty())
                    continue;
                if (auto to = lambda->to()->isa_lambda()) {
                    Scope scope(to);
                    std::vector<Def> args;
                    std::vector<size_t> idx;

                    for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                        if (auto run = lambda->arg(i)->isa<Run>()) {
                            args.push_back(run);
                            idx.push_back(i);
                        }
                    }

                    if (args.empty())
                        continue;

                    GenericMap map;
                    bool res = to->type()->infer_with(map, lambda->arg_pi());
                    assert(res);
                    auto dropped = drop(scope, idx, args, map);

                    lambda->jump(dropped, lambda->args().cut(idx));
                    todo = true;

                    // propagate run ops
                    {
                        Scope scope(dropped);
                        std::queue<Def> queue;
                        const auto pass1 = scope.mark();
                        const auto pass2 = world.new_pass();

                        auto fill_queue = [&] (Def def) {
                            assert(!def->isa<Lambda>());
                            for (auto op : def->ops()) {
                                if (op->cur_pass() < pass1)
                                    continue;

                                if (op->isa<Lambda>())
                                    continue;

                                assert(!op->isa<EvalOp>());

                                if (!op->visit(pass2)) {
                                    if (auto param = op->isa<Param>()) {
                                        for (auto peek : param->peek()) {
                                            //if (peek.def()->is_visited(pass2)) continue;
                                            //if (!scope.contains(peek.from())) continue;
                                            //if (!peek.def()->is_const() && peek.def()->cur_pass() < pass1) continue;
                                            //if (!peek.def()->is_const()) continue;
                                            //if (!peek.def()->order() == 0) continue;

                                            auto nrun = world.run(peek.def());
                                            //nrun->visit_first(pass2);
                                            nrun->visit(pass2);
                                            peek.from()->update_arg(param->index(), nrun);
                                            queue.push(nrun);
                                        }
                                    } else
                                        queue.push(op);
                                }
                            }
                        };

                        for (auto lambda : scope.rpo()) {
                            for (auto op : lambda->ops()) {
                                if (auto run = op->isa<Run>()) {
                                    fill_queue(run);
                                }
                            }
                        }

                        while (!queue.empty()) {
                            auto def = queue.front();
                            queue.pop();
                            fill_queue(def);
                        }
                    }
                }
            }
        }
    } while (todo);

#if 0
    for (auto lambda : world.lambdas()) {
        for (size_t i = 0, e = lambda->size(); i != e; ++i) {
            auto op = lambda->op(i);
            if (auto evalop = op->isa<EvalOp>())
                lambda->update_op(i, evalop->def());
        }
    }
#endif
}

}
