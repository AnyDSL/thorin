#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/schedule.h"
#include "anydsl2/analyses/looptree.h"
#include "anydsl2/transform/mangle.h"
#include "anydsl2/transform/merge_lambdas.h"

namespace anydsl2 {

class PartialEvaluator {
public:
    PartialEvaluator(Scope& scope, ArrayRef<size_t> not_evaluable_params)
        : scope_(scope)
        , looptree_(scope.looptree())
        , schedule_(schedule_early(scope))
        , pass_(scope.world().new_pass())
        , todo_(true)
    {
        for (auto i : not_evaluable_params)
            set_not_evaluable(scope.entries()[0]->param(i));

        while (todo_) {
            todo_ = false;

            for (auto lambda : scope.rpo()) {
                for (auto param : lambda->params()) {
                    for (auto peek : param->peek()) {
                        if (get_evaluable(param)) {
                            if (!get_evaluable(peek.def()))
                                set_not_evaluable(param);
                        } else
                            goto next_param;
                    }
next_param:;
                }

                for (auto primop : schedule_[lambda->sid()]) {
                    bool evaluable = true;
                    for (auto op : primop->ops())
                        evaluable &= get_evaluable(op);
                    if (!evaluable)
                        set_not_evaluable(primop);
                }
            }
        }

        fill(looptree_.root());
    }

    bool get_evaluable(Def def) { 
        if (!def->visit(pass_))
            return def->flags[0] = todo_ = true;
        return def->flags[0];
    }

    void set_not_evaluable(Def def) { 
        if (!def->visit(pass_)) {
            todo_ = true;
            def->flags[0] = false;
        } else {
            if (def->flags[0] != false) {
                todo_ = true;
                assert(def->flags[0]);
                def->flags[0] = false;
            }
        }
    }

    void fill(const LoopNode*);

    const Scope& scope_;
    const LoopTree& looptree_;
    Schedule schedule_;
    const size_t pass_;
    bool todo_;
    std::vector<Lambda*> run_;
};

void PartialEvaluator::fill(const LoopNode* n) {
    if (auto header = n->isa<LoopHeader>()) {
        for (auto lambda : header->headers()) {
            if (get_evaluable(lambda->to())) {
                for (auto use : lambda->uses()) {
                    if (auto ulambda = use->isa_lambda()) {
                        if (scope_.contains(ulambda))
                            run_.push_back(ulambda);
                    }
                }
            }
        }
        for (auto child : header->children())
            fill(child);
    }
}

void partial_evaluation(World& world) {
    bool todo = false;
    do {
        todo = false;
        for (auto lambda : world.lambdas()) {
            if (!lambda->empty()) {
                if (auto to = lambda->to()->isa_lambda()) {
                    if (!to->empty() && lambda->attribute().is(Lambda::Run)) {
                        Scope scope(to);
                        Array<size_t> idx(lambda->num_args());
                        size_t x = 0;
                        for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                            if (lambda->arg(i)->is_const())
                                idx[x++] = i;
                        }
                        idx.shrink(x);
                        PartialEvaluator pe(scope, idx);
                        auto dropped = drop(scope, lambda->args(), pe.run_);
                        lambda->jump(dropped, {});
                        //lambda->attr().unset_run();
                        todo = true;
                    }
                }
            }
        }

        merge_lambdas(world);
        world.unreachable_code_elimination();
        //std::cout << world.gid() << std::endl;
    } while (todo);
}

}
