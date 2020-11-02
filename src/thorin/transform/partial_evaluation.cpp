#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"
#include "thorin/util/log.h"

namespace thorin {

class PartialEvaluator {
public:
    PartialEvaluator(World& world, bool lower2cff)
        : world_(world)
        , lower2cff_(lower2cff)
        , boundary_(Def::gid_counter())
    {}

    World& world() { return world_; }
    bool run();
    void enqueue(Lam* lam) {
        if (lam->gid() < 2 * boundary_ && done_.emplace(lam).second)
            queue_.push(lam);
    }
    void eat_pe_info(Lam*);

private:
    World& world_;
    bool lower2cff_;
    AppMap<Lam*> cache_;
    LamSet done_;
    std::queue<Lam*> queue_;
    LamMap<bool> top_level_;
    size_t boundary_;
};

class CondEval {
public:
    CondEval(Lam* callee, Defs args, LamMap<bool>& top_level)
        : callee_(callee)
        , args_(args)
        , top_level_(top_level)
    {
        //assert(callee->filter().empty() || callee->filter().size() == args.size());
        assert(callee->num_params() == args.size());

        for (size_t i = 0, e = args.size(); i != e; ++i)
            old2new_[callee->param(i)] = args[i];
    }

    World& world() { return callee_->world(); }
    const Def* instantiate(const Def* odef) {
        if (auto ndef = find(old2new_, odef))
            return ndef;

        if (auto oprimop = odef->isa<PrimOp>()) {
            Array<const Def*> nops(oprimop->num_ops());
            for (size_t i = 0; i != oprimop->num_ops(); ++i)
                nops[i] = instantiate(odef->op(i));

            auto nprimop = oprimop->rebuild(nops);
            return old2new_[oprimop] = nprimop;
        }

        return old2new_[odef] = odef;
    }

    bool eval(size_t i, bool lower2cff) {
        // the only higher order parameter that is allowed is a single 1st-order fn-parameter of a top-level lam
        // all other parameters need specialization (lower2cff)
        auto order = callee_->param(i)->order();
        if (lower2cff)
            if(order >= 2 || (order == 1
                        && (!callee_->param(i)->type()->isa<Pi>()
                        || (!callee_->is_returning() || (!is_top_level(callee_)))))) {
            DLOG("bad param({}) {} of lam {}", i, callee_->param(i), callee_);
            return true;
        }

        return (!callee_->is_exported() && callee_->num_uses() == 1) || is_one(instantiate(filter(i)));
        //return is_one(instantiate(filter(i)));
    }

    const Def* filter(size_t i) {
        return callee_->filter() == nullptr ? world().literal_bool(false, {}) : callee_->filter(i);
    }

    bool has_free_params(Lam* lam) {
        Scope scope(lam);
        return scope.has_free_params();
    }

    bool is_top_level(Lam* lam) {
        auto p = top_level_.emplace(lam, true);
        if (!p.second)
            return p.first->second;

        Scope scope(lam);
        unique_queue<DefSet> queue;

        for (auto def : scope.free())
            queue.push(def);

        while (!queue.empty()) {
            auto def = queue.pop();

            if (def->isa<Param>())
                return top_level_[lam] = false;
            if (auto free_cn = def->isa_lam()) {
                if (!is_top_level(free_cn))
                    return top_level_[lam] = false;
            } else {
                for (auto op : def->ops())
                    queue.push(op);
            }
        }

        return top_level_[lam] = true;
    }

private:
    Lam* callee_;
    Defs args_;
    Def2Def old2new_;
    LamMap<bool>& top_level_;
};

void PartialEvaluator::eat_pe_info(Lam* cur) {
    assert(cur->app()->arg(1)->type() == world().ptr_type(world().indefinite_array_type(world().type_pu8())));
    auto next = cur->app()->arg(3);

    if (is_const(cur->app()->arg(2))) {
        auto msg = cur->app()->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<DefiniteArray>();
        IDEF(cur->app()->callee(), "pe_info: {}: {}", msg->as_string(), cur->app()->arg(2));
        cur->app(next, {cur->app()->arg(0)}, cur->app()->debug());

        // always re-insert into queue because we've changed cur's jump
        queue_.push(cur);
    } else if (auto lam = next->isa_lam()) {
        queue_.push(lam);
    }
}

bool PartialEvaluator::run() {
    bool todo = false;

    for (auto continuation : world().exported_lams()) {
        enqueue(continuation);
        top_level_[continuation] = true;
    }

    while (!queue_.empty()) {
        auto lam = pop(queue_);

        auto app = lam->app();
        if (app == nullptr) continue;

        bool force_fold = false;
        auto callee_def = app->callee();


        if (auto run = app->callee()->isa<Run>()) {
            force_fold = true;
            callee_def = run->def();
        }

        if (auto callee = callee_def->isa_lam()) {
            if (callee->intrinsic() == Intrinsic::PeInfo) {
                eat_pe_info(lam);
                continue;
            }

            if (!callee->is_empty()) {
                size_t num_args = app->num_args();
                Array<const Def*> args(num_args);

                CondEval cond_eval(callee, app->args(), top_level_);

                bool fold = false;
                for (size_t i = 0; i != num_args; ++i) {
                    if (force_fold || cond_eval.eval(i, lower2cff_)) {
                        args[i] = app->arg(i);
                        fold = true;
                    } else {
                        args[i] = world().top(callee->param(i)->type());
                    }
                }

                if (fold) {
                    auto app = world().app(callee, args)->as<App>();
                    const auto& p = cache_.emplace(app, nullptr);
                    Lam*& target = p.first->second;
                    // create new specialization if not found in cache
                    if (p.second) {
                        target = drop(app);
                        todo = true;
                    }

                    app_to_dropped_app(lam, target, app);

                    if (lower2cff_ && fold) {
                        // re-examine next iteration:
                        // maybe the specialization is not top-level anymore which might need further specialization
                        queue_.push(lam);
                        continue;
                    }
                }
            }
        }

        for (auto succ : lam->succs())
            enqueue(succ);
    }

    return todo;
}

//------------------------------------------------------------------------------

bool partial_evaluation(World& world, bool lower2cff) {
    auto name = lower2cff ? "lower2cff" : "partial_evaluation";
    VLOG("start {}", name);
    auto res = PartialEvaluator(world, lower2cff).run();
    VLOG("end {}", name);
    return res;
}

//------------------------------------------------------------------------------

}
