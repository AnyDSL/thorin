#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"
#include "partial_evaluation.h"

namespace thorin {

struct HashApp {
    inline static uint64_t hash(const App* app) {
        return murmur3(uint64_t(app));
    }
    inline static bool eq(const App* a1, const App* a2) { return a1 == a2; }
    inline static const App* sentinel() { return static_cast<const App*>((void*)size_t(-1)); }
};

class PartialEvaluator {
public:
    PartialEvaluator(World& world, bool lower2cff)
        : world_(world)
        , lower2cff_(lower2cff)
    {}

    World& world() { return world_; }
    bool run();
    void enqueue(Continuation* continuation) {
        queue_.push(continuation);
    }
    void eat_pe_info(Continuation*);

private:
    World& world_;
    bool lower2cff_;
    HashMap<const App*, Continuation*, HashApp> cache_;
    ContinuationSet done_;
    unique_queue<ContinuationSet> queue_;
};

const Def* BetaReducer::rewrite(const Def* odef) {
    // leave nominal defs alone
    if (odef->isa_nom())
        return odef;
    return Rewriter::rewrite(odef);
}

class CondEval {
public:
    CondEval(const App* app)
        : app_(app)
    {
        callee_ = app->callee()->as_nom<Continuation>();
        BetaReducer reducer(app->world());
        for (size_t i = 0; i < app->num_args(); i++)
            reducer.provide_arg(callee_->param(i), app->arg(i));
        filter_ = reducer.reduce(callee_->filter())->as<Filter>();

        // this ought to work but doesn't :(
        // filter_ = app->filter();
        assert(app->filter()->is_empty() || app->filter()->size() == app->num_args());
    }

    bool eval(size_t i, bool lower2cff) {
        // the only higher order parameter that is allowed is a single 1st-order fn-parameter of a top-level continuation
        // all other parameters need specialization (lower2cff)
        auto order = callee_->param(i)->order();
        if (lower2cff)
            if(order >= 2 || (order == 1
                        && (!callee_->param(i)->type()->isa<FnType>()
                            || (!callee_->is_returning() || (!is_top_level(callee_)))))) {
            world().DLOG("bad param({}) {} for continuation {}", i, callee_->param(i), callee_);
            return true;
        }

        return (!callee_->is_exported() && callee_->can_be_inlined()) || is_one(filter(i));
    }

protected:
    World& world() { return app_->world(); }

    const Def* filter(size_t i) {
        return filter_->is_empty() ? world().literal_bool(false, {}) : filter_->condition(i);
    }

    bool is_top_level(Continuation* continuation) {
        auto p = top_level_.emplace(continuation, true);
        if (!p.second)
            return p.first->second;

        if (continuation->is_exported())
            return top_level_[continuation] = true;

        Scope scope(continuation);
        unique_queue<DefSet> queue;

        for (auto def : scope.free())
            queue.push(def);

        while (!queue.empty()) {
            auto def = queue.pop();

            if (def->isa<Param>()) // if FV in this scope is a param, this cont can't be top-level
                return top_level_[continuation] = false;
            if (auto free_cn = def->isa_nom<Continuation>()) {
                // if we have a non-top level continuation in scope as a free variable,
                // then it must be bound by some outer continuation, and so we aren't top-level
                if (!is_top_level(free_cn))
                    return top_level_[continuation] = false;
            } else {
                for (auto op : def->ops())
                    queue.push(op);
            }
        }

        return top_level_[continuation] = true;
    }

private:
    const App* app_;
    Continuation* callee_;
    const Filter* filter_;
    ContinuationMap<bool> top_level_;
};

void PartialEvaluator::eat_pe_info(Continuation* cur) {
    assert(cur->has_body());
    auto body = cur->body();
    assert(body->arg(1)->type() == world().ptr_type(world().indefinite_array_type(world().type_pu8())));
    auto next = body->arg(3);

    if (!body->arg(2)->has_dep(Dep::Param)) {
        auto msg = body->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<DefiniteArray>();
        world().idef(body->callee(), "pe_info: {}: {}", msg->as_string(), body->arg(2));
        cur->jump(next, {body->arg(0)}, cur->debug()); // TODO debug

        // always re-insert into queue because we've changed cur's jump
        queue_.push(cur);
    } else if (auto continuation = next->isa_nom<Continuation>()) {
        queue_.push(continuation);
    }
}

bool PartialEvaluator::run() {
    bool todo = false;

    for (auto&& [_, def] : world().externals()) {
        auto cont = def->isa<Continuation>();
        if (!cont) continue;
        if (!cont->has_body()) continue;
        enqueue(cont);
    }

    while (!queue_.empty()) {
        auto continuation = queue_.pop();

        bool force_fold = false;

        if (!continuation->has_body())
            continue;
        const App* body = continuation->body();
        const Def* callee_def = continuation->body()->callee();

        if (auto callee = callee_def->isa_nom<Continuation>()) {
            if (callee->intrinsic() == Intrinsic::PeInfo) {
                eat_pe_info(continuation);
                continue;
            }

            if (callee->has_body()) {
                CondEval cond_eval(continuation->body());

                std::vector<const Def*> specialize(body->num_args());

                bool fold = false;
                for (size_t i = 0, e = body->num_args(); i != e; ++i) {
                    if (force_fold || cond_eval.eval(i, lower2cff_)) {
                        specialize[i] = body->arg(i);
                        fold = true;
                    } else
                        specialize[i] = nullptr;
                }

                if (fold) {
                    const auto& p = cache_.emplace(body, nullptr);
                    Continuation*& target = p.first->second;
                    // create new specialization if not found in cache
                    if (p.second) {
                        target = drop(callee, specialize);
                        todo = true;
                    }

                    jump_to_dropped_call(continuation, target, specialize);

                    while (callee && callee->never_called()) {
                        if (callee->has_body()) {
                            auto new_callee = const_cast<Continuation*>(callee->body()->callee()->isa<Continuation>());
                            callee->destroy("partial_evaluation");
                            callee = new_callee;
                        } else {
                            callee = nullptr;
                        }
                    }

                    if (lower2cff_ && fold) {
                        // re-examine next iteration:
                        // maybe the specialization is not top-level anymore which might need further specialization
                        queue_.push(continuation);
                        continue;
                    }
                }
            }
        }

        for (auto succ : continuation->succs())
            enqueue(succ);
    }

    return todo;
}

//------------------------------------------------------------------------------

bool partial_evaluation(World& world, bool lower2cff) {
    auto name = lower2cff ? "lower2cff" : "partial_evaluation";
    world.VLOG("start {}", name);
    auto res = PartialEvaluator(world, lower2cff).run();
    world.VLOG("end {}", name);
    return res;
}

//------------------------------------------------------------------------------

}
