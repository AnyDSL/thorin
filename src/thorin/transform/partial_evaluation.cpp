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
    PartialEvaluator(Thorin& thorin, bool lower2cff)
        : thorin_(thorin)
        , lower2cff_(lower2cff)
        , boundary_(Def::gid_counter())
    {}

    World& world() { return thorin_.world(); }
    bool run();
    void enqueue(Continuation* continuation) {
        if (continuation->gid() < 2 * boundary_ && done_.emplace(continuation).second)
            queue_.push(continuation);
    }
    void eat_pe_info(Continuation*);

private:
    Thorin& thorin_;
    bool lower2cff_;
    HashMap<const App*, Continuation*, HashApp> cache_;
    ContinuationSet done_;
    unique_queue<ContinuationSet> queue_;
    size_t boundary_;
};

const Def* BetaReducer::rewrite(const Def* odef) {
    // leave nominal defs alone
    if (odef->isa_nom())
        return odef;
    return Rewriter::rewrite(odef);
}

class CondEval {
public:
    CondEval(Continuation* callee, ScopesForest& forest, Defs args)
        : reducer_(callee->world())
        , callee_(callee)
        , forest_(forest)
    {
        assert(callee->filter()->is_empty() || callee->filter()->size() == args.size());
        assert(callee->num_params() == args.size());

        for (size_t i = 0, e = args.size(); i != e; ++i)
            reducer_.provide_arg(callee->param(i), args[i]);
    }

    World& world() { return callee_->world(); }

    bool eval(size_t i, bool lower2cff) {
        // the only higher order parameter that is allowed is a single 1st-order fn-parameter of a top-level continuation
        // all other parameters need specialization (lower2cff)
        auto order = callee_->param(i)->order();
        if (lower2cff)
            if(order >= 2 || (order == 1
                              && (!callee_->param(i)->type()->isa<FnType>()
                                  || (!callee_->is_returning() || (!is_top_level(callee_)))))) {
                world().DLOG("bad param({}) {} of continuation {}", i, callee_->param(i), callee_);
                return true;
            }

        return ((!callee_->is_exported() || callee_->attributes().cc == CC::Thorin) && callee_->can_be_inlined()) || is_one(reducer_.instantiate(filter(i)));
        //return is_one(instantiate(filter(i)));
    }

    const Def* filter(size_t i) {
        return callee_->filter()->is_empty() ? world().literal_bool(false, {}) : callee_->filter()->condition(i);
    }

    bool is_top_level(Continuation* continuation) {
        return !forest_.get_scope(continuation).has_free_params();
    }

private:
    BetaReducer reducer_;
    Continuation* callee_;
    ScopesForest& forest_;
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

        while (auto run = callee_def->isa<Run>()) {
            force_fold = true;
            callee_def = run->def();
        }

        if (auto callee = callee_def->isa_nom<Continuation>()) {
            if (callee->intrinsic() == Intrinsic::PeInfo) {
                eat_pe_info(continuation);
                continue;
            }

            if (callee->intrinsic() == Intrinsic::Plugin) {
                if (callee->attributes().depends) {
                    size_t num_dependend_uses = callee->attributes().depends->num_uses() - callee->attributes().depends->num_params();

                    //std::cerr << "Analyzing " << callee->unique_name() << " with dependency " << callee->attributes().depends->unique_name() << "\n";
                    //std::cerr << " => has " << num_dependend_uses << " real dependencies\n";
                    if (num_dependend_uses > 0) {
                        //Push the next continue so that other plugins get executed.
                        for (auto arg : body->args()) {
                            if (auto cont = arg->isa<Continuation>()) {
                                queue_.push(const_cast<Continuation*>(cont));
                            }
                        }
                        todo = true;
                        continue;
                    }
                }

                ScopesForest forest(world());
                CondEval cond_eval(callee, forest, body->args());

                //TODO: build specialize here to allow for parameter hiding.
                bool fold = false;
                for (size_t i = 0, e = body->num_args(); i != e; ++i) {
                    if (cond_eval.eval(i, lower2cff_)) {
                        fold = true;
                        break;
                    }
                }

                if (fold) {
                    std::vector<const Def*> specialize(body->arg(body->num_args() - 1)->as<Continuation>()->num_params());
                    specialize[0] = body->arg(0);

                    const auto& p = cache_.emplace(body, nullptr);
                    const Continuation* target = p.first->second;
                    // create new specialization if not found in cache
                    if (p.second) {
                        world().idef(continuation, "Plugin execute: {}", callee);

                        auto plugin_function = thorin_.search_plugin_function(callee->name().c_str());
                        if (!plugin_function) {
                            world().ELOG("Plugin function not found for: {}", callee->name());
                            continue;
                        }

                        const Def* output = plugin_function(&world(), body);
                        if (output)
                            specialize[1] = output;

                        target = body->arg(body->num_args() - 1)->as<Continuation>();
                        todo = true;
                    }
                    continuation->jump(target, specialize);

                    if (lower2cff_ && fold) {
                        // re-examine next iteration:
                        // maybe the specialization is not top-level anymore which might need further specialization
                        queue_.push(continuation);
                        continue;
                    }
                }

                continue;
            }

            if (callee->has_body()) {
                // TODO cache the forest and only rebuild it when we need to
                ScopesForest forest(world());
                CondEval cond_eval(callee, forest, body->args());

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
                        world().ddef(continuation, "Specializing call to {}", callee);
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

bool partial_evaluation(Thorin& thorin, bool lower2cff) {
    auto name = lower2cff ? "lower2cff" : "partial_evaluation";
    thorin.world().VLOG("start {}", name);
    auto res = PartialEvaluator(thorin, lower2cff).run();
    thorin.world().VLOG("end {}", name);
    return res;
}

//------------------------------------------------------------------------------

}
