#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"

namespace thorin {

class PartialEvaluator {
public:
    PartialEvaluator(World& world, bool lower2cff)
        : world_(world)
        , lower2cff_(lower2cff)
        , boundary_(world.cur_gid())
    {}

    World& world() { return world_; }
    bool run();
    void enqueue(Continuation* continuation) {
        if (continuation->gid() < 2 * boundary_ && done_.emplace(continuation).second)
            queue_.push(continuation);
    }
    void eat_pe_info(Continuation*);

private:
    World& world_;
    bool lower2cff_;
    HashMap<Call, Continuation*> cache_;
    ContinuationSet done_;
    std::queue<Continuation*> queue_;
    ContinuationMap<bool> top_level_;
    size_t boundary_;
};

class CondEval {
public:
    CondEval(Continuation* callee, Defs args, ContinuationMap<bool>& top_level)
        : callee_(callee)
        , args_(args)
        , top_level_(top_level)
    {
        assert(callee->filter().empty() || callee->filter().size() == args.size());
        assert(callee->num_params() == args.size());

        for (size_t i = 0, e = args.size(); i != e; ++i)
            old2new_[callee->param(i)] = args[i];
    }

    World& world() { return callee_->world(); }
    const Def* instantiate(const Def* odef) {
        if (auto ndef = old2new_.lookup(odef))
            return *ndef;

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

        return (!callee_->is_exported() && callee_->num_uses() == 1) || is_one(instantiate(filter(i)));
        //return is_one(instantiate(filter(i)));
    }

    const Def* filter(size_t i) {
        return callee_->filter().empty() ? world().literal_bool(false, {}) : callee_->filter(i);
    }

    bool has_free_params(Continuation* continuation) {
        Scope scope(continuation);
        return scope.has_free_params();
    }

    bool is_top_level(Continuation* continuation) {
        auto p = top_level_.emplace(continuation, true);
        if (!p.second)
            return p.first->second;

        Scope scope(continuation);
        unique_queue<DefSet> queue;

        for (auto def : scope.free())
            queue.push(def);

        while (!queue.empty()) {
            auto def = queue.pop();

            if (def->isa<Param>())
                return top_level_[continuation] = false;
            if (auto free_cn = def->isa_continuation()) {
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
    Continuation* callee_;
    Defs args_;
    Def2Def old2new_;
    ContinuationMap<bool>& top_level_;
};

void PartialEvaluator::eat_pe_info(Continuation* cur) {
    assert(cur->arg(1)->type() == world().ptr_type(world().indefinite_array_type(world().type_pu8())));
    auto next = cur->arg(3);

    if (is_const(cur->arg(2))) {
        auto msg = cur->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<DefiniteArray>();
        world().idef(cur->callee(), "pe_info: {}: {}", msg->as_string(), cur->arg(2));
        cur->jump(next, {cur->arg(0)}, cur->debug()); // TODO debug

        // always re-insert into queue because we've changed cur's jump
        queue_.push(cur);
    } else if (auto continuation = next->isa_continuation()) {
        queue_.push(continuation);
    }
}

bool PartialEvaluator::run() {
    bool todo = false;

    for (auto continuation : world().exported_continuations()) {
        enqueue(continuation);
        top_level_[continuation] = true;
    }

    while (!queue_.empty()) {
        auto continuation = pop(queue_);

        bool force_fold = false;
        auto callee_def = continuation->callee();

        if (auto run = continuation->callee()->isa<Run>()) {
            force_fold = true;
            callee_def = run->def();
        }

        if (auto callee = callee_def->isa_continuation()) {
            if (callee->intrinsic() == Intrinsic::PeInfo) {
                eat_pe_info(continuation);
                continue;
            }

            if (!callee->empty()) {
                Call call(continuation->num_ops());
                call.callee() = callee;

                CondEval cond_eval(callee, continuation->args(), top_level_);

                bool fold = false;
                for (size_t i = 0, e = call.num_args(); i != e; ++i) {
                    if (force_fold || cond_eval.eval(i, lower2cff_)) {
                        call.arg(i) = continuation->arg(i);
                        fold = true;
                    } else
                        call.arg(i) = nullptr;
                }

                if (fold) {
                    const auto& p = cache_.emplace(call, nullptr);
                    Continuation*& target = p.first->second;
                    // create new specialization if not found in cache
                    if (p.second) {
                        target = drop(call);
                        todo = true;
                    }

                    jump_to_dropped_call(continuation, target, call);

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
