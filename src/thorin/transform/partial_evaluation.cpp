#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/free_defs.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"
#include "thorin/util/log.h"

namespace thorin {

class PartialEvaluator {
public:
    PartialEvaluator(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
    void run();
    void enqueue(Continuation* continuation) {
        if (done_.emplace(continuation).second)
            queue_.push(continuation);
    }

private:
    World& world_;
    HashMap<Call, Continuation*> cache_;
    ContinuationSet done_;
    std::queue<Continuation*> queue_;
};

class CondEval {
public:
    CondEval(Continuation* callee, Defs args)
        : callee_(callee)
        , args_(args)
    {
        assert(callee->pe_profile().empty() || callee->pe_profile().size() == args.size());
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

    bool eval(size_t i) {
        // the only higher order parameter that is allowed is a single 1st-order paramter of a top-level continuation
        // all other paramters need specializtion (lower2cff)
        auto order = callee_->param(i)->order();
        if (order >= 2 || (order == 1 && (!callee_->is_returning() || has_free_vars(callee_)))) {
            DLOG("bad param({}) {} of continuation {}", i, callee_->param(i), callee_);
            return true;
        }

        return is_one(instantiate(pe_profile(i))) ? true : false;
    }

    const Def* pe_profile(size_t i) {
        return callee_->pe_profile().empty() ? world().literal_bool(false, {}) : callee_->pe_profile(i);
    }

private:
    Continuation* callee_;
    Defs args_;
    Def2Def old2new_;
};

void PartialEvaluator::run() {
    for (auto external : world().externals())
        enqueue(external);

    while (!queue_.empty()) {
        auto continuation = pop(queue_);

        if (auto callee = continuation->callee()->isa_continuation()) {
            if (!callee->empty()) {
                Call call(continuation);
                call.callee() = callee;

                bool fold = false;
                CondEval cond_eval(callee, continuation->args());

                for (size_t i = 0, e = call.num_args(); i != e; ++i) {
                    if (cond_eval.eval(i)) {
                        call.arg(i) = continuation->arg(i);
                        fold = true;
                    } else
                        call.arg(i) = nullptr;
                }

                const auto& p = cache_.emplace(call, nullptr);
                Continuation*& target = p.first->second;
                // create new specialization if not found in cache
                if (p.second)
                    target = drop(call);

                if (fold)
                    jump_to_dropped_call(continuation, target, call);
            }
        }

        for (auto succ : continuation->succs())
            enqueue(succ);
    }
}

void eat_pe_info(Continuation* cur, bool eval) {
    World& world = cur->world();
    assert(cur->arg(1)->type() == world.ptr_type(world.indefinite_array_type(world.type_pu8())));
    auto msg = cur->arg(1)->as<Bitcast>()->from()->as<Global>()->init()->as<DefiniteArray>();
    ILOG(cur->callee(), "{}pe_info: {}: {}", eval ? "" : "NOT evaluated: ", msg->as_string(), cur->arg(2));
    auto next = cur->arg(3);
    cur->jump(next, {cur->arg(0), world.tuple({})}, cur->jump_debug());
}

void eat_pe_known(Continuation* cur, bool eval) {
    World& world = cur->world();
    auto val = cur->arg(1);
    auto next = cur->arg(2);
    cur->jump(next, {cur->arg(0), world.literal(eval && is_const(val))}, cur->jump_debug());
}

bool eat_intrinsic(Intrinsic intrinsic, Continuation* cur, bool eval) {
    switch (intrinsic) {
        case Intrinsic::PeInfo:  eat_pe_info (cur, eval); return true;
        case Intrinsic::PeKnown: eat_pe_known(cur, eval); return true;
        default: return false;
    }
}

//------------------------------------------------------------------------------

template <typename F>
void eval_intrinsics(World& world, F f) {
    Scope::for_each(world, [&] (Scope& scope) {
        bool dirty = false;
        for (auto n : scope.f_cfg().post_order()) {
            auto continuation = n->continuation();
            if (auto callee = continuation->callee()->isa<Continuation>()) {
                dirty |= f(callee, continuation);
            }
        }

        if (dirty)
            scope.update();
    });
}

void eval(World& world) {
    PartialEvaluator(world).run();

#if 0
    // Eat all pe_known calls
    eval_intrinsics(world, [&] (auto callee, auto continuation) {
        if (callee->intrinsic() == Intrinsic::PeKnown) {
            eat_pe_known(continuation, false);
            return true;
        }
        return false;
    });

    // Eat all other intrinsics
    eval_intrinsics(world, [&] (auto callee, auto continuation) {
        return eat_intrinsic(callee->intrinsic(), continuation, false);
    });
#endif
}

void partial_evaluation(World& world) {
    world.cleanup();
    VLOG_SCOPE(eval(world));

    world.thorin();

    world.mark_pe_done();
    world.cleanup();
}

//------------------------------------------------------------------------------

}
