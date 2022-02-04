#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"

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
        , boundary_(Def::gid_counter())
    {}

    World& world() { return world_; }
    bool run();
    void enqueue(Lam* lambda) {
        if (lambda->gid() < 2 * boundary_ && done_.emplace(lambda).second)
            queue_.push(lambda);
    }
    void eat_pe_info(Lam*);

private:
    World& world_;
    bool lower2cff_;
    HashMap<const App*, Lam*, HashApp> cache_;
    LamSet done_;
    std::queue<Lam*> queue_;
    LamMap<bool> top_level_;
    size_t boundary_;
};

class CondEval {
public:
    CondEval(Lam* callee, Defs args, LamMap<bool>& top_level)
        : callee_(callee)
        , top_level_(top_level)
    {
        assert(callee->filter()->is_empty() || callee->filter()->size() == args.size());
        assert(callee->num_params() == args.size());

        for (size_t i = 0, e = args.size(); i != e; ++i)
            old2new_[callee->param(i)] = args[i];
    }

    World& world() { return callee_->world(); }
    const Def* instantiate(const Def* odef) {
        if (auto ndef = old2new_.lookup(odef))
            return *ndef;

        if (odef->isa_structural()) {
            Array<const Def*> nops(odef->num_ops());
            for (size_t i = 0; i != odef->num_ops(); ++i)
                nops[i] = instantiate(odef->op(i));

            auto nprimop = odef->rebuild(world(), odef->type(), nops);
            return old2new_[odef] = nprimop;
        }

        return old2new_[odef] = odef;
    }

    bool eval(size_t i, bool lower2cff) {
        // the only higher order parameter that is allowed is a single 1st-order fn-parameter of a top-level lambda
        // all other parameters need specialization (lower2cff)
        auto order = callee_->param(i)->order();
        if (lower2cff)
            if(order >= 2 || (order == 1
                        && (!callee_->param(i)->type()->isa<FnType>()
                            || (!callee_->is_returning() || (!is_top_level(callee_)))))) {
            world().DLOG("bad param({}) {} of lambda {}", i, callee_->param(i), callee_);
            return true;
        }

        return (!callee_->is_exported() && callee_->can_be_inlined()) || is_one(instantiate(filter(i)));
        //return is_one(instantiate(filter(i)));
    }

    const Def* filter(size_t i) {
        return callee_->filter()->is_empty() ? world().literal_bool(false, {}) : callee_->filter()->condition(i);
    }

    bool is_top_level(Lam* lambda) {
        auto p = top_level_.emplace(lambda, true);
        if (!p.second)
            return p.first->second;

        Scope scope(lambda);
        unique_queue<DefSet> queue;

        for (auto def : scope.free())
            queue.push(def);

        while (!queue.empty()) {
            auto def = queue.pop();

            if (def->isa<Param>()) // if FV in this scope is a param, this lam can't be top-level
                return top_level_[lambda] = false;
            if (auto free_cn = def->isa_nom<Lam>()) {
                // if we have a non-top level lambda in scope as a free variable,
                // then it must be bound by some outer lambda, and so we aren't top-level
                if (!is_top_level(free_cn))
                    return top_level_[lambda] = false;
            } else {
                for (auto op : def->ops())
                    queue.push(op);
            }
        }

        return top_level_[lambda] = true;
    }

private:
    Lam* callee_;
    Def2Def old2new_;
    LamMap<bool>& top_level_;
};

void PartialEvaluator::eat_pe_info(Lam* cur) {
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
    } else if (auto lambda = next->isa_nom<Lam>()) {
        queue_.push(lambda);
    }
}

bool PartialEvaluator::run() {
    bool todo = false;

    for (auto&& [_, lam] : world().externals()) {
        if (!lam->has_body()) continue;
        enqueue(lam);
        top_level_[lam] = true;
    }

    while (!queue_.empty()) {
        auto lam = pop(queue_);

        bool force_fold = false;

        if (!lam->has_body())
            continue;
        const App* body = lam->body();
        const Def* callee_def = lam->body()->callee();

        if (auto run = callee_def->isa<Run>()) {
            force_fold = true;
            callee_def = run->def();
        }

        if (auto callee = callee_def->isa_nom<Lam>()) {
            if (callee->intrinsic() == Intrinsic::PeInfo) {
                eat_pe_info(lam);
                continue;
            }

            if (callee->has_body()) {
                CondEval cond_eval(callee, body->args(), top_level_);

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
                    Lam*& target = p.first->second;
                    // create new specialization if not found in cache
                    if (p.second) {
                        target = drop(callee, specialize);
                        todo = true;
                    }

                    jump_to_dropped_call(lam, target, specialize);

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
    world.VLOG("start {}", name);
    auto res = PartialEvaluator(world, lower2cff).run();
    world.VLOG("end {}", name);
    return res;
}

//------------------------------------------------------------------------------

}
