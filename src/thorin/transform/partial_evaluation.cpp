#include "thorin/world.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"

// WARNING This file will be nuked

namespace thorin {

void app_to_dropped_app(Lam* src, Lam* dst, const App* app) {
    std::vector<const Def*> nargs;
    auto src_app = src->body()->as<App>();
    for (size_t i = 0, e = src_app->num_args(); i != e; ++i) {
        if (app->arg(i)->isa<Top>())
            nargs.push_back(src_app->arg(i));
    }

    src->app(dst, nargs, src_app->dbg());
}

std::vector<Lam*> succs(Lam* lam) {
    std::vector<Lam*> succs;
    std::queue<const Def*> queue;
    DefSet done;

    auto enqueue = [&] (const Def* def) {
        if (done.find(def) == done.end()) {
            queue.push(def);
            done.insert(def);
        }
    };

    done.insert(lam);
    enqueue(lam->body());

    while (!queue.empty()) {
        auto def = pop(queue);
        if (auto lam = def->isa_nom<Lam>()) {
            succs.push_back(lam);
            continue;
        }

        for (auto op : def->ops())
            enqueue(op);
    }

    return succs;
}

class PartialEvaluator {
public:
    PartialEvaluator(World& world, bool lower2cff)
        : world_(world)
        , lower2cff_(lower2cff)
        , boundary_(world.cur_gid())
    {}

    World& world() { return world_; }
    bool run();
    void enqueue(Lam* lam) {
        if (lam->is_set() && lam->gid() < 2 * boundary_ && done_.emplace(lam).second)
            queue_.push(lam);
    }
    void eat_pe_info(Lam*);

private:
    World& world_;
    bool lower2cff_;
    GIDMap<const App*, Lam*> cache_;
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
        assert(callee->num_vars() == args.size());

        for (size_t i = 0, e = args.size(); i != e; ++i)
            old2new_[callee->var(i)] = args[i];
    }

    World& world() { return callee_->world(); }
    const Def* instantiate(const Def* odef) {
        if (auto ndef = old2new_.lookup(odef))
            return *ndef;

        if (!odef->isa_nom()) {
            Array<const Def*> nops(odef->num_ops());
            for (size_t i = 0; i != odef->num_ops(); ++i)
                nops[i] = instantiate(odef->op(i));

            auto ndef = odef->rebuild(world(), odef->type(), nops, odef->dbg());
            return old2new_[odef] = ndef;
        }

        return old2new_[odef] = odef;
    }

    bool eval(size_t i, bool lower2cff) {
        // the only higher order var that is allowed is a single 1st-order fn-var of a top-level lam
        // all other vars need specialization (lower2cff)
        auto order = callee_->var(i)->type()->order();
        if (lower2cff)
            if(order >= 2 || (order == 1
                        && (!callee_->var(i)->type()->isa<Pi>()
                        || (!callee_->is_returning() || (!is_top_level(callee_)))))) {
            world().DLOG("bad var({}) {} of lam {}", i, callee_->var(i), callee_);
            return true;
        }

        auto instance = isa_lit<u64>(instantiate(filter(i)));
        auto is_one = instance ? *instance : false;

        return (!callee_->is_external() && callee_->num_uses() == 1) || is_one;
        //return is_one(instantiate(filter(i)));
    }

    const Def* filter(size_t /*i*/) { return callee_->filter(); }

    bool is_top_level(Lam* lam) {
        auto p = top_level_.emplace(lam, true);
        if (!p.second)
            return p.first->second;

        Scope scope(lam);
        unique_queue<DefSet> queue;

        if (!scope.free().vars.empty())
            return top_level_[lam] = false;

        for (auto nom : scope.free().noms) {
            if (auto free_cn = nom->isa<Lam>()) {
                if (!is_top_level(free_cn))
                    return top_level_[lam] = false;
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
    auto next = cur->body()->as<App>()->arg(3);

    if (cur->body()->as<App>()->arg(2)->is_const()) {
        //auto msg = cur->body()->as<App>()->arg(1)->as<Bitcast>()->from()->as<Global>()->init();
        world().idef(cur->body()->as<App>()->callee(), "pe_info: {}: {}", "TODO", cur->body()->as<App>()->arg(2));
        cur->app(next, {cur->body()->as<App>()->arg(0)}, cur->body()->as<App>()->dbg());

        // always re-insert into queue because we've changed cur's jump
        queue_.push(cur);
    } else if (auto lam = next->isa_nom<Lam>()) {
        queue_.push(lam);
    }
}

bool PartialEvaluator::run() {
    bool todo = false;

    for (const auto& [name, nom] : world().externals()) {
        if (auto lam = nom->isa<Lam>()) {
            enqueue(lam);
            top_level_[lam] = true;
        }
    }

    while (!queue_.empty()) {
        auto lam = pop(queue_);

        auto app = lam->body()->isa<App>();
        if (app == nullptr) continue;

        bool force_fold = false;
        auto callee_def = app->callee();


        if (auto run = isa<Tag::PE>(PE::run, app->callee())) {
            force_fold = true;
            callee_def = run->arg();
        }

        if (auto callee = callee_def->isa_nom<Lam>()) {
            // TODO
            //if (callee->intrinsic() == Lam::Intrinsic::PeInfo) {
                //eat_pe_info(lam);
                //continue;
            //}

            if (callee->is_set()) {
                size_t num_args = app->num_args();
                Array<const Def*> args(num_args);

                CondEval cond_eval(callee, app->args(), top_level_);

                bool fold = false;
                for (size_t i = 0; i != num_args; ++i) {
                    if (force_fold || cond_eval.eval(i, lower2cff_)) {
                        args[i] = app->arg(i);
                        fold = true;
                    } else {
                        args[i] = world().top(callee->var(i)->type());
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

        for (auto succ : succs(lam))
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
