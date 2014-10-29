#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"
#include "thorin/util/queue.h"

namespace thorin {

static std::vector<Lambda*> top_level_lambdas(World& world) {
    std::vector<Lambda*> result;
    Scope::for_each<false>(world, [&] (const Scope& scope) { result.push_back(scope.entry()); });
    return result;
}

//------------------------------------------------------------------------------

class Call {
public:
    Call() {}
    Call(Lambda* to)
        : to_(to)
        , args_(to->type()->num_args())
    {}

    Lambda* to() const { return to_; }
    ArrayRef<Def> args() const { return args_; }
    Def& arg(size_t i) { return args_[i]; }
    const Def& arg(size_t i) const { return args_[i]; }
    bool operator == (const Call& other) const { return this->to() == other.to() && this->args() == other.args(); }

private:
    Lambda* to_;
    Array<Def> args_;
};

struct CallHash {
    size_t operator () (const Call& call) const {
        return hash_combine(hash_value(call.args()), call.to());
    }
};

//------------------------------------------------------------------------------

class PartialEvaluator {
public:
    PartialEvaluator(World& world)
        : scope_(world, top_level_lambdas(world))
        , postdomtree_(scope_.postdomtree())
    {
        for (auto lambda : world.lambdas()) {
            new2old_[lambda] = lambda;
            old2new_[lambda] = lambda;
        }
    }

    World& world() { return scope().world(); }
    const Scope& scope() const { return scope_; }
    void seek();
    void eval(const Run* cur_run, Lambda* cur);
    void rewrite_jump(Lambda* src, Lambda* dst, const Call&);
    void update_new2old(const Def2Def& map);

private:
    Scope scope_;
    const PostDomTree* postdomtree_;
    Lambda2Lambda new2old_;
    Lambda2Lambda old2new_;
    LambdaSet done_;
    HashMap<Call, Lambda*, CallHash> cache_;
};

void PartialEvaluator::seek() {
    LambdaSet visited;
    std::queue<Lambda*> queue;
    for (auto lambda : world().externals()) {
        queue.push(lambda);
        visited.insert(lambda);
    }

    while (!queue.empty()) {
        auto lambda = pop(queue);
        if (!lambda->empty()) {
            if (auto run = lambda->to()->isa<Run>())
                eval(run, lambda);
        }
        for (auto succ : lambda->succs()) {
            if (!visited.contains(succ)) {
                queue.push(succ);
                visited.insert(succ);
            }
        }
    }
}

void PartialEvaluator::eval(const Run* cur_run, Lambda* cur) {
    while (!done_.contains(cur)) {
        done_.insert(cur);
        if (cur->empty())
            return;

        Lambda* dst = nullptr;
        if (auto hlt = cur->to()->isa<Hlt>()) {
            cur = nullptr;
            for (auto use : hlt->uses()) {  // TODO assert that there is only one EndHlt user
                if (auto end = use->isa<EndHlt>()) {
                    if (auto lambda = end->def()->isa_lambda())
                        lambda->update_to(world().run(lambda->to()));
                    //break; // TODO there may be multiple versions of that due to updates
                }
            }
            return;
        } else if (auto run = cur->to()->isa<Run>()) {
            dst = run->def()->isa_lambda();
        } else {
            dst = cur->to()->isa_lambda();
        }

        if (dst == world().branch())
            dst = nullptr;

        if (dst == nullptr) {               // skip to immediate post-dominator
            cur = old2new_[postdomtree_->idom(new2old_[cur])];
        } else if (dst->empty()) {
            if (!cur->args().empty()) {
                if (auto lambda = cur->args().back()->isa_lambda()) {
                    cur = lambda;
                    continue;
                } else if (dst->is_intrinsic()) {
                    for (size_t i = cur->num_args(); i-- != 0;) {
                        if (auto lambda = cur->arg(i)->isa_lambda()) {
                            cur = lambda;
                            goto next_lambda;
                        }
                    }
                }
            }
            return;
        } else {
            Call call(dst);
            for (size_t i = 0; i != cur->num_args(); ++i) {
                call.arg(i) = nullptr;
                if (cur->arg(i)->isa<Hlt>()) {
                    continue;
                } else if (auto end = cur->arg(i)->isa<EndRun>()) {
                    if (end->run() == cur_run) {
                        end->replace(end->def()); // TODO factor
                        continue;
                    } else {
                        end->replace(end->def()); // TODO factor
                        call.arg(i) = end->def();
                        continue;
                    }
                }
                call.arg(i) = cur->arg(i);
            }

            if (auto cached = find(cache_, call)) { // check for cached version
                rewrite_jump(cur, cached, call);
                return;
            } else {                                // no cached version found... create a new one
                Scope scope(dst);
                Def2Def old2new;
                Type2Type type2type;
                bool res = dst->type()->infer_with(type2type, cur->arg_fn_type());
                assert(res);
                auto dropped = drop(scope, old2new, call.args(), type2type);
                old2new[dst] = dropped;
                update_new2old(old2new);
                rewrite_jump(cur, dropped, call);
                cur = dropped;
            }
        }
next_lambda:;
    }
}

void PartialEvaluator::rewrite_jump(Lambda* src, Lambda* dst, const Call& call) {
    std::vector<Def> nargs;
    for (size_t i = 0, e = src->num_args(); i != e; ++i) {
        if (call.arg(i) == nullptr)
            nargs.push_back(src->arg(i));
    }

    src->jump(dst, nargs);
    cache_[call] = dst;
}

void PartialEvaluator::update_new2old(const Def2Def& old2new) {
    for (auto p : old2new) {
        if (auto olambda = p.first->isa_lambda()) {
            auto nlambda = p.second->as_lambda();
            if (!nlambda->empty() && nlambda->to()->isa<Bottom>())
                continue;
            assert(new2old_.contains(olambda));
            auto orig = new2old_[olambda];
            new2old_[nlambda] = orig;
            old2new_[orig] = nlambda;
        }
    }
}

//------------------------------------------------------------------------------

void partial_evaluation(World& world) {
    PartialEvaluator(world).seek();

    for (auto primop : world.primops()) {
        if (auto evalop = Def(primop)->isa<EvalOp>())
            evalop->replace(evalop->def());
        else if (auto end = Def(primop)->isa<EndEvalOp>())
            end->replace(end->def());

    }
}

//------------------------------------------------------------------------------

}
