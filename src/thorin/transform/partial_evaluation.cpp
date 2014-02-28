#include <queue>
#include <unordered_map>

#include "thorin/literal.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"

namespace thorin {

static std::vector<Lambda*> top_level_lambdas(World& world) {
    std::vector<Lambda*> result;
    auto scopes = top_level_scopes(world);
    for (auto scope : scopes)
        result.push_back(scope->entry());
    return result;
}

//------------------------------------------------------------------------------

class Call {
public:
    Call() {}
    Call(Lambda* to)
        : to_(to)
        , args_(to->type()->as<Pi>()->size())
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
        return hash_combine(hash_value(call.to()), std::hash<ArrayRef<Def>>()(call.args())); 
    }
};

//------------------------------------------------------------------------------

class PartialEvaluator {
public:
    PartialEvaluator(World& world)
        : scope_(world, top_level_lambdas(world), false)
        , postdomtree_(scope_)
    {
        for (auto lambda : world.lambdas()) {
            new2old_[lambda] = lambda;
            old2new_[lambda] = lambda;
        }
    }

    World& world() { return scope().world(); }
    const Scope& scope() const { return scope_; }
    void seek();
    void eval(Lambda* cur);
    void rewrite_jump(Lambda* src, Lambda* dst, const Call&);
    void update_new2old(const Def2Def& map);

private:
    Scope scope_;
    const DomTree postdomtree_;
    Lambda2Lambda new2old_;
    Lambda2Lambda old2new_;
    LambdaSet done_;
    std::unordered_map<Call, Lambda*, CallHash> cache_;
};

void PartialEvaluator::seek() {
    LambdaSet visited;
    std::queue<Lambda*> queue;
    for (auto lambda : world().externals()) {
        queue.push(lambda);
        visited.insert(lambda);
    }

    while (!queue.empty()) {
        auto lambda = queue.front();
        queue.pop();
        if (!lambda->empty() && lambda->to()->isa<Run>())
            eval(lambda);
        for (auto succ : lambda->succs()) {
            if (!visited.contains(succ)) {
                queue.push(succ);
                visited.insert(succ);
            }
        }
    }
}

void PartialEvaluator::eval(Lambda* cur) {
    while (!done_.contains(cur)) {
        done_.insert(cur);
        if (cur->empty()) 
            break;

        auto to = cur->to();
        if (auto run = to->isa<Run>())
            to = run->def();

        auto dst = to->isa_lambda();
        if (dst == nullptr) {                   // skip to immediate post-dominator
            cur = old2new_[postdomtree_.idom(new2old_[cur])];
        } else {
            Call call(dst);
            bool fold = false;
            for (size_t i = 0; i != cur->num_args(); ++i) {
                call.arg(i) = cur->arg(i)->is_const() ? cur->arg(i) : nullptr;
                fold |= call.arg(i) != nullptr; // don't fold if there is nothing to fold
            }

            if (!fold || dst->empty()) {
                cur = dst;
            } else {
                auto i = cache_.find(call);
                if (i != cache_.end()) {        // check for cached version
                    rewrite_jump(cur, i->second, call);
                    break;
                } else {                        // no no cached version found... create a new one
                    Scope scope(dst);
                    Def2Def old2new;
                    GenericMap generic_map;
                    bool res = dst->type()->infer_with(generic_map, cur->arg_pi());
                    assert(res);
                    auto dropped = drop(scope, old2new, call.args(), generic_map);
                    old2new[dst] = dropped;
                    update_new2old(old2new);
                    rewrite_jump(cur, dropped, call);
                    cur = dropped;
                }
            }
        }
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
        if (auto evalop = primop->isa<EvalOp>())
            evalop->replace(evalop->def());
    }
}

//------------------------------------------------------------------------------

}
