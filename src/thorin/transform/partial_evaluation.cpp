#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/hash.h"
#include "thorin/util/queue.h"

#include <iostream>

namespace thorin {

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
        : world_(world)
    {}

    World& world() { return world_; }
    void seek();
    void eval(Lambda* begin, Lambda* cur, Lambda* end);
    void rewrite_jump(Lambda* src, Lambda* dst, const Call&);
    void enqueue(Lambda* lambda) { 
        if (!visit(visited_, lambda))
            queue_.push(lambda); 
    }

private:
    World& world_;
    LambdaSet done_;
    std::queue<Lambda*> queue_;
    LambdaSet visited_;
    HashMap<Call, Lambda*, CallHash> cache_;
};

static Lambda* continuation(Lambda* lambda) {
    return lambda->num_args() != 0 ? lambda->args().back()->isa_lambda() : (Lambda*) nullptr;
}

void PartialEvaluator::seek() {
    for (auto lambda : world().externals())
        enqueue(lambda);

    Lambda* top = nullptr;
    while (!queue_.empty()) {
        auto lambda = pop(queue_);
        if (world().is_external(lambda))
            top = lambda;

        if (lambda->to()->isa<Run>())
            eval(top, lambda, continuation(lambda));

        for (auto succ : lambda->succs())
            enqueue(succ);
    }
}

void PartialEvaluator::eval(Lambda* top, Lambda* cur, Lambda* end) {
    if (end == nullptr)
        std::cout << "no matching end: " << cur->unique_name() << std::endl;
    else 
        std::cout << cur->unique_name() << " -> " << end->unique_name() << std::endl;

    while (true) {
        if (cur == nullptr) {
            std::cout << "cur is nullptr: " << std::endl;
            return;
        }
        if (done_.contains(cur)) {
            std::cout << "already done: " << cur->unique_name() << std::endl;
            return;
        }
        if (cur->empty()) {
            std::cout << "empty: " << cur->unique_name() << std::endl;
            return;
        }

        Lambda* dst = nullptr;
        if (auto run = cur->to()->isa<Run>()) {
            dst = run->def()->isa_lambda();
        } else if (cur->to()->isa<Hlt>()) {
            for (auto succ : cur->succs())
                enqueue(succ);
            cur = continuation(cur);
            continue;
        } else {
            dst = cur->to()->isa_lambda();
        }

        if (dst == nullptr) {
            std::cout << "dst is nullptr: " << cur->unique_name() << std::endl;
            return;
        }

        if (dst == end) {
            std::cout << "end: " << end->unique_name() << std::endl;
            return;
        }

        done_.insert(cur);

        if (dst->empty()) {
            if (dst == world().branch()) {
                Scope scope(top);
                auto& postdomtree = *scope.b_cfg()->domtree();
                if (auto n = scope._cfa()->lookup(cur)) {
                    cur = postdomtree.lookup(n)->idom()->lambda();
                    continue;
                }
                std::cout << "no postdom found: " << cur->unique_name() << std::endl;
                return;
            } else
                cur = continuation(cur);
            continue;
        }

        Call call(dst);
        bool all = true;
        for (size_t i = 0; i != cur->num_args(); ++i) {
            if (!cur->arg(i)->isa<Hlt>())
                call.arg(i) = cur->arg(i);
            else
                all = false;
        }

        if (auto cached = find(cache_, call)) { // check for cached version
            rewrite_jump(cur, cached, call);
            std::cout << "using cached call: " << std::endl;
            cur->dump_head();
            cur->dump_jump();
            return;
        } else {                                // no cached version found... create a new one
            Scope scope(dst);
            Type2Type type2type;
            bool res = dst->type()->infer_with(type2type, cur->arg_fn_type());
            assert(res);
            auto dropped = drop(scope, call.args(), type2type);
            rewrite_jump(cur, dropped, call);
            if (all) {
                cur->jump(dropped->to(), dropped->args());
                done_.erase(cur);
            } else
                cur = dropped;
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

//------------------------------------------------------------------------------

void partial_evaluation(World& world) {
    PartialEvaluator(world).seek();
    std::cout << "PE done" << std::endl;

    for (auto primop : world.primops()) {
        if (auto evalop = Def(primop)->isa<EvalOp>())
            evalop->replace(evalop->def());
    }
}

//------------------------------------------------------------------------------

}
