#include <iostream>
#include <list>
#include <unordered_map>
#include <queue>

#include "thorin/literal.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/be/thorin.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/transform/mangle.h"
#include "thorin/transform/merge_lambdas.h"
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

class TraceEntry {
public:
    TraceEntry(Lambda* nlambda, Lambda* olambda) 
        : nlambda_(nlambda)
        , olambda_(olambda)
        , is_evil_(false)
        , todo_(true)
    {}

    bool is_evil() const { return is_evil_; }
    bool todo() { 
        bool old = todo_; 
        todo_ = false; 
        return old; 
    }
    Lambda* olambda() const { return olambda_; }
    Lambda* nlambda() const { return nlambda_; }
    void dump() const {
        if (olambda()) {
            std::cout << olambda()->unique_name() << '/' << nlambda()->unique_name() 
                << " todo: " << todo_ << " evil: " << is_evil_ << std::endl;
        } else
            std::cout << "<ghost entry>" << std::endl;
    }
    void set_evil() { is_evil_ = true; }

private:
    Lambda* nlambda_;
    Lambda* olambda_;
    bool is_evil_;
    bool todo_;
};

//------------------------------------------------------------------------------

class Edge {
public:
    Edge() {}
    Edge(Lambda* src, Lambda* dst, bool is_within, int n)
        : src_(src)
        , dst_(dst)
        , is_within_(is_within)
        , n_(n)
    {}

    Lambda* src() const { return src_; }
    Lambda* dst() const { return dst_; }
    int n() const { return n_; }
    bool is_within() const { return is_within_; }
    bool is_cross() const { return !is_within(); }
    int order() const { return is_within() ? 2*n() : 2*n() + 1; }
    bool operator < (const Edge& other) const { return this->order() < other.order(); }
    void dump() {
        std::cout << (is_within() ? "within " : "cross ") << n() << ": "
                  << src_->unique_name() << " -> " << dst_->unique_name() << std::endl;
    }

private:
    Lambda* src_;
    Lambda* dst_;
    bool is_within_;
    int n_;
};

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
    void dump() const {
        to()->dump_head();
        for (auto def : args()) {
            if (def) 
                def->dump();
            else
                std::cout << "<null>" << std::endl;
        }
    }

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
        postdomtree_.dump();
        for (auto lambda : world.lambdas())
            new2old_[lambda] = lambda;
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
    LambdaSet done_;
    std::unordered_map<Call, Lambda*, CallHash> cache_;
};

void PartialEvaluator::seek() {
    std::queue<Lambda*> queue;
    for (auto lambda : world().externals())
        queue.push(lambda);

    while (!queue.empty()) {
        auto lambda = queue.front();
        queue.pop();

        if (!done_.contains(lambda)) {
            eval(lambda);
            for (auto succ : lambda->succs())
                queue.push(succ);
        }
    }
}

void PartialEvaluator::eval(Lambda* cur) {
    while (true) {
        if (done_.contains(cur))
            break;
        done_.insert(cur);
        if (cur->empty()) 
            break;

        auto to = cur->to();
        if (auto run = to->isa<Run>())
            to = run->def();

        auto dst = to->isa_lambda();
        if (dst == nullptr)
            break;

        Call call(dst);
        bool fold = false;
        for (size_t i = 0; i != cur->num_args(); ++i) {
            call.arg(i) = cur->arg(i)->is_const() ? cur->arg(i) : nullptr;
            fold |= call.arg(i) != nullptr; // don't fold if there is nothing to fold
        }

        if (!fold) {
            cur = dst;
            continue;
        } 
        
        auto i = cache_.find(call);
        if (i != cache_.end()) {    // check for cached version
            rewrite_jump(cur, i->second, call);
            break;
        } else {                    // no no cached version found... create a new one
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
            //std::cout << nlambda->unique_name() << " -> "  << olambda->unique_name() << std::endl;
            assert(new2old_.contains(olambda));
            new2old_[nlambda] = new2old_[olambda];
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
