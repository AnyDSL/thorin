#include <iostream>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <queue>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/be/thorin.h"
#include "thorin/util/hash.h"

namespace thorin {

//------------------------------------------------------------------------------

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
        , args_(to->num_args())
    {}

    Lambda* to() const { return to_; }
    ArrayRef<Def> args() const { return args_; }
    Def& arg(size_t i) { return args_[i]; }
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
        , n_(n)
        , is_within_(is_within)
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
    int n_;
    bool is_within_;
};

//------------------------------------------------------------------------------

class PartialEvaluator {
public:
    PartialEvaluator(World& world)
        : world_(world)
        , scope_(world, top_level_lambdas(world))
        , loops_(scope_)
    {
        done_.insert(nullptr);
        loops_.dump();
        collect_headers(loops_.root());
        for (auto lambda : world.lambdas())
            new2old_[lambda] = lambda;
    }

    void collect_headers(const LoopNode*);
    void process();
    void rewrite_jump(Lambda* lambda, const Call&);
    void remove_runs(Lambda* lambda);
    void update_new2old(const Def2Def& map);
    Edge edge(Lambda* src, Lambda* dst) const;
    TraceEntry trace_entry(Lambda* lambda) { return TraceEntry(lambda, new2old_[lambda]); }
    TraceEntry ghost_entry() { 
        TraceEntry entry(nullptr, nullptr); 
        entry.todo();
        return entry;
    }
    void push(Lambda* src, ArrayRef<Lambda*> dst);
    Lambda* pop();

    std::list<TraceEntry>::iterator search_loop(const Edge& edge, std::function<void(TraceEntry&)> body = [] (TraceEntry&) {}) {
        auto i = trace_.rbegin();
        if (edge.order() <= 0) {
            int num = -edge.order()/2 + 1;
            for (; num != 0; ++i) {
                assert(i != trace_.rend());
                if (is_header(i->nlambda())) {
                    --num;
                    body(*i);
                }
            }
        }
        return i.base();
    }

    const LoopHeader* is_header(Lambda* lambda) const {
        if (lambda) {
            auto i = lambda2header_.find(new2old_[lambda]);
            if (i != lambda2header_.end())
                return i->second;
            return nullptr;
        }
        return loops_.root();
    }

    void dump_trace() {
        std::cout << "*** trace ***" << std::endl;
        for (auto entry : trace_)
            entry.dump();
        std::cout << "*************" << std::endl;
    }

    World& world_;
    Scope scope_;
    LoopTree loops_;
    Lambda2Lambda new2old_;
    std::unordered_map<Lambda*, const LoopHeader*> lambda2header_;
    std::unordered_set<Lambda*> done_;
    std::unordered_set<Call, CallHash> cache_;
    std::list<TraceEntry> trace_;
    Def2Def old2new;
};

void PartialEvaluator::push(Lambda* src, ArrayRef<Lambda*> dst) {
    Array<Edge> edges(dst.size());
    for (size_t i = 0, e = dst.size(); i != e; ++i)
        edges[i] = edge(src, dst[i]);

    std::stable_sort(edges.begin(), edges.end());

    if (dst.size() > 1) {
        for (auto& edge : edges)
            search_loop(edge, [&] (TraceEntry& entry) { entry.set_evil(); });
    }

    for (auto& edge : edges) {
        auto i = done_.find(edge.dst());
        if (i != done_.end())
            continue;
        done_.insert(edge.dst());
        if (edge.order() <= 0) { // more elegant here
            auto j = search_loop(edge);
            trace_.insert(j, trace_entry(edge.dst()));
        } else {
            if (edge.is_cross()) {
                for (auto i = 0; i < edge.n(); ++i)
                    trace_.push_back(ghost_entry());
            }
            trace_.push_back(trace_entry(edge.dst()));
        }
    }
}

Lambda* PartialEvaluator::pop() {
    while (!trace_.empty() && !trace_.back().todo())
        trace_.pop_back();
    return trace_.empty() ? nullptr : trace_.back().nlambda();
}

Edge PartialEvaluator::edge(Lambda* nsrc, Lambda* ndst) const {
    auto src = new2old_[nsrc];
    auto dst = new2old_[ndst];
    auto hsrc = loops_.lambda2header(src);
    auto hdst = loops_.lambda2header(dst);

    if (is_header(dst)) {
        if (loops_.contains(hsrc, dst))
            return Edge(nsrc, ndst, true, hdst->depth() - hsrc->depth()); // within n, n positive
        if (loops_.contains(hdst, src))
            return Edge(nsrc, ndst, true, hsrc->depth() - hdst->depth()); // within n, n negative
    }

    return Edge(nsrc, ndst, false, hdst->depth() - hsrc->depth());        // cross n
}

void PartialEvaluator::collect_headers(const LoopNode* n) {
    if (const LoopHeader* header = n->isa<LoopHeader>()) {
        for (auto lambda : header->lambdas())
            lambda2header_[lambda] = header;
        for (auto child : header->children())
            collect_headers(child);
    }
}

void PartialEvaluator::process() {
    for (auto src : top_level_lambdas(world_)) {
        trace_.clear();
        if (src->empty())
            continue;
        trace_.push_back(trace_entry(src));

        while ((src = pop())) {
            std::cout << "---------------------------------" << std::endl;
            std::cout << "*** src: " << src->unique_name() << std::endl;
            dump_trace();
            emit_thorin(world_);

            if (src->empty())
                continue;

            auto succs = src->direct_succs();
            bool fold = false;

            auto to = src->to();
            if (auto run = to->isa<Run>()) {
                to = run->def();
                fold = true;
            }

            Lambda* dst = to->isa_lambda();

            if (dst == nullptr) {
                push(src, succs);
                continue;
            }

            Call call(dst);
            for (size_t i = 0, e = dst->num_params(); i != e; ++i) {
                auto arg = src->arg(i);
                call.arg(i) = arg->is_const() ? arg : nullptr;
            }

            if (!fold) {
                push(src, {dst});
                continue;
            } else {
                if (auto header = is_header(new2old_[dst])) {
                    auto e = edge(src, dst);
                    //e.dump();
                    assert(e.is_within());
                    if (e.n() <= 0) {
                        auto i = search_loop(e);
                        if (!is_header(i->nlambda())->is_root()) {
                            assert(header == is_header(i->nlambda()) && "headers don't match");
                            if (i->is_evil())
                                continue;
                        }
                    }
                }
            }

            // check for cached version
            auto i = cache_.find(call);
            if (i != cache_.end())
                rewrite_jump(src, *i);
            else {
#if 0
                // no no cached version found... create a new one
                Scope scope(dst);
                Def2Def f_map;
                GenericMap generic_map;
                bool res = dst->type()->infer_with(generic_map, src->arg_pi());
                assert(res);
                auto f_to = drop(scope, f_map, fcache.idxs(), fcache.args(), generic_map);
                std::cout << "dropped: " << f_to->unique_name() << std::endl;
                for (auto arg : fcache.args())
                    arg->dump();
                f_map[to] = f_to;
                update_new2old(f_map);

                if (f_to->to()->isa_lambda()
                        || (f_to->to()->isa<Run>() && f_to->to()->as<Run>()->def()->isa_lambda())) {
                    rewrite_jump(src, f_to, fcache);
                    for (auto lambda : scope.rpo()) {
                        auto mapped = f_map[lambda]->as_lambda();
                        if (mapped != lambda)
                            mapped->update_to(world_.run(mapped->to()));
                    }
                    push(src, {f_to});
                } else {
                    Def2Def r_map;
                    auto r_to = drop(scope, r_map, rcache.idxs(), rcache.args(), generic_map);
                    std::cout << "dropped: " << r_to->unique_name() << std::endl;
                    for (auto arg : rcache.args())
                        arg->dump();
                    r_map[to] = r_to;
                    update_new2old(r_map);
                    rewrite_jump(src, r_to, rcache);
                    push(src, {r_to});
                }
#endif
            }
        }
    }
}

#if 0
void PartialEvaluator::rewrite_jump(Lambda* lambda, Lambda* to, Cache &cache) {
    std::vector<Def> new_args;
    size_t x = 0;
    for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
        if (x < cache.idxs().size() && i == cache.idxs()[x])
            ++x;
        else
            new_args.push_back(lambda->arg(i));
    }

    lambda->jump(to, new_args);
    cache2lambda_[cache] = to;
}
#endif

void PartialEvaluator::remove_runs(Lambda* lambda) {
    for (size_t i = 0, e = lambda->size(); i != e; ++i) {
        if (auto run = lambda->op(i)->isa<Run>())
            lambda->update_op(i, run->def());
    }
}

void PartialEvaluator::update_new2old(const Def2Def& old2new) {
    for (auto p : old2new) {
        if (auto olambda = p.first->isa_lambda()) {
            auto nlambda = p.second->as_lambda();
            //std::cout << nlambda->unique_name() << " -> "  << olambda->unique_name() << std::endl;
            assert(new2old_.contains(olambda));
            new2old_[nlambda] = new2old_[olambda];
        }
    }
}

//------------------------------------------------------------------------------

void partial_evaluation(World& world) { 
    PartialEvaluator(world).process(); 

    for (auto primop : world.primops()) {
        if (auto evalop = primop->isa<EvalOp>())
            evalop->replace(evalop->def());
    }
}

//------------------------------------------------------------------------------

}
