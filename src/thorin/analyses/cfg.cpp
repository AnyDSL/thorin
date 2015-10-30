#include "thorin/analyses/cfg.h"

#include <fstream>
#include <stack>

#include "thorin/primop.h"
#include "thorin/analyses/domfrontier.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/ycomp.h"
#include "thorin/util/log.h"
#include "thorin/util/queue.h"

namespace thorin {

//------------------------------------------------------------------------------

struct CFNodeHash {
    uint64_t operator() (const CFNodeBase* n) const;
};

typedef thorin::HashSet<const CFNodeBase*, CFNodeHash> CFNodeSet;

/// Any jumps targeting a @p Lambda or @p Param outside the @p CFA's underlying @p Scope target this node.
class OutNode : public CFNodeBase {
public:
    OutNode(const CFNode* context, Def def)
        : CFNodeBase(def)
        , context_(context)
    {
        assert(def->isa<Param>() || def->isa<Lambda>());
    }

    const CFNode* context() const { return context_; }
    const CFNodeSet& ancestors() const { return ancestors_; }
    virtual std::ostream& stream(std::ostream&) const override;

private:
    const CFNode* context_;
    mutable CFNodeSet ancestors_;

    friend class CFABuilder;
};

uint64_t CFNodeHash::operator() (const CFNodeBase* n) const {
    if (auto in = n->isa<CFNode>())
        return hash_value(in->lambda()->gid());
    auto out = n->as<OutNode>();
    return hash_combine(hash_value(out->def()->gid()), out->context()->lambda()->gid());
}

void CFNode::link(const CFNode* other) const {
    this ->succs_.push_back(other);
    other->preds_.push_back(this);
}

std::ostream& CFNode::stream(std::ostream& out) const {
    return streamf(out, "%", lambda()->unique_name());
}

std::ostream& OutNode::stream(std::ostream& out) const {
    return streamf(out, "[Out: % (%)]", def()->unique_name(), context());
}

//------------------------------------------------------------------------------

class CFABuilder : public YComp {
public:
    CFABuilder(CFA& cfa)
        : YComp(cfa.scope(), "cfa")
        , cfa_(cfa)
        , lambda2param2nodes_(cfa.scope(), std::vector<CFNodeSet>(0))
        , entry_(in_node(scope().entry()))
        , exit_ (in_node(scope().exit()))
    {
        ILOG("*** CFA: %", scope().entry()->unique_name());
        ILOG_SCOPE(propagate_higher_order_values());
        ILOG_SCOPE(run_cfa());
        ILOG_SCOPE(build_cfg());
        ILOG_SCOPE(unreachable_node_elimination());
        ILOG_SCOPE(link_to_exit());
        ILOG_SCOPE(transetive_cfg());
#ifndef NDEBUG
        ILOG_SCOPE(verify());
#endif
    }

    ~CFABuilder() {
        for (auto p : out_nodes_) {
            for (auto q : p.second)
                delete q.second;
        }
    }

    void propagate_higher_order_values();
    void run_cfa();
    void build_cfg();
    void unreachable_node_elimination();
    void link_to_exit();
    void transetive_cfg();
    void verify();
    virtual void stream_ycomp(std::ostream& out) const override;

    const CFA& cfa() const { return cfa_; }
    const CFNode* entry() const { return entry_; }
    const CFNode* exit() const { return exit_; }

    CFNodeSet cf_nodes(const CFNode*, size_t i);
    CFNodeSet to_cf_nodes(const CFNode* in) { return in->lambda()->empty() ? CFNodeSet() : cf_nodes(in, 0); }
    Array<CFNodeSet> args_cf_nodes(const CFNode*);

    const CFNode* in_node(Lambda* lambda) {
        assert(scope().outer_contains(lambda));
        if (auto in = find(cfa().nodes(), lambda))
            return in;
        auto in = cfa_.nodes_[lambda] = new CFNode(lambda);
        lambda2param2nodes_[lambda].resize(lambda->num_params()); // make room for params
        return in;
    }

    const OutNode* out_node(const CFNode* in, Def def) {
        if (auto out = find(out_nodes_[in], def))
            return out;
        return out_nodes_[in][def] = new OutNode(in, def);
    }

    const OutNode* out_node(const OutNode* ancestor, const CFNode* in) {
        auto out = out_node(in, ancestor->def());
        out->ancestors_.insert(ancestor);
        return out;
    }

    CFNodeSet& param2nodes(const Param* param) {
        in_node(param->lambda()); // alloc CFNode and make room in lambda2param2nodes_
        return lambda2param2nodes_[param->lambda()][param->index()];
    }

    void link(const CFNodeBase* src, const CFNodeBase* dst) {
        DLOG("% -> %", src, dst);

        assert(src->f_index_ == CFNode::Reachable);
        dst->f_index_ = CFNode::Reachable;

        auto p = edges_[src].insert(dst);

        // recursively link ancestors
        if (p.second) {
            if (auto out = dst->isa<OutNode>()) {
                for (auto ancestor : out->ancestors())
                    link(out, ancestor);
            }
        }
    }

private:
    CFA& cfa_;
    Scope::Map<std::vector<CFNodeSet>> lambda2param2nodes_; ///< Maps param in scope to CFNodeSet.
    DefMap<DefSet> def2set_;
    HashMap<const CFNodeBase*, CFNodeSet, CFNodeHash> edges_;
    const CFNode* entry_;
    const CFNode* exit_;
    size_t num_out_nodes_ = 0;
    mutable HashMap<const CFNode*, DefMap<const OutNode*>> out_nodes_;
};

void CFABuilder::propagate_higher_order_values() {
    std::stack<Def> stack;

    auto push = [&] (Def def) -> bool {
        auto p = def2set_.emplace(def, DefSet());
        if (p.second) { // if first insert
            if (def->order() > 0) {
                DLOG("pushing %", def->unique_name());
                stack.push(def);
                return true;
            }
        }
        return false;
    };

    for (auto lambda : scope()) {
        for (auto op : lambda->ops()) {
            push(op);

            while (!stack.empty()) {
                auto def = stack.top();
                auto& set = def2set_[def];

                if (def->isa<Param>() || def->isa<Lambda>()) {
                    assert(def->order() > 0);
                    set.insert(def);
                    stack.pop();
                } else {
                    bool todo = false;
                    for (auto op : def->as<PrimOp>()->ops())
                        todo |= push(op);
                    if (!todo) {
                        for (auto op : def->as<PrimOp>()->ops())
                            set.insert_range(def2set_[op]);
                        stack.pop();
                    }
                }
            }
        }
    }
}

CFNodeSet CFABuilder::cf_nodes(const CFNode* in, size_t i) {
    CFNodeSet result;
    auto cur_lambda = in->lambda();

    for (auto def : def2set_[cur_lambda->op(i)]) {
        assert(def->order() > 0);

        if (auto lambda = def->isa_lambda()) {
            if (scope().inner_contains(lambda))
                result.insert(in_node(lambda));
            else
                result.insert(out_node(in, lambda));
        } else {
            auto param = def->as<Param>();
            if (scope().inner_contains(param)) {
                const auto& set = param2nodes(param);
                for (auto n : set) {
                    if (auto out = n->isa<OutNode>())
                        result.insert(out_node(out, in)); // create a new context if applicable
                    else
                        result.insert(n->as<CFNode>());
                }
            } else
                result.insert(out_node(in, param));
        }
    }

    return result;
}

Array<CFNodeSet> CFABuilder::args_cf_nodes(const CFNode* in) {
    Array<CFNodeSet> result(in->lambda()->num_args());
    for (size_t i = 0; i != result.size(); ++i)
        result[i] = cf_nodes(in, i+1); // shift by one due to args/ops discrepancy
    return result;
}

void CFABuilder::run_cfa() {
    std::queue<Lambda*> queue;

    auto enqueue = [&] (const CFNode* in) {
        DLOG("enqueuing %", in->lambda()->unique_name());
        queue.push(in->lambda());
        in->f_index_ = CFNode::Unfresh;
    };

    enqueue(entry());

    while (!queue.empty()) {
        auto cur_lambda = pop(queue);
        auto cur_in = in_node(cur_lambda);
        auto args = args_cf_nodes(cur_in);

        for (auto n : to_cf_nodes(cur_in)) {
            if (n->def()->type() != cur_lambda->to()->type())
                continue;

            if (auto in = n->isa<CFNode>()) {
                bool todo = in->f_index_ == CFNode::Fresh;
                for (size_t i = 0; i != cur_lambda->num_args(); ++i) {
                    if (in->lambda()->param(i)->order() > 0)
                        todo |= lambda2param2nodes_[in->lambda()][i].insert_range(args[i]);
                }
                if (todo)
                    enqueue(in);

            } else {
                auto out = n->as<OutNode>();
                assert(in_node(cur_lambda) == out->context() && "OutNode's context does not match");

                for (const auto& nodes : args) {
                    for (auto n : nodes) {
                        if (auto in = n->isa<CFNode>()) {
                            bool todo = in->f_index_ == CFNode::Fresh;
                            for (size_t i = 0; i != in->lambda()->num_params(); ++i) {
                                if (in->lambda()->param(i)->order() > 0)
                                    todo |= lambda2param2nodes_[in->lambda()][i].insert(out).second;
                            }
                            if (todo)
                                enqueue(in);
                        }
                    }
                }
            }
        }
    }
}

void CFABuilder::build_cfg() {
    std::queue<const CFNode*> queue;

    auto enqueue = [&] (const CFNode* in) {
        if (in->f_index_ != CFNode::Reachable) {
            queue.push(in);
            DLOG("enqueuing %", in);
        }
    };

    queue.push(entry());
    entry()->f_index_ = exit()->f_index_ = CFNode::Reachable;

    while (!queue.empty()) {
        auto cur_in = pop(queue);
        for (auto n : to_cf_nodes(cur_in)) {
            if (auto in = n->isa<CFNode>()) {
                enqueue(in);
                link(cur_in, in);
            } else {
                auto out = n->as<OutNode>();
                link(cur_in, out);
                for (const auto& nodes : args_cf_nodes(cur_in)) {
                    for (auto n : nodes) {
                        if (auto in = n->isa<CFNode>()) {
                            enqueue(in);
                            link(out, n);
                        } else if (!n->as<OutNode>()->ancestors().empty())
                            link(out, n);
                    }
                }
            }
        }
    }
}

void CFABuilder::unreachable_node_elimination() {
    for (auto in : cfa().nodes()) {
        if (in->f_index_ == CFNode::Reachable) {
            ++cfa_.size_;

            auto& out_nodes = out_nodes_[in];
            for (auto i = out_nodes.begin(); i != out_nodes.end();) {
                auto out = i->second;
                if (out->f_index_ == CFNode::Reachable) {
                    ++i;
                    ++num_out_nodes_;
                } else {
                    DLOG("removing: %", out);
                    i = out_nodes.erase(i);
                    delete out;
                }
            }
        } else {
#ifndef NDEBUG
            for (auto p : out_nodes_[in])
                assert(p.second->f_index_ != CFNode::Reachable);
#endif
            DLOG("removing: %", in);
            cfa_.nodes_[in->lambda()] = nullptr;
            delete in;
        }
    }
}

void CFABuilder::verify() {
    bool error = false;
    for (auto in : cfa().nodes()) {
        if (in != entry() && in->preds_.size() == 0) {
            WLOG("missing predecessors: %", in->lambda()->unique_name());
            error = true;
        }
    }

    if (error) {
        ycomp();
        abort();
    }
}

void CFABuilder::link_to_exit() {
    auto link_dead_end_to_exit = [&] (const CFNodeBase* n) {
        if (edges_[n].empty() && n != exit())
            link(n, exit());
    };

    for (auto in : cfa().nodes()) {
        link_dead_end_to_exit(in);
        for (auto p : out_nodes_[in]) {
            auto out = p.second;
            link_dead_end_to_exit(out);
            if (out->ancestors().empty() && out->def()->isa<Param>())
                link(out, exit());
        }
    }

    // TODO deal with endless loops
}

void CFABuilder::transetive_cfg() {
    std::queue<const CFNodeBase*> queue;

    auto link_to_succs = [&] (const CFNode* src) {
        auto enqueue = [&] (const CFNodeBase* n) {
            auto i = edges_.find(n);
            for (auto succ : i->second)
                queue.push(succ);
        };

        enqueue(src);

        while (!queue.empty()) {
            auto n = pop(queue);
            if (auto dst = n->isa<CFNode>())
                src->link(dst);
            else
                enqueue(n);
        }
    };

    for (auto p : edges_) {
        if (auto in = p.first->isa<CFNode>())
            link_to_succs(in);
    }
}

void CFABuilder::stream_ycomp(std::ostream& out) const {
    std::vector<const CFNodeBase*> nodes(cfa().nodes().begin(), cfa().nodes().end());
    for (auto p : out_nodes_) {
        for (auto q : p.second)
            nodes.push_back(q.second);
    }

    thorin::ycomp(out, YCompOrientation::TopToBottom, scope(), range(nodes),
        [&] (const CFNodeBase* n) { return edges_.find(n)->second; }
    );
}

//------------------------------------------------------------------------------

CFA::CFA(const Scope& scope)
    : scope_(scope)
    , nodes_(scope)
{
    CFABuilder cfa(*this);
}

CFA::~CFA() {
    for (auto n : nodes_.array()) delete n;
}

const F_CFG& CFA::f_cfg() const { return lazy_init(this, f_cfg_); }
const B_CFG& CFA::b_cfg() const { return lazy_init(this, b_cfg_); }

//------------------------------------------------------------------------------

template<bool forward>
CFG<forward>::CFG(const CFA& cfa)
    : YComp(cfa.scope(), forward ? "f_cfg" : "b_cfg")
    , cfa_(cfa)
    , rpo_(*this)
{
#ifndef NDEBUG
    assert(post_order_visit(entry(), size()) == 0);
#else
    post_order_visit(entry(), size());
#endif
}

template<bool forward>
size_t CFG<forward>::post_order_visit(const CFNode* n, size_t i) {
    auto& n_index = forward ? n->f_index_ : n->b_index_;
    assert(n_index == CFNode::Reachable);
    n_index = CFNode::Visited;

    for (auto succ : succs(n)) {
        if (index(succ) == CFNode::Reachable)
            i = post_order_visit(succ, i);
    }

    n_index = i-1;
    rpo_[n] = n;
    return n_index;
}

template<bool forward>
void CFG<forward>::stream_ycomp(std::ostream& out) const {
    thorin::ycomp(out, YCompOrientation::TopToBottom, scope(), range(rpo()),
        [] (const CFNode* n) { return range(n->succs()); }
    );
}

template<bool forward> const DomTreeBase<forward>& CFG<forward>::domtree() const { return lazy_init(this, domtree_); }
template<bool forward> const LoopTree<forward>& CFG<forward>::looptree() const { return lazy_init(this, looptree_); }
template<bool forward> const DomFrontierBase<forward>& CFG<forward>::domfrontier() const { return lazy_init(this, domfrontier_); }

template class CFG<true>;
template class CFG<false>;

//------------------------------------------------------------------------------

}
