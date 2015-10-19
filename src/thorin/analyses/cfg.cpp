#include "thorin/analyses/cfg.h"

#include <fstream>
#include <stack>

#include "thorin/primop.h"
#include "thorin/analyses/dfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/ycomp.h"
#include "thorin/util/log.h"
#include "thorin/util/queue.h"

namespace thorin {

//------------------------------------------------------------------------------

uint64_t CFNodeHash::operator() (const CFNode* n) const {
    if (auto in = n->isa<InNode>())
        return hash_value(in->lambda()->gid());
    auto out = n->as<OutNode>();
    // TODO include ancestor?
    return hash_combine(hash_value(out->def()->gid()), out->context()->lambda()->gid());
}

//------------------------------------------------------------------------------

void CFNode::link(const CFNode* other) const {
    DLOG("% -> %", this, other);

    assert(this->f_index_ == CFNode::Reachable);
    other->f_index_ = CFNode::Reachable;

    this->succs_.insert(other);
    auto p = other->preds_.insert(this);

    // recursively link ancestors
    if (p.second) {
        if (auto out = other->isa<OutNode>()) {
            if (auto ancestor = out->ancestor())
                out->link(ancestor);
        }
    }
}

std::ostream& InNode::stream(std::ostream& out) const {
    return streamf(out, "<In: %>", lambda()->unique_name());
}

std::ostream& OutNode::stream(std::ostream& out) const {
    return streamf(out, "[Out: %, %]", def()->unique_name(), context());
}

InNode::~InNode() {
    for (auto p : out_nodes_)
        delete p.second;
}

//------------------------------------------------------------------------------

class CFABuilder {
public:
    CFABuilder(CFA& cfa)
        : cfa_(cfa)
        , lambda2param2nodes_(cfa.scope(), std::vector<CFNodeSet>(0))
    {
        ILOG("*** CFA: propagating higher-order values: start");
        propagate();
        ILOG("*** CFA: propagating higher-order values: done");
        ILOG("*** CFA: actual CFA: start");
        in_node(scope().entry());
        in_node(scope().exit() );
        run_cfa();
        ILOG("*** CFA: actual CFA: done");
        ILOG("*** CFA: build CFG: start");
        build_cfg();
        ILOG("*** CFA: build CFG: done");
    }

    void propagate();
    void run_cfa();
    void build_cfg();

    const CFA& cfa() const { return cfa_; }
    const Scope& scope() const { return cfa_.scope(); }
    Array<CFNodeSet> cf_nodes(const InNode*);

    const InNode* in_node(Lambda* lambda) {
        assert(scope().outer_contains(lambda));
        if (auto in = find(cfa().in_nodes(), lambda))
            return in;
        auto in = cfa_.in_nodes_[lambda] = new InNode(lambda);
        lambda2param2nodes_[lambda].resize(lambda->num_params()); // make room for params
        return in;
    }

    const OutNode* out_node(const InNode* in, const OutNode* ancestor, Def def) {
        if (auto out = find(in->out_nodes_, def))
            return out;
        return in->out_nodes_[def] = new OutNode(in, ancestor, def);
    }

    CFNodeSet& param2nodes(const Param* param) {
        in_node(param->lambda()); // alloc InNode and make room in lambda2param2nodes_
        return lambda2param2nodes_[param->lambda()][param->index()];
    }

private:
    CFA& cfa_;
    Scope::Map<std::vector<CFNodeSet>> lambda2param2nodes_; ///< Maps param in scope to CFNodeSet.
    DefMap<DefSet> def2set_;
};

void CFABuilder::propagate() {
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
        for (auto op : lambda->ops())
            push(op);
    }

    while (!stack.empty()) {
        auto def = stack.top();
        auto& set = def2set_[def];

        if (def->isa<Param>() || def->isa<Lambda>()) {
            set.insert(def);
            stack.pop();
        } else {
            bool todo = false;
            for (auto op : def->as<PrimOp>()->ops())
                todo |= push(op);
            if (!todo) {
                for (auto op : def->as<PrimOp>()->ops()) {
                    auto& op_set = def2set_[op];
                    set.insert(op_set.begin(), op_set.end());
                }
                stack.pop();
            }
        }
    }
}

Array<CFNodeSet> CFABuilder::cf_nodes(const InNode* in) {
    auto cur_lambda = in->lambda();

    // create dummy empty set entry for lambdas without body
    if (cur_lambda->empty())
        return Array<CFNodeSet>(1);

    size_t num = cur_lambda->size();
    Array<CFNodeSet> result(num);

    for (size_t i = 0; i != num; ++i) {
        for (auto def : def2set_[cur_lambda->op(i)]) {
            assert(def->order() > 0);

            if (auto lambda = def->isa_lambda()) {
                if (scope().inner_contains(lambda))
                    result[i].insert(in_node(lambda));
                else
                    result[i].insert(out_node(in, nullptr, lambda));
            } else {
                auto param = def->as<Param>();
                if (scope().inner_contains(param)) {
                    const auto& set = param2nodes(param);
                    for (auto n : set) {
                        if (auto out = n->isa<OutNode>()) {
                            result[i].insert(out_node(in, out, out->def())); // create a new context if applicable
                        } else
                            result[i].insert(n->as<InNode>());
                    }
                } else
                    result[i].insert(out_node(in, nullptr, param));
            }
        }
    }

    return result;
}

void CFABuilder::run_cfa() {
    std::queue<Lambda*> queue;

    auto enqueue = [&] (const InNode* in) {
        DLOG("enqueuing %", in->lambda()->unique_name());
        queue.push(in->lambda());
        in->f_index_ = CFNode::Unreachable;
    };

    enqueue(cfa().entry());

    while (!queue.empty()) {
        auto cur_lambda = pop(queue);
        auto info = cf_nodes(in_node(cur_lambda));
        size_t num = cur_lambda->size();

        for (auto n : info.front()) {
            if (auto in = n->isa<InNode>()) {
                bool todo = in->f_index_ == CFNode::Fresh;
                for (size_t i = 1; i != num; ++i) {
                    const auto& set = info[i];
                    todo |= lambda2param2nodes_[in->lambda()][i-1].insert(set.begin(), set.end());
                }

                if (todo)
                    enqueue(in);

            } else {
                auto out = n->as<OutNode>();
                assert(in_node(cur_lambda) == out->context() && "OutNode's context does not match");

                for (const auto& nodes : info.skip_front()) {
                    for (auto n : nodes) {
                        if (auto in = n->isa<InNode>()) {
                            auto in_lambda = in->lambda();
                            bool todo = in->f_index_ == CFNode::Fresh;
                            for (size_t p = 0; p != in_lambda->num_params(); ++p) {
                                if (in_lambda->param(p)->order() > 0)
                                    todo |= lambda2param2nodes_[in_lambda][p].insert(out).second;
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
    std::queue<const InNode*> queue;

    auto enqueue = [&] (const InNode* in) {
        if (in->f_index_ == CFNode::Unreachable) {
            queue.push(in);
            DLOG("enqueuing %", in);
        }
    };

    enqueue(cfa().entry());
    cfa().entry()->f_index_ = CFNode::Reachable;

    while (!queue.empty()) {
        auto cur_in = pop(queue);
        auto info = cf_nodes(cur_in);
        for (auto n : info.front()) {
            if (auto in = n->isa<InNode>()) {
                enqueue(in);
                cur_in->link(in);
            } else {
                auto out = n->as<OutNode>();
                cur_in->link(out);
                for (const auto& nodes : info.skip_front()) {
                    for (auto n : nodes) {
                        if (auto in = n->isa<InNode>()) {
                            enqueue(in);
                            out->link(n);
                        }
                    }
                }
            }
        }
    }

    auto exit = cfa().exit();

    for (auto in : cfa().in_nodes()) {
        if (in->f_index_ == CFNode::Reachable) {
            ++cfa_.num_in_nodes_;
            auto& out_nodes = in->out_nodes_;
            for (auto i = out_nodes.begin(); i != out_nodes.end();) {
                auto out = i->second;
                if (out->f_index_ == CFNode::Reachable) {
                    if (out->succs().empty())
                        out->link(exit);
                    ++i;
                    ++cfa_.num_out_nodes_;
                } else {
                    i = out_nodes.erase(i);
                    delete out;
                }
            }
        } else {
#ifndef NDEBUG
            for (auto p : in->out_nodes())
                assert(p.second->f_index_ != CFNode::Reachable);
#endif
            cfa_.in_nodes_[in->lambda()] = nullptr;
            delete in;
        }
    }

#ifndef NDEBUG
    bool error = false;
    for (auto in : cfa().in_nodes()) {
        if (in != cfa().entry() && in->preds_.size() == 0) {
            WLOG("missing predecessors: %", in->lambda()->unique_name());
            error = true;
        }
    }

    if (error)
        cfa().error_dump();
#endif
}

void CFA::error_dump() const {
    std::ofstream out("out.vcg");
    emit_ycomp(scope(), false, out);
    out.close();
    abort();
}

//------------------------------------------------------------------------------

CFA::CFA(const Scope& scope)
    : scope_(scope)
    , in_nodes_(scope)
{
    CFABuilder cfa(*this);
}

CFA::~CFA() {
    for (auto n : in_nodes_.array()) delete n;
}

const F_CFG& CFA::f_cfg() const { return lazy_init(this, f_cfg_); }
const B_CFG& CFA::b_cfg() const { return lazy_init(this, b_cfg_); }

//------------------------------------------------------------------------------

template<bool forward>
CFG<forward>::CFG(const CFA& cfa)
    : cfa_(cfa)
    , rpo_(*this)
{
    size_t result = post_order_visit(entry(), cfa.num_cf_nodes());
#ifndef NDEBUG
    if (result != 0)
        cfa.error_dump();
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

template<bool forward> const DomTreeBase<forward>& CFG<forward>::domtree() const { return lazy_init(this, domtree_); }
template<bool forward> const LoopTree<forward>& CFG<forward>::looptree() const { return lazy_init(this, looptree_); }
template<bool forward> const DFGBase<forward>& CFG<forward>::dfg() const { return lazy_init(this, dfg_); }

template class CFG<true>;
template class CFG<false>;

//------------------------------------------------------------------------------

}
