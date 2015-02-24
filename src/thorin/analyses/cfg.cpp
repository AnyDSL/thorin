#include "thorin/primop.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/util/queue.h"

namespace thorin {

//------------------------------------------------------------------------------

InCFNode::~InCFNode() {
    for (auto p : out_nodes_) 
        delete p.second;
}

//------------------------------------------------------------------------------

struct CFNodeHash {
    size_t operator() (const CFNode* n) const { 
        if (auto in = n->isa<InCFNode>())
            return hash_value(in->lambda()->gid());
        auto out = n->as<OutCFNode>();
        return hash_combine(hash_value(out->def()->gid()), out->parent()->lambda()->gid());
    }
};

typedef thorin::HashSet<const CFNode*, CFNodeHash> CFNodeSet;

//------------------------------------------------------------------------------

static void leaves(Def def, std::function<void(Def)> f) {
    DefSet done;
    std::queue<Def> queue;

    auto enqueue = [&] (Def def) {
        if (!done.contains(def)) {
            queue.push(def);
            done.insert(def);
        }
    };

    enqueue(def);
    while (!queue.empty()) {
        auto def = pop(queue);
        if (def->isa<Param>() || def->isa<Lambda>())
            f(def);
        else {
            for (auto op : def->as<PrimOp>()->ops())
                enqueue(op);
        }
    }
}

class CFABuilder {
public:
    CFABuilder(CFA& cfa);

    const CFA& cfa() const { return cfa_; }
    const Scope& scope() const { return cfa_.scope(); }
    void run();
    //bool contains(Lambda* lambda) { return scope().contains(lambda); };
    bool contains(Lambda* lambda) { return scope().entry() != lambda && scope().contains(lambda); };
    bool contains(const Param* param) { return scope().entry() != param->lambda() && contains(param->lambda()); }
    const CFNode* lookup(Lambda* lambda) const { return cfa_.in_nodes_[lambda]; }
    const OutCFNode* out_node(const InCFNode* in, Lambda* lambda) {
        if (auto out = find(in->out_nodes_, lambda))
            return out;
        return in->out_nodes_[lambda] = new OutCFNode(in, lambda);
    }
    const InCFNode* exit() const { return cfa().exit(); }

private:
    CFA& cfa_;
    Scope::Map<const InCFNode*> lambda2in_node_;            ///< Maps lambda in scope to InCFNode. 
    Scope::Map<std::vector<CFNodeSet>> lambda2param2nodes_; ///< Maps param in scope to CFNodeSet.
};

CFABuilder::CFABuilder(CFA& cfa)
    : cfa_(cfa)
    , lambda2in_node_(cfa.scope())
    , lambda2param2nodes_(cfa.scope(), std::vector<CFNodeSet>(0))
{
    for (auto lambda : scope()) {
        lambda2in_node_[lambda] = new InCFNode(lambda);
        lambda2param2nodes_[lambda].resize(lambda->num_params()); // make room for params
    }

    run();
}

void CFABuilder::run() {
    for (bool todo = true; todo;) {
        todo = false;

        for (auto lambda : scope()) {
            auto in = lambda2in_node_[lambda];
            size_t num = lambda->size();

            // compute for each of lambda's op the current set of CFNodes
            Array<CFNodeSet> info(num);
            for (size_t i = 0; i != num; ++i) {
                leaves(lambda->op(i), [&] (Def def) {
                    if (auto op_lambda = def->isa_lambda()) {
                        const CFNode* n;
                        if (contains(op_lambda))
                            n = lambda2in_node_[op_lambda];
                        else
                            n = out_node(in, op_lambda);
                        info[i].insert(n);
                    } else {
                        auto param = def->as<Param>();
                        if (contains(param)) {
                            const auto& set = lambda2param2nodes_[param->lambda()][param->index()];
                            info[i].insert(set.begin(), set.end());
                        } else
                            info[i].insert(exit());
                    }
                });
            }

            for (auto n : info[0]) {
                if (auto in = n->isa<InCFNode>()) {
                    for (size_t i = 1; i != num; ++i) {
                        const auto& set = info[i];
                        todo |= lambda2param2nodes_[in->lambda()][i-1].insert(set.begin(), set.end());
                    }
                } else {
                    //auto out = n->as<OutCFNode>();
                    for (size_t i = 1; i != num; ++i) {
                        const auto& set = info[i];
                        for (auto n : set) {
                            if (auto info_in = n->isa<InCFNode>()) {
                                for (size_t p = 0; p != info_in->lambda()->num_params(); ++p) {
                                    todo |= lambda2param2nodes_[info_in->lambda()][i-1].insert(set.begin(), set.end());
                                }
                            } else {
                                //auto info_out = n->as<OutCFNode>();
                            }
                        }
                    }
                }
            }
        }
    }
}

#if 0

//------------------------------------------------------------------------------

struct OutCFNodeHash {
    size_t operator() (const OutCFNode* n) const { return n->lambda()->gid(); }
};

typedef thorin::HashMap<const CFNode*, CFNodeSet*, CFNodeHash> OutCFNode2Set;

//------------------------------------------------------------------------------


class FlowVal {
public:
    explicit FlowVal(CFNodeSet& nodes)
        : nodes_(&nodes)
    {}

    const CFNodeSet& nodes() const { return *nodes_; }
    bool join(const FlowVal& other);

private:
    CFNodeSet* nodes_;

    friend class CFABuilder;
};

bool FlowVal::join(const FlowVal& other) {
    bool todo = false;
    if (this->nodes_ != other.nodes_) {
        for (auto n : other.nodes())
            todo |= this->nodes_->insert(n).second;
    }
    return todo;
}

//------------------------------------------------------------------------------

class CFABuilder {
public:
    enum { 
        Unreachable       = 1 << 0, 
        ForwardReachable  = 1 << 1,
        BackwardReachable = 1 << 2,
    };

    CFABuilder(CFA& cfa);
#ifndef NDEBUG
    ~CFABuilder() { assert(exit_.size() == 1); }
#endif

    const CFA& cfa() const { return cfa_; }
    const Scope& scope() const { return cfa_.scope(); }
    void run();
    bool is_forward_reachable(Lambda* lambda) { return reachable_[lambda] & ForwardReachable; }
    bool is_backward_reachable (Lambda* lambda) { return reachable_[lambda] & BackwardReachable; }
    //bool contains(Lambda* lambda) { return scope().contains(lambda); };
    bool contains(Lambda* lambda) { return scope().entry() != lambda && scope().contains(lambda); };
    bool contains(const Param* param) { return scope().entry() != param->lambda() && contains(param->lambda()); }
    FlowVal flow_val(Lambda* src, Def def) { return flow_val(cfa().in_nodes()[src], def); }
    FlowVal flow_val(const InCFNode* src, Def);
    void forward_visit(const CFNode* cur);
    void backward_visit(const CFNode* cur);
    const CFNode* lookup(Lambda* lambda) const { return cfa_.in_nodes_[lambda]; }

private:
    CFA& cfa_;
    Scope::Map<CFNodeSet> lambda2nodes_;                            ///< Maps lambda in scope to CFNodeSet containing InCFNode. 
    Scope::Map<std::vector<CFNodeSet>> lambda2param2nodes_;         ///< Maps param in scope to CFNodeSet.
    OutCFNode2Set out_node2set_;
    CFNodeSet exit_;
    Scope::Map<uint8_t> reachable_;
};

CFABuilder::CFABuilder(CFA& cfa)
    : cfa_(cfa)
    , lambda2nodes_(cfa.scope())
    , lambda2param2nodes_(cfa.scope(), std::vector<CFNodeSet>(0))
    , reachable_(cfa.scope(), Unreachable)
{
    for (auto lambda : scope()) { 
        lambda2nodes_[lambda].insert(new InCFNode(lambda));         // only add current lambda to set and that's it
        lambda2param2nodes_[lambda].resize(lambda->num_params());   // make room for params
    }

    exit_.insert(cfa.exit());
    run();
}

FlowVal CFABuilder::flow_val(const InCFNode* src, Def def) {
    if (auto lambda = def->isa_lambda()) {
        if (contains(lambda))
            return FlowVal(lambda2nodes_[lambda]);
        else {
            auto out = find(src->out_nodes_, lambda);
            if (out == nullptr)
                src->out_nodes_[lambda] = out = new OutCFNode(lambda);
            auto set = find(out_node2set_, out);
            if (set == nullptr) {
                out_node2set_[out] = set = new CFNodeSet();
                set->insert(out);
            }
            return FlowVal(*set);
        }
    } else if (auto param = def->isa<Param>()) {
        if (contains(param))
            return FlowVal(lambda2param2nodes_[param->lambda()][param->index()]);
        else
            return FlowVal(exit_);
    } else
        THORIN_UNREACHABLE;
}

void CFABuilder::run() {
    auto init = [&] (Lambda* lambda) {
        auto in = find(cfa().in_nodes(), lambda);
        if (in == nullptr) {
            in = new InCFNode(lambda);
            cfa_.in_nodes_[lambda] = in;
            lambda2nodes_[lambda].insert(in);
            lambda2param2nodes_[lambda].resize(lambda->num_params());   // make room for params
        }
    };

    init(scope().entry());

    for (bool todo = true; todo;) { // keep iterating to collect param flow infos until things are stable
        todo = false;
        for (auto lambda : scope()) {
            if (auto in = find(cfa().in_nodes(), lambda)) {
                leaves(lambda->to(), [&] (Def def) {
                    //for (auto to : flow_val(src, def).nodes()) {
                        //for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                            //auto arg = lambda->arg(i);
                            //if (arg->order() >= 1) {
                                //leaves(arg, [&] (Def def) {
                                    //todo |= flow_val(src, to->param(i)).join(flow_val(src, def));
                                //});
                            //}
                        //}
                    //}
                    auto val = flow_val(src, def);
                    for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                        auto arg = lambda->arg(i);
                        if (arg->order() >= 1) {
                            leaves(arg, [&] (Def def) {
                                todo |= val.join(flow_val(src, def));
                            });
                        }
                    }
                //}
            }
        }
    }


            //});
        //}
    //}
}

#if 0
void CFABuilder::run() {
    for (bool todo = true; todo;) { // keep iterating to collect param flow infos until things are stable
        todo = false;
        for (auto lambda : scope()) {
            auto src = cfa().in_nodes_[lambda];
            leaves(lambda->to(), [&] (Def def) {
                //for (auto to : flow_val(src, def).nodes()) {
                    //for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                        //auto arg = lambda->arg(i);
                        //if (arg->order() >= 1) {
                            //leaves(arg, [&] (Def def) {
                                //todo |= flow_val(src, to->param(i)).join(flow_val(src, def));
                            //});
                        //}
                    //}
                //}
                auto val = flow_val(src, def);
                for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                    auto arg = lambda->arg(i);
                    if (arg->order() >= 1) {
                        leaves(arg, [&] (Def def) {
                            todo |= val.join(flow_val(src, def));
                        });
                    }
                }
            //}

            });
        }
    }

    // build CFG
    forward_visit(cfa().in_nodes_.entry());
    F_CFG f_cfg(cfa());

    // link with virtual exit
    for (auto n : cfa().in_nodes_.array().slice_num_from_end(1)) { // skip virtual exit
        if (is_forward_reachable(n->lambda()) && n->succs_.empty())
            n->link(cfa().in_nodes_.exit()); 
    }

    //// keep linking nodes not reachable from exit
    //for (bool todo = true; todo;) {
        //for (size_t i = f_cfg.size(); i-- != 0;) {

        //}
    //}
}
#endif

void CFABuilder::forward_visit(const CFNode* cur) {
    auto lambda = cur->lambda();
    assert(!is_forward_reachable(lambda));
    auto& reachable = reachable_[lambda];
    auto link_and_visit = [&] (const CFNode* succ) {
        assert(contains(succ->lambda()));
        cur->link(succ);
        if (!is_forward_reachable(succ->lambda()))
            forward_visit(succ);
    };

    auto visit_args = [&] {
        for (auto arg : lambda->args()) {
            auto src = cfa().in_nodes()[lambda];
            leaves(arg, [&] (Def def) {
                for (auto succ : flow_val(src, def).nodes())
                    link_and_visit(succ);
            });
        }
    };

    reachable = ForwardReachable;
    leaves(lambda->to(), [&] (Def def) {
        if (auto to_lambda = def->isa_lambda()) {
            if (contains(to_lambda))
                link_and_visit(lookup(to_lambda));
            else
                visit_args();
        } else if (auto param = def->isa<Param>()) {
            if (contains(param)) {
                for (auto succ : flow_val(lambda, param).nodes())
                    link_and_visit(succ);
            } else
                visit_args();
        }
    });
    reachable = ForwardReachable;
}

//------------------------------------------------------------------------------

CFA::CFA(const Scope& scope) 
    : scope_(scope)
    , in_nodes_(scope)
    , exit_(new OutCFNode(scope.exit()))
{
    for (size_t i = 0, e = size(); i != e; ++i)
        in_nodes_[scope[i]] = new InCFNode(scope[i]);

    CFABuilder cfa(*this);
}

CFA::~CFA() {
    for (auto n : in_nodes_.array()) delete n;
}

const F_CFG* CFA::f_cfg() const { return lazy_init(this, f_cfg_); }
const B_CFG* CFA::b_cfg() const { return lazy_init(this, b_cfg_); }
const DomTree* CFA::domtree() const { return f_cfg()->domtree(); }
const PostDomTree* CFA::postdomtree() const { return b_cfg()->domtree(); }
const LoopTree* CFA::looptree() const { return looptree_ ? looptree_ : looptree_ = new LoopTree(*f_cfg()); }

//------------------------------------------------------------------------------

template<bool forward>
CFG<forward>::CFG(const CFA& cfa)
    : cfa_(cfa)
    , indices_(cfa.scope(), -1 /*mark as not visited*/)
    , rpo_(*this, cfa.in_nodes().array().begin(), cfa.in_nodes().array().end()) // copy over - sort later
{
    auto num = post_order_number(entry(), 0);
    for (size_t i = 0, e = size(); i != e; ++i) { // convert to reverse post-order
        auto& index = indices_.array()[i];
        if (index != size_t(-1))
            index = num-1 - index;
    }

    // sort in reverse post-order
    std::sort(rpo_.array().begin(), rpo_.array().end(), [&] (const CFNode* n1, const CFNode* n2) { 
        return index(n1) < index(n2); 
    });
    rpo_.array().shrink(num); // remove unreachable stuff
}

template<bool forward>
size_t CFG<forward>::post_order_number(const CFNode* n, size_t i) {
    auto& n_index = indices_[n->lambda()];
    n_index = -2; // mark as visited

    for (auto succ : succs(n)) {
        if (index(succ) == size_t(-1)) // if not visited
            i = post_order_number(succ, i);
    }

    return (n_index = i) + 1;
}

template<bool forward>
const DomTreeBase<forward>* CFG<forward>::domtree() const { return lazy_init(this, domtree_); }

template class CFG<true>;
template class CFG<false>;


#endif

//------------------------------------------------------------------------------

}
