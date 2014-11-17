#include "thorin/primop.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/util/queue.h"

namespace thorin {

//------------------------------------------------------------------------------

class FlowVal {
public:
    explicit FlowVal()
        : lambdas_(nullptr)
    {}
    explicit FlowVal(LambdaSet& lambdas)
        : lambdas_(&lambdas)
    {}

    const LambdaSet& lambdas() const { return lambdas_ ? *lambdas_ : none_; }
    bool is_valid() const { return lambdas_ != nullptr; }
    bool join(const FlowVal& other);

private:
    static LambdaSet none_;
    LambdaSet* lambdas_;

    friend class CFABuilder;
};

LambdaSet FlowVal::none_;

bool FlowVal::join(const FlowVal& other) {
    bool todo = false;
    if (this->is_valid() && other.is_valid() && this->lambdas_ != other.lambdas_) {
        for (auto lambda : other.lambdas())
            todo |= lambdas_->insert(lambda).second;
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

    CFABuilder(CFA& cfa)
        : cfa_(cfa)
        , lambda2lambdas_(cfa.size())
        , lambda2param2lambdas_(cfa.size(), std::vector<LambdaSet>(0))
        , reachable_(cfa.size(), Unreachable)
    {
        for (size_t sid = 0, e = cfa.size(); sid != e; ++sid) {
            auto lambda = scope()[sid];
            lambda2lambdas_[sid].insert(lambda);                        // only add current lamba to set and that's it
            lambda2param2lambdas_[sid].resize(lambda->num_params());    // make room for params
        }

        run();
    }

    size_t sid(Lambda* lambda) const { return cfa().sid(lambda); }
    const CFA& cfa() const { return cfa_; }
    const Scope& scope() const { return cfa_.scope(); }
    void run();
    bool is_forward_reachable(Lambda* lambda) { return reachable_[cfa_.sid(lambda)] & ForwardReachable; }
    bool is_backward_reachable (Lambda* lambda) { return reachable_[cfa_.sid(lambda)] & BackwardReachable; }
    //bool contains(Lambda* lambda) { return scope().contains(lambda); };
    bool contains(Lambda* lambda) { return scope().entry() != lambda && scope().contains(lambda); };
    bool contains(const Param* param) { return scope().entry() != param->lambda() && contains(param->lambda()); }
    FlowVal flow_val(Def);
    void forward_visit(CFNode* cur);
    void backward_visit(CFNode* cur);

private:
    CFNode* _lookup(Lambda* lambda) const { return cfa_._lookup(lambda); }

    CFA& cfa_;
    std::vector<LambdaSet> lambda2lambdas_;
    std::vector<std::vector<LambdaSet>> lambda2param2lambdas_;
    std::vector<uint8_t> reachable_;
};

FlowVal CFABuilder::flow_val(Def def) {
    if (auto lambda = def->isa_lambda()) {
        if (contains(lambda))
            return FlowVal(lambda2lambdas_[sid(lambda)]);
    } else if (auto param = def->isa<Param>()) {
        if (contains(param))
            return FlowVal(lambda2param2lambdas_[sid(param->lambda())][param->index()]);
    }
    return FlowVal();
}

void search(Def def, std::function<void(Def)> f) {
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

void CFABuilder::run() {
    for (bool todo = true; todo;) { // keep iterating to collect param flow infos until things are stable
        todo = false;
        for (auto lambda : scope()) {
            search(lambda->to(), [&] (Def def) {
                for (auto to : flow_val(def).lambdas()) {
                    for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                        auto arg = lambda->arg(i);
                        if (arg->order() >= 1) {
                            search(arg, [&] (Def def) {
                                todo |= flow_val(to->param(i)).join(flow_val(def));
                            });
                        }
                    }
                }
            });
        }
    }

    // build CFG
    forward_visit(cfa().nodes_.front());
    F_CFG f_cfg(cfa());

    // link with virtual exit
    for (auto n : cfa().nodes_.slice_num_from_end(1)) { // skip virtual exit
        if (is_forward_reachable(n->lambda()) && n->succs_.empty())
            n->link(cfa().nodes_.back()); 
    }

    //// keep linking nodes not reachable from exit
    //for (bool todo = true; todo;) {
        //for (size_t i = f_cfg.size(); i-- != 0;) {

        //}
    //}
}

void CFABuilder::forward_visit(CFNode* cur) {
    assert(!is_forward_reachable(cur->lambda()));
    auto& reachable = reachable_[sid(cur->lambda())];
    auto link_and_visit = [&] (CFNode* succ) {
        assert(contains(succ->lambda()));
        cur->link(succ);
        if (!is_forward_reachable(succ->lambda()))
            forward_visit(succ);
    };

    auto visit_args = [&] {
        for (auto arg : cur->lambda()->args()) {
            search(arg, [&] (Def def) {
                for (auto succ : flow_val(def).lambdas())
                    link_and_visit(_lookup(succ));
            });
        }
    };

    reachable = ForwardReachable;
    search(cur->lambda()->to(), [&] (Def def) {
        if (auto to_lambda = def->isa_lambda()) {
            if (contains(to_lambda))
                link_and_visit(_lookup(to_lambda));
            else
                visit_args();
        } else if (auto param = def->isa<Param>()) {
            if (contains(param)) {
                for (auto succ : flow_val(param).lambdas())
                    link_and_visit(_lookup(succ));
            } else
                visit_args();
        }
    });
    reachable = ForwardReachable;
}

//------------------------------------------------------------------------------

CFA::CFA(const Scope& scope) 
    : scope_(scope)
    , nodes_(scope.size())
{
    for (size_t i = 0, e = size(); i != e; ++i)
        nodes_[i] = new CFNode(scope[i]);

    CFABuilder cfa(*this);
}

size_t CFA::sid(Lambda* lambda) const { 
    if (auto info = lambda->find_scope(&scope()))
        return info->sid;
    return size_t(-1);
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
    , rpo_ids_(cfa.size())
    , rpo_(cfa.nodes()) // copy over - sort later
{
    std::fill(rpo_ids_.begin(), rpo_ids_.end(), -1);    // mark as not visited
    auto num = number(entry(), 0);                      // number in post-order
    
    for (size_t i = 0, e = size(); i != e; ++i) {       // convert to reverse post-order
        auto& rpo_id = rpo_ids_[i];
        if (rpo_id != size_t(-1))
            rpo_id = num-1 - rpo_id;
    }

    // sort in reverse post-order
    std::sort(rpo_.begin(), rpo_.end(), [&] (const CFNode* n1, const CFNode* n2) { return rpo_id(n1) < rpo_id(n2); });
    rpo_.shrink(num);                                   // remove unreachable stuff
}

template<bool forward>
size_t CFG<forward>::number(const CFNode* n, size_t i) {
    auto& n_rpo_id = _rpo_id(n);
    n_rpo_id = -2; // mark as visited

    for (auto succ : succs(n)) {
        if (rpo_id(succ) == size_t(-1)) // if not visited
            i = number(succ, i);
    }

    return (n_rpo_id = i) + 1;
}

template<bool forward>
const DomTreeBase<forward>* CFG<forward>::domtree() const { return lazy_init(this, domtree_); }

template class CFG<true>;
template class CFG<false>;

//------------------------------------------------------------------------------

}
