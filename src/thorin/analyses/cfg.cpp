#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"

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

    friend class CFA;
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

class CFA {
public:
    enum class Color : uint8_t { White, Gray, Black };

    CFA(CFG& cfg)
        : cfg_(cfg)
        , lambda2lambdas_(cfg.size())
        , lambda2param2lambdas_(cfg.size(), std::vector<LambdaSet>(0))
        , colors_(cfg.size(), Color::White)
    {
        for (size_t sid = 0, e = cfg.size(); sid != e; ++sid) {
            auto lambda = scope()[sid];
            lambda2lambdas_[sid].insert(lambda);                        // only add current lamba to set and that's it
            lambda2param2lambdas_[sid].resize(lambda->num_params());    // make room for params
        }

        std::cout << "-----" << std::endl;
        run();
    }

    size_t sid(Lambda* lambda) const { return cfg().sid(lambda); }
    const CFG& cfg() const { return cfg_; }
    const Scope& scope() const { return cfg_.scope(); }
    void run();
    bool is_reachable(Lambda* lambda) { return colors_[cfg_.sid(lambda)] == Color::Black; }
    bool contains(Lambda* lambda) { return scope().contains(lambda); };
    bool contains(const Param* param) { return scope().entry() != param->lambda() && contains(param->lambda()); }
    FlowVal flow_val(Def);
    void visit(CFGNode* prev, CFGNode* cur);

private:
    CFGNode* _lookup(Lambda* lambda) const { return cfg_._lookup(lambda); }

    CFG& cfg_;
    std::vector<LambdaSet> lambda2lambdas_;
    std::vector<std::vector<LambdaSet>> lambda2param2lambdas_;
    std::vector<Color> colors_;
};

FlowVal CFA::flow_val(Def def) {
    if (auto lambda = def->isa_lambda()) {
        if (contains(lambda))
            return FlowVal(lambda2lambdas_[sid(lambda)]);
    } else if (auto param = def->isa<Param>()) {
        if (contains(param))
            return FlowVal(lambda2param2lambdas_[sid(param->lambda())][param->index()]);
    }
    return FlowVal();
}

void CFA::run() {
    for (bool todo = true; todo;) { // keep iterating to collect param flow infos until things are stable
        todo = false;
        for (auto lambda : scope()) {
            for (auto to : flow_val(lambda->to()).lambdas()) {
                for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                    auto arg = lambda->arg(i);
                    if (arg->order() >= 1)
                        todo |= flow_val(to->param(i)).join(flow_val(arg));
                }
            }
        }
    }

    // build CFG
    visit(nullptr, cfg().nodes_.front());

    // link with virtual exit
    for (auto n : cfg().nodes_.slice_num_from_end(1)) {                 // skip virtual exit
        if (is_reachable(n->lambda()) && n->reduced_succs_.empty()) {   // only consider reachable nodes
            n->link(cfg().nodes_.back()); 
            n->reduced_link(cfg().nodes_.back());
        }
    }
}

void CFA::visit(CFGNode* prev, CFGNode* cur) {
    auto& col = colors_[sid(cur->lambda())];

    auto visit_args = [&] {
        for (auto arg : cur->lambda()->args()) {
            if (auto succ = arg->isa_lambda()) {
                if (contains(succ))
                    visit(cur, _lookup(succ));
            }
        }
    };

    switch (col) {
        case Color::White:              // white: not yet visited
            col = Color::Gray;          // mark gray: is on recursion stack
            if (auto to_lambda = cur->lambda()->to()->isa_lambda()) {
                if (contains(to_lambda))
                    visit(cur, _lookup(to_lambda));
                else
                    visit_args();
            } else if (auto param = cur->lambda()->to()->isa<Param>()) {
                if (contains(param)) {
                    for (auto succ : flow_val(param).lambdas())
                        visit(cur, _lookup(succ));
                } else
                    visit_args();
            }
            col = Color::Black;         // mark black: done
            break;                      // link
        case Color::Gray:               // back edge:
            prev->link(cur);            // only link full CFG
            return;
        case Color::Black:              // cross or forward edge: 
            break;                      // link
    }

    if (prev) {
        prev->link(cur);
        prev->reduced_link(cur);
    }
}

//------------------------------------------------------------------------------

CFG::CFG(const Scope& scope) 
    : scope_(scope)
    , nodes_(scope.size())
{
    for (size_t i = 0, e = size(); i != e; ++i)
        nodes_[i] = new CFGNode(scope[i]);

    CFA cfa(*this);
}

size_t CFG::sid(Lambda* lambda) const { 
    if (auto info = lambda->find_scope(&scope()))
        return info->sid;
    return size_t(-1);
}

void CFG::cfa() {
}

const F_CFG* CFG::f_cfg() const { return lazy_init(this, f_cfg_); }
const B_CFG* CFG::b_cfg() const { return lazy_init(this, b_cfg_); }
const DomTree* CFG::domtree() const { return f_cfg()->domtree(); }
const PostDomTree* CFG::postdomtree() const { return b_cfg()->domtree(); }
const LoopTree* CFG::looptree() const { return looptree_ ? looptree_ : looptree_ = new LoopTree(*f_cfg()); }

//------------------------------------------------------------------------------

template<bool forward>
CFGView<forward>::CFGView(const CFG& cfg)
    : cfg_(cfg)
    , rpo_ids_(cfg.size())
    , rpo_(cfg.nodes()) // copy over - sort later
{
    std::fill(rpo_ids_.begin(), rpo_ids_.end(), -1);    // mark as not visited
    auto num = number(entry(), 0);                      // number in post-order
    
    for (size_t i = 0, e = size(); i != e; ++i) {       // convert to reverse post-order
        auto& rpo_id = rpo_ids_[i];
        if (rpo_id != size_t(-1))
            rpo_id = num-1 - rpo_id;
    }

    // sort in reverse post-order
    std::sort(rpo_.begin(), rpo_.end(), [&] (const CFGNode* n1, const CFGNode* n2) { return rpo_id(n1) < rpo_id(n2); });
    rpo_.shrink(num);                                   // remove unreachable stuff
}

template<bool forward>
size_t CFGView<forward>::number(const CFGNode* n, size_t i) {
    auto& n_rpo_id = _rpo_id(n);
    n_rpo_id = -2; // mark as visited

    for (auto succ : succs(n)) {
        if (rpo_id(succ) == size_t(-1)) // if not visited
            i = number(succ, i);
    }

    return (n_rpo_id = i) + 1;
}

template<bool forward>
const DomTreeBase<forward>* CFGView<forward>::domtree() const { return lazy_init(this, domtree_); }

//------------------------------------------------------------------------------

}
