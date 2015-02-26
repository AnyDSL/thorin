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
    CFABuilder(CFA& cfa)
        : cfa_(cfa)
        , lambda2param2nodes_(cfa.scope(), std::vector<CFNodeSet>(0))
    {
        run_cfa();
        build_cfg();
    }

    void run_cfa();
    void build_cfg();

    const CFA& cfa() const { return cfa_; }
    const Scope& scope() const { return cfa_.scope(); }
    //bool contains(Lambda* lambda) { return scope().contains(lambda); };
    bool contains(Lambda* lambda) { return scope().entry() != lambda && scope().contains(lambda); };
    bool contains(const Param* param) { return scope().entry() != param->lambda() && contains(param->lambda()); }
    const CFNode* lookup(Lambda* lambda) const { return cfa().in_nodes_[lambda]; }
    const InCFNode* in_node(Lambda* lambda) {
        if (auto in = find(cfa().in_nodes(), lambda))
            return in;
        ++cfa_.num_cf_nodes_;
        auto in = cfa_.in_nodes_[lambda] = new InCFNode(lambda);
        lambda2param2nodes_[lambda].resize(lambda->num_params()); // make room for params
        return in;
    }
    const OutCFNode* out_node(const InCFNode* in, Def def) {
        if (auto out = find(in->out_nodes_, def))
            return out;
        ++cfa_.num_cf_nodes_;
        return in->out_nodes_[def] = new OutCFNode(in, def);
    }
    const InCFNode* exit() const { return cfa().exit(); }
    Array<CFNodeSet> cf_nodes_per_op(Lambda* lambda);

private:
    CFA& cfa_;
    Scope::Map<std::vector<CFNodeSet>> lambda2param2nodes_; ///< Maps param in scope to CFNodeSet.
};

Array<CFNodeSet> CFABuilder::cf_nodes_per_op(Lambda* lambda) {
    auto in = in_node(lambda);
    size_t num = lambda->size();

    Array<CFNodeSet> result(num);
    for (size_t i = 0; i != num; ++i) {
        leaves(lambda->op(i), [&] (Def def) {
            if (auto op_lambda = def->isa_lambda()) {
                if (contains(op_lambda))
                    result[i].insert(in_node(op_lambda));
                else if (i == 0)
                    result[i].insert(out_node(in, op_lambda));
            } else {
                auto param = def->as<Param>();
                if (contains(param)) {
                    const auto& set = lambda2param2nodes_[param->lambda()][param->index()];
                    result[i].insert(set.begin(), set.end());
                } else if (i == 0)
                    result[i].insert(out_node(in, param));
            }
        });
    }

    return result;
}

void CFABuilder::run_cfa() {
    std::queue<Lambda*> queue;
    queue.push(scope().entry());

    while (!queue.empty()) {
        auto lambda = pop(queue);
        size_t num = lambda->size();

        auto info = cf_nodes_per_op(lambda);
        for (auto to : info[0]) {
            if (auto in = to->isa<InCFNode>()) {
                bool todo = false;
                for (size_t i = 1; i != num; ++i) {
                    const auto& set = info[i];
                    todo |= lambda2param2nodes_[in->lambda()][i-1].insert(set.begin(), set.end());
                }
                if (todo)
                    queue.push(in->lambda());
            } else {
                auto out = to->as<OutCFNode>();
                for (size_t i = 1; i != num; ++i) {
                    for (auto n : info[i]) {
                        if (auto info_in = n->isa<InCFNode>()) {
                            auto in_lambda = info_in->lambda();
                            bool todo = false;
                            for (size_t p = 0; p != in_lambda->num_params(); ++p)
                                if (in_lambda->param(p)->order() >= 1)
                                    todo |= lambda2param2nodes_[in_lambda][p].insert(out).second;
                            if (todo)
                                queue.push(in_lambda);
                        }
                    }
                }
            }
        }
    }
}

void CFABuilder::build_cfg() {
    for (auto in : cfa().in_nodes()) {
        auto info = cf_nodes_per_op(in->lambda());

        for (auto to : info[0])
            in->link(to);

        for (auto pair : in->out_nodes()) {
            auto out = pair.second;

            for (const auto& arg : info.slice_from_begin(1)) {
                for (auto n : arg) {
                    out->link(n);
                    n->link(out);
                }
            }
        }
    }
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

const F_CFG* CFA::f_cfg() const { return lazy_init(this, f_cfg_); }
const B_CFG* CFA::b_cfg() const { return lazy_init(this, b_cfg_); }
const DomTree* CFA::domtree() const { return f_cfg()->domtree(); }
const PostDomTree* CFA::postdomtree() const { return b_cfg()->domtree(); }
const LoopTree* CFA::looptree() const { return looptree_ ? looptree_ : looptree_ = new LoopTree(*f_cfg()); }

//------------------------------------------------------------------------------

template<bool forward>
CFG<forward>::CFG(const CFA& cfa)
    : cfa_(cfa)
    , rpo_(*this)
{
    size_t result = post_order_visit(entry(), size());
    assert(result == 0);
}

template<bool forward>
size_t CFG<forward>::post_order_visit(const CFNode* n, size_t i) {
    auto& n_index = forward ? n->f_index_ : n->b_index_;
    assert(n_index == size_t(-1));
    n_index = size_t(-2);

    for (auto succ : succs(n)) {
        if (index(succ) == size_t(-1))
            i = post_order_visit(succ, i);
    }

    n_index = i-1;
    rpo_[n] = n;
    return n_index;
}

template<bool forward>
const DomTreeBase<forward>* CFG<forward>::domtree() const { return lazy_init(this, domtree_); }

template class CFG<true>;
template class CFG<false>;

//------------------------------------------------------------------------------

}
