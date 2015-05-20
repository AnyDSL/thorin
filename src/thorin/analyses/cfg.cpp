#include "thorin/analyses/cfg.h"

#include <iostream>
#include <fstream>

#include "thorin/primop.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"
#include "thorin/util/queue.h"

#include "thorin/be/ycomp.h"

namespace thorin {

//------------------------------------------------------------------------------

uint64_t CFNodeHash::operator() (const CFNode* n) const { 
    if (auto in = n->isa<InNode>())
        return hash_value(in->lambda()->gid());
    auto out = n->as<OutNode>();
    return hash_combine(hash_value(out->def()->gid()), out->context()->lambda()->gid());
}

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

//------------------------------------------------------------------------------

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
        in_node(scope().entry());
        in_node(scope().exit());
        run_cfa();
        build_cfg();
    }

    void run_cfa();
    void build_cfg();

    const CFA& cfa() const { return cfa_; }
    const Scope& scope() const { return cfa_.scope(); }
    Array<CFNodeSet> cf_nodes_per_op(Lambda* lambda);

    const InNode* in_node(Lambda* lambda) {
        assert(scope().outer_contains(lambda));
        if (auto in = find(cfa().in_nodes(), lambda))
            return in;
        ++cfa_.num_in_nodes_;
        auto in = cfa_.in_nodes_[lambda] = new InNode(lambda);
        lambda2param2nodes_[lambda].resize(lambda->num_params()); // make room for params
        return in;
    }

    const OutNode* out_node(const InNode* in, Def def) {
        if (auto out = find(in->out_nodes_, def))
            return out;
        ++cfa_.num_out_nodes_;
        return in->out_nodes_[def] = new OutNode(in, def);
    }

    CFNodeSet& param2nodes(const Param* param) {
        in_node(param->lambda()); // alloc InNode and make room in lambda2param2nodes_
        return lambda2param2nodes_[param->lambda()][param->index()];
    }

private:
    CFA& cfa_;
    Scope::Map<std::vector<CFNodeSet>> lambda2param2nodes_; ///< Maps param in scope to CFNodeSet.
};

Array<CFNodeSet> CFABuilder::cf_nodes_per_op(Lambda* lambda) {
    auto in = in_node(lambda);

    // create dummy empty set entry for lambdas without body
    if (lambda->empty()) 
        return Array<CFNodeSet>(1);

    size_t num = lambda->size();
    Array<CFNodeSet> result(num);

    for (size_t i = 0; i != num; ++i) {
        leaves(lambda->op(i), [&] (Def def) {
            if (auto op_lambda = def->isa_lambda()) {
                if (scope().inner_contains(op_lambda))
                    result[i].insert(in_node(op_lambda));
                else if (i == 0)
                    result[i].insert(out_node(in, op_lambda));
            } else {
                auto param = def->as<Param>();
                if (scope().inner_contains(param)) {
                    const auto& set = param2nodes(param);
                    if (i == 0) {
                        for (auto n : set) {
                            if (auto out = n->isa<OutNode>()) {
                                assert(out->context() != in);
                                auto new_out = out_node(in, out->def());
                                new_out->link(out);
                                result[0].insert(new_out);
                            } else
                                result[0].insert(n);
                        }
                    } else
                        result[i].insert(set.begin(), set.end());
                } else if (i == 0)
                    result[0].insert(out_node(in, param));
            }
        });
    }

    return result;
}

void CFABuilder::run_cfa() {
    for (bool todo = true; todo;) {
        todo = false;

        for (auto lambda : scope()) {
            if (find(cfa().in_nodes(), lambda) == nullptr)
                continue;

            size_t old = cfa().num_cf_nodes();
            auto info = cf_nodes_per_op(lambda);
            todo |= old != cfa().num_cf_nodes();
            size_t num = lambda->size();

            for (auto to : info[0]) {
                if (auto in = to->isa<InNode>()) {
                    for (size_t i = 1; i != num; ++i) {
                        const auto& set = info[i];
                        todo |= lambda2param2nodes_[in->lambda()][i-1].insert(set.begin(), set.end());
                    }
                } else {
                    auto out = to->as<OutNode>();
                    assert(in_node(lambda) == out->context() && "OutNode's context does not match");
                    for (size_t i = 1; i != num; ++i) {
                        for (auto n : info[i]) {
                            if (auto info_in = n->isa<InNode>()) {
                                auto in_lambda = info_in->lambda();
                                for (size_t p = 0; p != in_lambda->num_params(); ++p) {
                                    if (in_lambda->param(p)->order() >= 1)
                                        todo |= lambda2param2nodes_[in_lambda][p].insert(out).second;
                                }
                            }
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

            if (out->def()->isa<Param>())
                out->link(cfa().exit());

            for (const auto& arg : info.skip_front()) {
                for (auto n_arg : arg)
                    out->link(n_arg);
            }
        }
    }

    // TODO link CFNodes not reachable from exit
    // HACK
    if (scope().entry()->empty())
        cfa().entry()->link(cfa().exit());

#ifndef NDEBUG
    bool error = false;
    for (auto in : cfa().in_nodes()) {
        if (in != cfa().entry() && in->preds_.size() == 0) {
            std::cout << "missing predecessors: " << in->lambda()->unique_name() << std::endl;
            error = true;
        }
    }
    if (error) {
        std::ofstream out("out.vcg");
        emit_ycomp(scope(), false, out);
        out.close();
        abort();
    }
#endif
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
void CFG<forward>::dump() const {
    for (auto n : rpo()) {
        for (auto succ : n->succs())
            std::cout << n->def()->unique_name() << " -> " << succ->def()->unique_name() << std::endl;
    }
}

template<bool forward> const DomTreeBase<forward>& CFG<forward>::domtree() const { return lazy_init(this, domtree_); }
template<bool forward> const LoopTree<forward>& CFG<forward>::looptree() const { return lazy_init(this, looptree_); }

template class CFG<true>;
template class CFG<false>;

//------------------------------------------------------------------------------

}
