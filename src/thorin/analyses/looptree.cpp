#include "thorin/analyses/looptree.h"

#include <algorithm>
#include <iostream>
#include <stack>

#include "thorin/lam.h"
#include "thorin/analyses/cfg.h"

/*
 * The implementation is based on Steensgard's algorithm to find loops in irreducible CFGs.
 *
 * In short, Steensgard's algorithm recursively applies Tarjan's SCC algorithm to find nested SCCs.
 * In the next recursion, backedges from the prior run are ignored.
 * Please, check out
 * http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
 * for more details on Tarjan's SCC algorithm
 */

namespace thorin {

enum {
    InSCC    = 1, // is in current walk_scc run?
    OnStack  = 2, // is in current SCC stack?
    IsHead   = 4, // all heads are marked, so subsequent runs can ignore backedges when searching for SCCs
};

template<bool forward>
class LoopTreeBuilder {
public:
    typedef typename LoopTree<forward>::Node Node;
    typedef typename LoopTree<forward>::Leaf Leaf;
    typedef typename LoopTree<forward>::Head Head;

    LoopTreeBuilder(LoopTree<forward>& looptree)
        : looptree_(looptree)
        , numbers_(cfg())
        , states_(cfg())
        , set_(cfg())
        , index_(0)
    {
        stack_.reserve(looptree.cfg().size());
        build();
    }

private:
    struct Number {
        Number()
            : dfs(-1)
            , low(-1)
        {}
        Number(size_t i)
            : dfs(i)
            , low(i)
        {}

        size_t dfs; // depth-first-search number
        size_t low; // low link (see Tarjan's SCC algo)
    };

    void build();
    const CFG<forward>& cfg() const { return looptree_.cfg(); }
    Number& number(const CFNode* n) { return numbers_[n]; }
    size_t& lowlink(const CFNode* n) { return number(n).low; }
    size_t& dfs(const CFNode* n) { return number(n).dfs; }
    bool on_stack(const CFNode* n) { assert(set_.contains(n)); return (states_[n] & OnStack) != 0; }
    bool in_scc(const CFNode* n) { return states_[n] & InSCC; }
    bool is_head(const CFNode* n) { return states_[n] & IsHead; }

    bool is_leaf(const CFNode* n, size_t num) {
        if (num == 1) {
            for (const auto& succ : cfg().succs(n)) {
                if (!is_head(succ) && n == succ)
                    return false;
            }
            return true;
        }
        return false;
    }

    void push(const CFNode* n) {
        assert(set_.contains(n) && (states_[n] & OnStack) == 0);
        stack_.emplace_back(n);
        states_[n] |= OnStack;
    }

    int visit(const CFNode* n, int counter) {
        visit_first(set_, n);
        numbers_[n] = Number(counter++);
        push(n);
        return counter;
    }

    void recurse(Head* parent, ArrayRef<const CFNode*> heads, int depth);
    int walk_scc(const CFNode* cur, Head* parent, int depth, int scc_counter);

private:
    LoopTree<forward>& looptree_;
    typename CFG<forward>::template Map<Number> numbers_;
    typename CFG<forward>::template Map<uint8_t> states_;
    typename CFG<forward>::Set set_;
    size_t index_;
    std::vector<const CFNode*> stack_;
};

template<bool forward>
void LoopTreeBuilder<forward>::build() {
    for (const auto& n : cfg().reverse_post_order()) // clear all flags
        states_[n] = 0;

    auto head = new Head(nullptr, 0, std::vector<const CFNode*>(0));
    looptree_.root_.reset(head);
    recurse(head, {cfg().entry()}, 1);
}

template<bool forward>
void LoopTreeBuilder<forward>::recurse(Head* parent, ArrayRef<const CFNode*> heads, int depth) {
    size_t cur_new_child = 0;
    for (const auto& head : heads) {
        set_.clear();
        walk_scc(head, parent, depth, 0);

        // now mark all newly found heads globally as head
        for (size_t e = parent->num_children(); cur_new_child != e; ++cur_new_child) {
            for (const auto& head : parent->child(cur_new_child)->cf_nodes())
                states_[head] |= IsHead;
        }
    }

    for (const auto& node : parent->children()) {
        if (auto new_parent = node->template isa<Head>())
            recurse(new_parent, new_parent->cf_nodes(), depth + 1);
    }
}

template<bool forward>
int LoopTreeBuilder<forward>::walk_scc(const CFNode* cur, Head* parent, int depth, int scc_counter) {
    scc_counter = visit(cur, scc_counter);

    for (const auto& succ : cfg().succs(cur)) {
        if (is_head(succ))
            continue; // this is a backedge
        if (!set_.contains(succ)) {
            scc_counter = walk_scc(succ, parent, depth, scc_counter);
            lowlink(cur) = std::min(lowlink(cur), lowlink(succ));
        } else if (on_stack(succ))
            lowlink(cur) = std::min(lowlink(cur), lowlink(succ));
    }

    // root of SCC
    if (lowlink(cur) == dfs(cur)) {
        std::vector<const CFNode*> heads;

        // mark all cf_nodes in current SCC (all cf_nodes from back to cur on the stack) as 'InSCC'
        size_t num = 0, e = stack_.size(), b = e - 1;
        do {
            states_[stack_[b]] |= InSCC;
            ++num;
        } while (stack_[b--] != cur);

        // for all cf_nodes in current SCC
        for (size_t i = ++b; i != e; ++i) {
            const CFNode* n = stack_[i];

            if (cfg().entry() == n)
                heads.emplace_back(n); // entries are axiomatically heads
            else {
                for (const auto& pred : cfg().preds(n)) {
                    // all backedges are also inducing heads
                    // but do not yet mark them globally as head -- we are still running through the SCC
                    if (!in_scc(pred)) {
                        heads.emplace_back(n);
                        break;
                    }
                }
            }
        }

        if (is_leaf(cur, num)) {
            assert(heads.size() == 1);
            looptree_.leaves_[heads.front()] = new Leaf(index_++, parent, depth, heads);
        } else
            new Head(parent, depth, heads);

        // reset InSCC and OnStack flags
        for (size_t i = b; i != e; ++i)
            states_[stack_[i]] &= ~(OnStack | InSCC);

        // pop whole SCC
        stack_.resize(b);
    }

    return scc_counter;
}

//------------------------------------------------------------------------------

template<bool forward>
LoopTree<forward>::Node::Node(Head* parent, int depth, const std::vector<const CFNode*>& cf_nodes)
    : parent_(parent)
    , cf_nodes_(cf_nodes)
    , depth_(depth)
{
    if (parent_)
        parent_->children_.emplace_back(this);
}

template<bool forward>
std::ostream& LoopTree<forward>::Leaf::stream(std::ostream& out) const {
    return streamf(out, "<{} | dfs: {}", cf_node(), index());
}

template<bool forward>
std::ostream& LoopTree<forward>::Head::stream(std::ostream& out) const {
    out << "[";
    for (const auto& head : this->cf_nodes()) // TODO use stream comma list - once it is there
        out << head << ", ";
    return out << "]";
}

template<bool forward>
void LoopTree<forward>::stream_ycomp(std::ostream& out) const {
    std::vector<const Node *> nodes;
    get_nodes(nodes, root());

    thorin::ycomp(out, YCompOrientation::LeftToRight, cfg().scope(), range(nodes),
        [] (const Node* n) {
            if (auto head = n->template isa<Head>())
                return range(head->children());
            return range(ArrayRef<std::unique_ptr<Node>>());
        }
    );
}

//------------------------------------------------------------------------------

template<bool forward>
LoopTree<forward>::LoopTree(const CFG<forward>& cfg)
    : YComp(cfg.scope(), forward ? "looptree" : "backwards_looptree")
    , cfg_(cfg)
    , leaves_(cfg)
{
    LoopTreeBuilder<forward>(*this);
}

template class LoopTree<true>;
template class LoopTree<false>;

//------------------------------------------------------------------------------

}
