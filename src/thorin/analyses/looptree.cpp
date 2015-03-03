#include "thorin/analyses/looptree.h"

#include <algorithm>
#include <iostream>
#include <stack>

#include "thorin/lambda.h"
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
    IsHeader = 4, // all headers are marked, so subsequent runs can ignore backedges when searching for SCCs
};

class LoopTreeBuilder {
public:
    LoopTreeBuilder(LoopTree& looptree)
        : looptree(looptree)
        , numbers(cfg())
        , states(cfg())
        , set(cfg())
        , dfs_id(0)
    {
        stack.reserve(looptree.cfg().size());
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
    const F_CFG& cfg() const { return looptree.cfg(); }
    Number& number(const CFNode* n) { return numbers[n]; }
    size_t& lowlink(const CFNode* n) { return number(n).low; }
    size_t& dfs(const CFNode* n) { return number(n).dfs; }
    bool on_stack(const CFNode* n) { assert(set.contains(n)); return (states[n] & OnStack) != 0; }
    bool in_scc(const CFNode* n) { return states[n] & InSCC; }
    bool is_header(const CFNode* n) { return states[n] & IsHeader; }

    bool is_leaf(const CFNode* n, size_t num) {
        if (num == 1) {
            for (auto succ : cfg().succs(n)) {
                if (!is_header(succ) && n == succ)
                    return false;
            }
            return true;
        }
        return false;
    }

    void push(const CFNode* n) {
        assert(set.contains(n) && (states[n] & OnStack) == 0);
        stack.push_back(n);
        states[n] |= OnStack;
    }

    int visit(const CFNode* n, int counter) {
        visit_first(set, n);
        numbers[n] = Number(counter++);
        push(n);
        return counter;
    }

    void recurse(LoopHeader* parent, ArrayRef<const CFNode*> headers, int depth);
    int walk_scc(const CFNode* cur, LoopHeader* parent, int depth, int scc_counter);

    LoopTree& looptree;
    F_CFG::Map<Number> numbers;
    F_CFG::Map<uint8_t> states;
    F_CFG::Set set;
    size_t dfs_id;
    std::vector<const CFNode*> stack;
};

void LoopTreeBuilder::build() {
    for (auto n : cfg().rpo()) // clear all flags
        states[n] = 0;

    recurse(looptree.root_ = new LoopHeader(nullptr, 0, std::vector<const CFNode*>(0)), {cfg().entry()}, 1);
}

void LoopTreeBuilder::recurse(LoopHeader* parent, ArrayRef<const CFNode*> headers, int depth) {
    size_t cur_new_child = 0;
    for (auto header : headers) {
        set.clear();
        walk_scc(header, parent, depth, 0);

        // now mark all newly found headers globally as header
        for (size_t e = parent->num_children(); cur_new_child != e; ++cur_new_child) {
            for (auto header : parent->child(cur_new_child)->cf_nodes())
                states[header] |= IsHeader;
        }
    }

    for (auto node : parent->children()) {
        if (auto new_parent = node->isa<LoopHeader>())
            recurse(new_parent, new_parent->cf_nodes(), depth + 1);
    }
}

int LoopTreeBuilder::walk_scc(const CFNode* cur, LoopHeader* parent, int depth, int scc_counter) {
    scc_counter = visit(cur, scc_counter);

    for (auto succ : cfg().succs(cur)) {
        if (is_header(succ))
            continue; // this is a backedge
        if (!set.contains(succ)) {
            scc_counter = walk_scc(succ, parent, depth, scc_counter);
            lowlink(cur) = std::min(lowlink(cur), lowlink(succ));
        } else if (on_stack(succ))
            lowlink(cur) = std::min(lowlink(cur), lowlink(succ));
    }

    // root of SCC
    if (lowlink(cur) == dfs(cur)) {
        std::vector<const CFNode*> headers;

        // mark all cf_nodes in current SCC (all cf_nodes from back to cur on the stack) as 'InSCC'
        size_t num = 0, e = stack.size(), b = e - 1;
        do {
            states[stack[b]] |= InSCC;
            ++num;
        } while (stack[b--] != cur);

        // for all cf_nodes in current SCC
        for (size_t i = ++b; i != e; ++i) {
            const CFNode* n = stack[i];

            if (cfg().entry() == n)
                headers.push_back(n); // entries are axiomatically headers
            else {
                for (auto pred : cfg().preds(n)) {
                    // all backedges are also inducing headers
                    // but do not yet mark them globally as header -- we are still running through the SCC
                    if (!in_scc(pred)) {
                        headers.push_back(n);
                        break;
                    }
                }
            }
        }

        if (is_leaf(cur, num)) {
            assert(headers.size() == 1);
            looptree.cf2leaf_[headers.front()] = new LoopLeaf(dfs_id++, parent, depth, headers);
        } else
            new LoopHeader(parent, depth, headers);

        // reset InSCC and OnStack flags
        for (size_t i = b; i != e; ++i)
            states[stack[i]] &= ~(OnStack | InSCC);

        // pop whole SCC
        stack.resize(b);
    }

    return scc_counter;
}

//------------------------------------------------------------------------------

LoopNode::LoopNode(LoopHeader* parent, int depth, const std::vector<const CFNode*>& cf_nodes)
    : parent_(parent)
    , depth_(depth)
    , cf_nodes_(cf_nodes)
{
    if (parent_)
        parent_->children_.push_back(this);
}

std::ostream& LoopNode::indent() const {
    for (int i = 0; i < depth(); ++i)
        std::cout << '\t';
    return std::cout;
}

void LoopLeaf::dump() const {
    indent() << '<' << cf_node()->def()->unique_name() << '>' << std::endl;
    indent() << "+ dfs: " << dfs_id() << std::endl;
}

void LoopHeader::dump() const {
    indent() << "( ";
    for (auto header : cf_nodes())
        std::cout << header->def()->unique_name() << " ";
    std::cout << ") " << std::endl;

    for (auto child : children())
        child->dump();
}

//------------------------------------------------------------------------------

LoopTree::LoopTree(const F_CFG& cfg)
    : cfg_(cfg)
    , cf2leaf_(cfg)
{
    LoopTreeBuilder(*this);
}

//------------------------------------------------------------------------------

}
