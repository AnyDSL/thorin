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
        , dfs_id(0)
    {
        stack.reserve(looptree.cfg().size());
        build();
        propagate_bounds(looptree.root_);
        analyse_loops(looptree.root_);
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
    static std::pair<size_t, size_t> propagate_bounds(LoopNode* header);
    void analyse_loops(LoopHeader* header);
    const CFG<true>& cfg() const { return looptree.cfg(); }
    Number& number(const CFNode* cfg_node) { return numbers[cfg_node]; }
    size_t& lowlink(const CFNode* cfg_node) { return number(cfg_node).low; }
    size_t& dfs(const CFNode* cfg_node) { return number(cfg_node).dfs; }
    bool on_stack(const CFNode* cfg_node) { assert(set.contains(cfg_node)); return (states[cfg_node] & OnStack) != 0; }
    bool in_scc(const CFNode* cfg_node) { return states[cfg_node] & InSCC; }
    bool is_header(const CFNode* cfg_node) { return states[cfg_node] & IsHeader; }

    bool is_leaf(const CFNode* cfg_node, size_t num) {
        if (num == 1) {
            for (auto succ : cfg().succs(cfg_node)) {
                if (!is_header(succ) && cfg_node == succ)
                    return false;
            }
            return true;
        }
        return false;
    }

    void push(const CFNode* cfg_node) {
        assert(set.contains(cfg_node) && (states[cfg_node] & OnStack) == 0);
        stack.push_back(cfg_node);
        states[cfg_node] |= OnStack;
    }

    int visit(const CFNode* cfg_node, int counter) {
        visit_first(set, cfg_node);
        numbers[cfg_node] = Number(counter++);
        push(cfg_node);
        return counter;
    }

    void recurse(LoopHeader* parent, ArrayRef<const CFNode*> headers, int depth);
    int walk_scc(const CFNode* cur, LoopHeader* parent, int depth, int scc_counter);

    LoopTree& looptree;
    CFNodeMap<Number> numbers;
    CFNodeMap<uint8_t> states;
    CFNodeSet set;
    size_t dfs_id;
    std::vector<const CFNode*> stack;
};

void LoopTreeBuilder::build() {
    for (auto cfg_node : cfg()) // clear all flags
        states[cfg_node] = 0;

    recurse(looptree.root_ = new LoopHeader(nullptr, 0, std::vector<const CFNode*>(0)), {cfg().entry()}, 1);
}

void LoopTreeBuilder::recurse(LoopHeader* parent, ArrayRef<const CFNode*> headers, int depth) {
    size_t cur_new_child = 0;
    for (auto header : headers) {
        set.clear();
        walk_scc(header, parent, depth, 0);

        // now mark all newly found headers globally as header
        for (size_t e = parent->num_children(); cur_new_child != e; ++cur_new_child) {
            for (auto header : parent->child(cur_new_child)->cfg_nodes())
                states[header] |= IsHeader;
        }
    }

    for (auto node : parent->children()) {
        if (auto new_parent = node->isa<LoopHeader>())
            recurse(new_parent, new_parent->cfg_nodes(), depth + 1);
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

        // mark all cfg_nodes in current SCC (all cfg_nodes from back to cur on the stack) as 'InSCC'
        size_t num = 0, e = stack.size(), b = e - 1;
        do {
            states[stack[b]] |= InSCC;
            ++num;
        } while (stack[b--] != cur);

        // for all cfg_nodes in current SCC
        for (size_t i = ++b; i != e; ++i) {
            const CFNode* cfg_node = stack[i];

            if (cfg().entry() == cfg_node)
                headers.push_back(cfg_node); // entries are axiomatically headers
            else {
                for (auto pred : cfg().preds(cfg_node)) {
                    // all backedges are also inducing headers
                    // but do not yet mark them globally as header -- we are still running through the SCC
                    if (!in_scc(pred)) {
                        headers.push_back(cfg_node);
                        break;
                    }
                }
            }
        }

        if (is_leaf(cur, num)) {
            assert(headers.size() == 1);
            LoopLeaf* leaf = new LoopLeaf(dfs_id++, parent, depth, headers);
            looptree.map_[headers.front()] = looptree.dfs_leaves_[leaf->dfs_id()] = leaf;
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

std::pair<size_t, size_t> LoopTreeBuilder::propagate_bounds(LoopNode* n) {
    if (auto header = n->isa<LoopHeader>()) {
        size_t begin = -1, end = 0;
        for (auto child : header->children()) {
            auto p = propagate_bounds(child);
            begin = p.first  < begin ? p.first  : begin;
            end   = p.second > end   ? p.second : end;
        }

        header->dfs_begin_ = begin;
        header->dfs_end_   = end;
        return std::make_pair(begin, end);
    } else {
        LoopLeaf* leaf = n->as<LoopLeaf>();
        return std::make_pair(leaf->dfs_id(), leaf->dfs_id()+1);
    }
}

void LoopTreeBuilder::analyse_loops(LoopHeader* header) {
    header->headers_.insert(header->cfg_nodes().begin(), header->cfg_nodes().end());

    for (auto cfg_node : header->cfg_nodes()) {
        for (auto pred : cfg().preds(cfg_node)) {
            if (looptree.contains(header, pred)) {
                header->back_edges_.emplace_back(pred, cfg_node, 0);
                header->latches_.insert(pred);
            } else {
                header->entry_edges_.emplace_back(pred, cfg_node, 1);
                header->preheaders_.insert(pred);
            }
        }
    }

    for (auto cfg_node : looptree.loop_cfg_nodes(header)) {
        for (auto succ : cfg().succs(cfg_node)) {
            if (!looptree.contains(header, succ)) {
                header->exit_edges_.emplace_back(cfg_node, succ, looptree.cfg_node2header(succ)->depth() - header->depth());
                header->exitings_.insert(cfg_node);
                header->exits_.insert(succ);
            }
        }
    }

    for (auto child : header->children())
        if (auto header = child->isa<LoopHeader>())
            analyse_loops(header);
}

//------------------------------------------------------------------------------

LoopNode::LoopNode(LoopHeader* parent, int depth, const std::vector<const CFNode*>& cfg_nodes)
    : parent_(parent)
    , depth_(depth)
    , cfg_nodes_(cfg_nodes)
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
    indent() << '<' << cfg_node()->lambda()->unique_name() << '>' << std::endl;
    indent() << "+ dfs: " << dfs_id() << std::endl;
}

void LoopHeader::Edge::dump() {
    std::cout << src_->lambda()->unique_name() << " ->(" << levels_ << ") " << dst_->lambda()->unique_name() << "   ";
}

#define DUMP_SET(set) \
    indent() << "+ " #set ": "; \
    for (auto cfg_node : set()) \
        std::cout << cfg_node->lambda()->unique_name() << " "; \
    std::cout << std::endl;

#define DUMP_EDGES(edges) \
    indent() << "+ " #edges ": "; \
    for (auto edge : edges()) \
        edge.dump(); \
    std::cout << std::endl;

void LoopHeader::dump() const {
    indent() << "( ";
    for (auto header : cfg_nodes())
        std::cout << header->lambda()->unique_name() << " ";
    std::cout << ") " << std::endl;
    indent() << "+ dfs: " << dfs_begin() << " .. " << dfs_end() << std::endl;

    DUMP_SET(preheaders)
    DUMP_SET(latches)
    DUMP_SET(exitings)
    DUMP_EDGES(entry_edges)
    DUMP_EDGES(back_edges)
    DUMP_EDGES(exit_edges)

    for (auto child : children())
        child->dump();
}

//------------------------------------------------------------------------------

LoopTree::LoopTree(const CFG<true>& cfg)
    : cfg_(cfg)
    , dfs_leaves_(cfg.size())
{
    LoopTreeBuilder(*this);
}

bool LoopTree::contains(const LoopHeader* header, const CFNode* cfg_node) const {
    //if (!cfg().contains(cfg_node)) return false; // TODO
    size_t dfs = cfg_node2dfs(cfg_node);
    return header->dfs_begin() <= dfs && dfs < header->dfs_end();
}

const LoopHeader* LoopTree::cfg_node2header(const CFNode* cfg_node) const {
    auto leaf = cfg_node2leaf(cfg_node);
    if (leaf == nullptr)
        return root();
    auto header = leaf->parent();
    for (; !header->is_root(); header = header->parent()) {
        if (contains(header, cfg_node))
            break;
    }
    return header;
}

Array<const CFNode*> LoopTree::loop_cfg_nodes(const LoopHeader* header) {
    auto leaves = loop(header);
    Array<const CFNode*> result(leaves.size());
    for (size_t i = 0, e = leaves.size(); i != e; ++i)
        result[i] = leaves[i]->cfg_node();
    return result;
}

Array<const CFNode*> LoopTree::loop_cfg_nodes_in_rpo(const LoopHeader* header) {
    auto result = loop_cfg_nodes(header);
    std::stable_sort(result.begin(), result.end(), [&](const CFNode* l1, const CFNode* l2) {
        return cfg_.rpo_id(l1) < cfg_.rpo_id(l2);
    });
    return result;
}

//------------------------------------------------------------------------------

}
