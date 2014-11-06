#include "thorin/analyses/looptree.h"

#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

#include <algorithm>
#include <iostream>
#include <stack>

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
        stack.reserve(looptree.scope().size());
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
    const Scope& scope() const { return looptree.scope(); }
    Number& number(Lambda* lambda) { return numbers[lambda]; }
    size_t& lowlink(Lambda* lambda) { return number(lambda).low; }
    size_t& dfs(Lambda* lambda) { return number(lambda).dfs; }
    bool on_stack(Lambda* lambda) { assert(set.contains(lambda)); return (states[lambda] & OnStack) != 0; }
    bool in_scc(Lambda* lambda) { return states[lambda] & InSCC; }
    bool is_header(Lambda* lambda) { return states[lambda] & IsHeader; }

    bool is_leaf(Lambda* lambda, size_t num) {
        if (num == 1) {
            for (auto succ : scope().succs(lambda)) {
                if (!is_header(succ) && lambda == succ)
                    return false;
            }
            return true;
        }
        return false;
    }

    void push(Lambda* lambda) {
        assert(set.contains(lambda) && (states[lambda] & OnStack) == 0);
        stack.push_back(lambda);
        states[lambda] |= OnStack;
    }

    int visit(Lambda* lambda, int counter) {
        visit_first(set, lambda);
        numbers[lambda] = Number(counter++);
        push(lambda);
        return counter;
    }

    void recurse(LoopHeader* parent, ArrayRef<Lambda*> headers, int depth);
    int walk_scc(Lambda* cur, LoopHeader* parent, int depth, int scc_counter);

    LoopTree& looptree;
    LambdaMap<Number> numbers;
    LambdaMap<uint8_t> states;
    LambdaSet set;
    size_t dfs_id;
    std::vector<Lambda*> stack;
};

void LoopTreeBuilder::build() {
    for (auto lambda : scope()) // clear all flags
        states[lambda] = 0;

    recurse(looptree.root_ = new LoopHeader(nullptr, 0, std::vector<Lambda*>(0)), {scope().entry()}, 1);
}

void LoopTreeBuilder::recurse(LoopHeader* parent, ArrayRef<Lambda*> headers, int depth) {
    size_t cur_new_child = 0;
    for (auto header : headers) {
        set.clear();
        walk_scc(header, parent, depth, 0);

        // now mark all newly found headers globally as header
        for (size_t e = parent->num_children(); cur_new_child != e; ++cur_new_child) {
            for (auto header : parent->child(cur_new_child)->lambdas())
                states[header] |= IsHeader;
        }
    }

    for (auto node : parent->children()) {
        if (auto new_parent = node->isa<LoopHeader>())
            recurse(new_parent, new_parent->lambdas(), depth + 1);
    }
}

int LoopTreeBuilder::walk_scc(Lambda* cur, LoopHeader* parent, int depth, int scc_counter) {
    scc_counter = visit(cur, scc_counter);

    for (auto succ : scope().succs(cur)) {
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
        std::vector<Lambda*> headers;

        // mark all lambdas in current SCC (all lambdas from back to cur on the stack) as 'InSCC'
        size_t num = 0, e = stack.size(), b = e - 1;
        do {
            states[stack[b]] |= InSCC;
            ++num;
        } while (stack[b--] != cur);

        // for all lambdas in current SCC
        for (size_t i = ++b; i != e; ++i) {
            Lambda* lambda = stack[i];

            if (scope().entry() == lambda)
                headers.push_back(lambda); // entries are axiomatically headers
            else {
                for (auto pred : scope().preds(lambda)) {
                    // all backedges are also inducing headers
                    // but do not yet mark them globally as header -- we are still running through the SCC
                    if (!in_scc(pred)) {
                        headers.push_back(lambda);
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
    header->headers_.insert(header->lambdas().begin(), header->lambdas().end());

    for (auto lambda : header->lambdas()) {
        for (auto pred : lambda->preds()) {
            if (looptree.contains(header, pred)) {
                header->back_edges_.emplace_back(pred, lambda, 0);
                header->latches_.insert(pred);
            } else {
                header->entry_edges_.emplace_back(pred, lambda, 1);
                header->preheaders_.insert(pred);
            }
        }
    }

    for (auto lambda : looptree.loop_lambdas(header)) {
        for (auto succ : lambda->succs()) {
            if (!looptree.contains(header, succ)) {
                header->exit_edges_.emplace_back(lambda, succ, looptree.lambda2header(succ)->depth() - header->depth());
                header->exitings_.insert(lambda);
                header->exits_.insert(succ);
            }
        }
    }

    for (auto child : header->children())
        if (auto header = child->isa<LoopHeader>())
            analyse_loops(header);
}

//------------------------------------------------------------------------------

LoopNode::LoopNode(LoopHeader* parent, int depth, const std::vector<Lambda*>& lambdas)
    : parent_(parent)
    , depth_(depth)
    , lambdas_(lambdas)
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
    indent() << '<' << lambda()->unique_name() << '>' << std::endl;
    indent() << "+ dfs: " << dfs_id() << std::endl;
}

void LoopHeader::Edge::dump() {
    std::cout << src_->unique_name() << " ->(" << levels_ << ") " << dst_->unique_name() << "   ";
}

#define DUMP_SET(set) \
    indent() << "+ " #set ": "; \
    for (auto lambda : set()) \
        std::cout << lambda->unique_name() << " "; \
    std::cout << std::endl;

#define DUMP_EDGES(edges) \
    indent() << "+ " #edges ": "; \
    for (auto edge : edges()) \
        edge.dump(); \
    std::cout << std::endl;

void LoopHeader::dump() const {
    indent() << "( ";
    for (auto header : lambdas())
        std::cout << header->unique_name() << " ";
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

LoopTree::LoopTree(const Scope& scope)
    : scope_(scope)
    , dfs_leaves_(scope.size())
{
    LoopTreeBuilder(*this);
}

bool LoopTree::contains(const LoopHeader* header, Lambda* lambda) const {
    if (!scope().contains(lambda)) return false;
    size_t dfs = lambda2dfs(lambda);
    return header->dfs_begin() <= dfs && dfs < header->dfs_end();
}

const LoopHeader* LoopTree::lambda2header(Lambda* lambda) const {
    auto leaf = lambda2leaf(lambda);
    if (leaf == nullptr)
        return root();
    auto header = leaf->parent();
    for (; !header->is_root(); header = header->parent()) {
        if (contains(header, lambda))
            break;
    }
    return header;
}

Array<Lambda*> LoopTree::loop_lambdas(const LoopHeader* header) {
    auto leaves = loop(header);
    Array<Lambda*> result(leaves.size());
    for (size_t i = 0, e = leaves.size(); i != e; ++i)
        result[i] = leaves[i]->lambda();
    return result;
}

Array<Lambda*> LoopTree::loop_lambdas_in_rpo(const LoopHeader* header) {
    auto result = loop_lambdas(header);
    std::stable_sort(result.begin(), result.end(), [&](Lambda* l1, Lambda* l2) {
        return scope_.sid(l1) < scope_.sid(l2);
    });
    return result;
}

//------------------------------------------------------------------------------

}
