#include "thorin/analyses/looptree.h"

#include "thorin/lambda.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

#include <algorithm>
#include <limits>
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
        , dfs_index(0)
    {
        stack.reserve(looptree.scope().size());
        build();
        propagate_bounds(looptree.root_);
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
        set.visit_first(lambda);
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
    size_t dfs_index;
    std::vector<Lambda*> stack;
};

void LoopTreeBuilder::build() {
    for (auto lambda : scope()) // clear all flags
        states[lambda] = 0;

    recurse(looptree.root_ = new LoopHeader(0, -1, std::vector<Lambda*>(0)), {scope().entry()}, 0);
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
            LoopLeaf* leaf = new LoopLeaf(dfs_index++, parent, depth, headers);
            looptree.map_[headers.front()] = looptree.dfs_leaves_[leaf->dfs_index()] = leaf;
        } else
            new LoopHeader(parent, depth, headers);

        // for all lambdas in current SCC
        for (auto header : headers) {
            for (auto pred : scope().preds(header)) {
                if (in_scc(pred))
                    parent->backedges_.emplace_back(pred, header);
                else
                    parent->entries_.emplace_back(pred, header);
            }
        }

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
        return std::make_pair(leaf->dfs_index(), leaf->dfs_index()+1);
    }
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

Array<Lambda*> LoopTree::loop_lambdas(const LoopHeader* header) {
    auto leaves = loop(header);
    Array<Lambda*> result(leaves.size());
    for (size_t i = 0, e = leaves.size(); i != e; ++i)
        result[i] = leaves[i]->lambda();
    return result;
}

Array<Lambda*> LoopTree::loop_lambdas_in_rpo(const LoopHeader* header) {
    auto result = loop_lambdas(header);
    std::sort(result.begin(), result.end(), [&](Lambda* l1, Lambda* l2) {
        return scope_.sid(l1) < scope_.sid(l2);
    });
    return result;
}

void LoopTree::dump() const {
    std::cout << root_;
}

//------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& o, const LoopNode* node) {
    for (int i = 0; i < node->depth(); ++i)
        o << '\t';
    for (auto header : node->lambdas())
        o << header->unique_name() << " ";
    if (auto header = node->isa<LoopHeader>()) {
        o << ": " << header->dfs_begin() << "/" << header->dfs_end() << std::endl;
        for (auto child : header->children())
            o << child;
    } else
        o<< ": " << node->as<LoopLeaf>()->dfs_index() << std::endl;

    return o;
}

//------------------------------------------------------------------------------

} // namespace thorin
