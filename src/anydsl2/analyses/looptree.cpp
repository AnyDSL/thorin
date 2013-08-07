#include "anydsl2/analyses/looptree.h"

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"

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

namespace anydsl2 {

enum {
    InSCC    = 1, // is in current walk_scc run?
    OnStack  = 2, // is in current SCC stack?
    IsHeader = 4, // all headers are marked, so subsequent runs can ignore backedges when searching for SCCs
};

class LoopTreeBuilder {
public:

    LoopTreeBuilder(LoopTree& looptree) 
        : looptree(looptree)
        , numbers(size())
        , first_pass(size_t(-1))
        , dfs_index(0)
    {
        stack.reserve(size());
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
    size_t size() const { return looptree.size(); }
    Number& number(Lambda* lambda) { return numbers[lambda->sid()]; }
    size_t& lowlink(Lambda* lambda) { return number(lambda).low; }
    size_t& dfs(Lambda* lambda) { return number(lambda).dfs; }
    bool on_stack(Lambda* lambda) { assert(is_visited(lambda)); return (lambda->counter & OnStack) != 0; }
    bool in_scc(Lambda* lambda) { return lambda->cur_pass() >= first_pass ? (lambda->counter & InSCC) != 0 : false; }
    bool is_header(Lambda* lambda) const { return lambda->cur_pass() >= first_pass ? (lambda->counter & IsHeader) != 0 : false; }
    bool is_visited(Lambda* lambda) { return lambda->is_visited(pass); }
    bool is_leaf(Lambda* lambda, size_t num) {
        if (num == 1) {
            for (auto succ : looptree.succs(lambda)) {
                if (!is_header(succ) && lambda == succ)
                    return false;
            }
            return true;
        }
        return false;
    }

    void new_pass() {
        pass = scope().world().new_pass();
        first_pass = first_pass == size_t(-1) ? pass : first_pass;
    }

    void push(Lambda* lambda) { 
        assert(is_visited(lambda) && (lambda->counter & OnStack) == 0);
        stack.push_back(lambda);
        lambda->counter |= OnStack;
    }

    int visit(Lambda* lambda, int counter) {
        lambda->visit_first(pass);
        numbers[lambda->sid()] = Number(counter++);
        push(lambda);
        return counter;
    }

    template<bool start>
    void recurse(LoopHeader* parent, ArrayRef<Lambda*> headers, int depth);
    int walk_scc(Lambda* cur, LoopHeader* parent, int depth, int scc_counter);

    LoopTree& looptree;
    Array<Number> numbers;
    size_t pass;
    size_t first_pass;
    size_t dfs_index;
    std::vector<Lambda*> stack;
};

void LoopTreeBuilder::build() {
    // clear all flags
    for (auto lambda : scope().rpo())
        lambda->counter = 0;

    recurse<true>(looptree.root_ = new LoopHeader(0, -1, std::vector<Lambda*>(0)), scope().entries(), 0);
}

template<bool start>
void LoopTreeBuilder::recurse(LoopHeader* parent, ArrayRef<Lambda*> headers, int depth) {
    size_t cur_new_child = 0;
    for (auto header : headers) {
        new_pass();
        if (start && header->cur_pass() >= first_pass) 
            continue; // in the base case we only want to find SCC on all until now unseen lambdas
        walk_scc(header, parent, depth, 0);

        // now mark all newly found headers globally as header
        for (size_t e = parent->num_children(); cur_new_child != e; ++cur_new_child) {
            for (auto header : parent->child(cur_new_child)->headers())
                header->counter |= IsHeader;
        }
    }

    for (auto node : parent->children()) {
        if (LoopHeader* new_parent = node->isa<LoopHeader>())
            recurse<false>(new_parent, new_parent->headers(), depth + 1);
    }
}

int LoopTreeBuilder::walk_scc(Lambda* cur, LoopHeader* parent, int depth, int scc_counter) {
    scc_counter = visit(cur, scc_counter);

    for (auto succ : scope().succs(cur)) {
        if (is_header(succ))
            continue; // this is a backedge
        if (!is_visited(succ)) {
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
            stack[b]->counter |= InSCC;
            ++num;
        } while (stack[b--] != cur);

        // for all lambdas in current SCC
        for (size_t i = ++b; i != e; ++i) {
            Lambda* lambda = stack[i];

            if (scope().is_entry(lambda)) 
                headers.push_back(lambda); // entries are axiomatically headers
            else {
                for (auto pred : looptree.preds(lambda)) {
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
            LoopLeaf* leaf = new LoopLeaf(dfs_index++, parent, depth, headers);
            looptree.nodes_[headers.front()->sid()] = looptree.dfs_leaves_[leaf->dfs_index()] = leaf;
        } else
            new LoopHeader(parent, depth, headers);

        // for all lambdas in current SCC
        for (auto header : headers) {
            for (auto pred : looptree.preds(header)) {
                if (in_scc(pred))
                    parent->backedges_.push_back(Edge(pred, header));
                else
                    parent->entries_.push_back(Edge(pred, header));
            }
        }

        // reset InSCC and OnStack flags
        for (size_t i = b; i != e; ++i)
            stack[i]->counter &= ~(OnStack | InSCC);

        // pop whole SCC
        stack.resize(b);
    }

    return scc_counter;
}

std::pair<size_t, size_t> LoopTreeBuilder::propagate_bounds(LoopNode* n) {
    if (LoopHeader* header = n->isa<LoopHeader>()) {
        size_t begin = -1, end = 0;
        for (auto child : header->children()) {
            std::pair<size_t, size_t> p = propagate_bounds(child);
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

LoopNode::LoopNode(LoopHeader* parent, int depth, const std::vector<Lambda*>& headers)
    : parent_(parent)
    , depth_(depth)
    , headers_(headers)
{
    if (parent_)
        parent_->children_.push_back(this);
}

//------------------------------------------------------------------------------

LoopTree::LoopTree(const Scope& scope)
    : Super(scope)
    , dfs_leaves_(size())
{
    LoopTreeBuilder(*this);
}

Array<Lambda*> LoopTree::loop_lambdas(const LoopHeader* header) {
    ArrayRef<const LoopLeaf*> leaves = loop(header);
    Array<Lambda*> result(leaves.size());
    for_all2 (&lambda, result, leaf, leaves)
        lambda = leaf->lambda();
    return result;
}

Array<Lambda*> LoopTree::loop_lambdas_in_rpo(const LoopHeader* header) {
    Array<Lambda*> result = loop_lambdas(header);
    std::sort(result.begin(), result.end(), [] (const Lambda* l1, const Lambda* l2) { return l1->sid() < l2->sid(); });
    return result;
}

//------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& o, const LoopNode* node) {
    for (int i = 0; i < node->depth(); ++i)
        o << '\t';
    for (auto header : node->headers())
        o << header->unique_name() << " ";
    if (const LoopHeader* header = node->isa<LoopHeader>()) {
        o << ": " << header->dfs_begin() << "/" << header->dfs_end() << std::endl;
        for (auto child : header->children())
            o << child;
    } else
        o<< ": " << node->as<LoopLeaf>()->dfs_index() << std::endl;

    return o;
}

//------------------------------------------------------------------------------

} // namespace anydsl2
