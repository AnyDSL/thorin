#include "anydsl2/analyses/loopforest.h"

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
    InSCC,   // is in current walk_scc run?
    OnStack, // is in current SCC stack?
    IsHeader,// all headers are marked, so subsequent runs can ignore backedges when searching for SCCs
};

class LFBuilder {
public:

    LFBuilder(const Scope& scope) 
        : scope(scope)
        , numbers(scope.size())
        , first_pass(size_t(-1))
    {
        stack.reserve(scope.size());
    }

    LoopForestNode* build();
    World& world() const { return scope.world(); }

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

        size_t dfs;// depth-first-search number
        size_t low;// low link (see Tarjan's SCC algo)
    };

    Number& number(Lambda* lambda) { return numbers[lambda->sid()]; }
    size_t& lowlink(Lambda* lambda) { return number(lambda).low; }
    size_t& dfs(Lambda* lambda) { return number(lambda).dfs; }
    bool on_stack(Lambda* lambda) { assert(is_visited(lambda)); return lambda->flags[OnStack]; }
    bool is_header(Lambda* lambda) { return lambda->cur_pass() >= first_pass ? lambda->flags[IsHeader] : false; }
    bool is_visited(Lambda* lambda) { return lambda->is_visited(pass); }

    void new_pass() {
        pass = world().new_pass();
        if (first_pass == size_t(-1))
            first_pass = pass;
    }

    void push(Lambda* lambda) { 
        assert(is_visited(lambda) && !lambda->flags[OnStack]);
        stack.push_back(lambda);
        lambda->flags[OnStack] = true;
    }

    Lambda* pop() { 
        Lambda* lambda = stack.back();
        assert(is_visited(lambda) && lambda->flags[OnStack]);
        lambda->flags[OnStack] = false;
        stack.pop_back();
        return lambda;
    }

    int visit(Lambda* lambda, int counter) {
        if (lambda->cur_pass() < first_pass)
            lambda->flags[IsHeader] = false; // only set the very first time
        lambda->visit_first(pass);
        lambda->flags[OnStack] = false;
        lambda->flags[InSCC]   = false;
        numbers[lambda->sid()] = Number(counter++);
        push(lambda);
        return counter;
    }

    void recurse(LoopForestNode* node, int depth);
    int walk_scc(Lambda* cur, LoopForestNode* node, int depth, int counter);

    const Scope& scope;
    Array<Number> numbers;
    size_t pass;
    size_t first_pass;
    std::vector<Lambda*> stack;
};

LoopForestNode* LFBuilder::build() {
    LoopForestNode* root = new LoopForestNode(0, -1);

    //size_t cur_new_child = 0;
    //for_all (entry, scope.entries()) {
        //if (entry->cur_pass() >= first_pass)
            //continue;
        //new_pass();
        //walk_scc(entry, root, 0, 0);

        //// now mark all newly found headers globally as header
        //for (size_t e = root->num_children(); cur_new_child != e; ++cur_new_child) {
            //for_all (header, root->child(cur_new_child)->headers())
                //header->flags[IsHeader] = true;
        //}
    //}

    //for_all (node, root->children()) {
        //if (node->depth() < 0) // do not recurse on finished nodes (see below)
            //node->depth_ -= std::numeric_limits<int>::min();
        //else
            //recurse(node, 1);
    //}

    root->headers_.insert(root->headers_.begin(), scope.entries().begin(), scope.entries().end());
    recurse(root, 0);
    root->headers_.clear();
    return root;
}

void LFBuilder::recurse(LoopForestNode* parent, int depth) {
    size_t cur_new_child = 0;
    for_all (header, parent->headers()) {
        new_pass();
        walk_scc(header, parent, depth, 0);

        // now mark all newly found headers globally as header
        for (size_t e = parent->num_children(); cur_new_child != e; ++cur_new_child) {
            for_all (header, parent->child(cur_new_child)->headers())
                header->flags[IsHeader] = true;
        }
    }

    for_all (node, parent->children()) {
        if (node->depth() < 0) // do not recurse on finished nodes (see below)
            node->depth_ -= std::numeric_limits<int>::min();
        else
            recurse(node, depth + 1);
    }
}

int LFBuilder::walk_scc(Lambda* cur, LoopForestNode* parent, int depth, int counter) {
    counter = visit(cur, counter);

    for_all (succ, scope.succs(cur)) {
        if (is_header(succ))
            continue; // this is a backedge
        if (!is_visited(succ)) {
            counter = walk_scc(succ, parent, depth, counter);
            lowlink(cur) = std::min(lowlink(cur), lowlink(succ));
        } else if (on_stack(succ))
            lowlink(cur) = std::min(lowlink(cur), dfs(succ));
    }

    // root of SCC
    if (lowlink(cur) == dfs(cur)) {
        LoopForestNode* node = new LoopForestNode(parent, depth);
        std::vector<Lambda*>& headers = node->headers_;

        // mark all lambdas in current SCC (all lambdas from back to cur on the stack) as 'InSCC'
        size_t num = 0, e = stack.size(), b = e - 1;
        do {
            stack[b]->flags[InSCC] = true;
            ++num;
        } while (stack[b--] != cur);

        if (num == 1) {
            for_all (succ, scope.succs(cur)) {
                if (!is_header(succ) && cur == succ)
                    goto self_loop;
            }

            // mark node as done by setting its depth temporarily to (min + depth) < 0
            node->depth_ += std::numeric_limits<int>::min();
        }

self_loop:
        // for all lambdas in current SCC
        for (size_t i = ++b; i != e; ++i) {
            Lambda* lambda = stack[i];

            if (scope.is_entry(lambda)) 
                headers.push_back(lambda); // entries are axiomatically headers
            else {
                for_all (pred, scope.preds(lambda)) {
                    // all backedges are also inducing headers
                    // but do not yet mark them globally as header -- we are still running through the SCC
                    if (!pred->flags[InSCC]) {
                        headers.push_back(lambda);
                        break;
                    }
                }
            }
        }

        // reset InSCC flag
        for (size_t i = b; i != e; ++i)
            stack[i]->flags[InSCC] = false;

        // pop whole SCC
        stack.resize(b);
        assert(num != 1 || (node->headers().size() == 1 && node->headers().front() == cur));
    }

    return counter;
}

LoopForestNode* create_loop_forest(const Scope& scope) {
    LFBuilder builder(scope);
    return builder.build();
}

//------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& o, const LoopForestNode* node) {
    for (int i = 0; i < node->depth(); ++i)
        o << '\t';
    for_all (header, node->headers())
        o << header->unique_name() << " ";
    o << std::endl;
    for_all (child, node->children())
        o << child;
    return o;
}

//------------------------------------------------------------------------------

void LoopInfo::build_infos() {
    visit(scope_.loopforest());
}

void LoopInfo::visit(const LoopForestNode* n) {
    if (n->num_children() == 0) {
        assert(n->headers().size() == 1);
        Lambda* lambda = n->headers().front();
        depth_[lambda->sid()] = n->depth();
    } else {
        for_all (child, n->children())
            visit(child);
    }
}

//------------------------------------------------------------------------------

} // namespace anydsl2
