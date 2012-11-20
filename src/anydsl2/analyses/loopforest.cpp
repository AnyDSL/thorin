#include "anydsl2/analyses/loopforest.h"

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"

#include <algorithm>
#include <limits>
#include <iostream>
#include <stack>

namespace anydsl2 {

enum {
    OnStack, InSCC, IsHeader
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

        size_t dfs;
        size_t low;
    };

    Number& number(Lambda* lambda) { return numbers[lambda->sid()]; }
    size_t& lowlink(Lambda* lambda) { return number(lambda).low; }
    size_t& dfs(Lambda* lambda) { return number(lambda).dfs; }

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

    bool on_stack(Lambda* lambda) {
        assert(is_visited(lambda));
        return lambda->flags[OnStack];
    }

    bool is_header(Lambda* lambda) {
        if (lambda->cur_pass() >= first_pass)
            return lambda->flags[IsHeader];
        return false;
    }

    void visit(Lambda* lambda) {
        assert(!lambda->is_visited(pass));
        lambda->visit(pass);
        lambda->flags[OnStack] = false;
        lambda->flags[InSCC]   = false;
        if (pass == first_pass)
            lambda->flags[IsHeader] = false;
        numbers[lambda->sid()] = Number(counter++);
        push(lambda);
    }
    bool is_visited(Lambda* lambda) { return lambda->is_visited(pass); }

    void recurse(LoopForestNode* node);
    void walk_scc(Lambda* cur, LoopForestNode* node);

    const Scope& scope;
    Array<Number> numbers;
    size_t pass;
    size_t first_pass;
    size_t counter;
    std::vector<Lambda*> stack;
};

LoopForestNode* LFBuilder::build() {
    LoopForestNode* root = new LoopForestNode(0, -1);
    root->headers_.push_back(scope.entry());
    recurse(root);
    root->headers_.clear();
    return root;
}

void LFBuilder::recurse(LoopForestNode* parent) {
    pass = world().new_pass();
    if (first_pass == size_t(-1))
        first_pass = pass;

    counter = 0;
    walk_scc(parent->headers().front() /* pick one */, parent);

    for_all (node, parent->children()) {
        for_all (header, node->headers())
            header->flags[IsHeader] = true;
    }

    for_all (node, parent->children()) {
        if (node->depth() < -1)
            node->depth_ -= std::numeric_limits<int>::min();
        else
            recurse(node);
    }
}

void LFBuilder::walk_scc(Lambda* cur, LoopForestNode* parent) {
    visit(cur);

    for_all (succ, scope.succs(cur)) {
        if (is_header(succ))
            continue; // this is a backedge
        if (!is_visited(succ)) {
            walk_scc(succ, parent);
            lowlink(cur) = std::min(lowlink(cur), lowlink(succ));
        } else if (on_stack(succ))
            lowlink(cur) = std::min(lowlink(cur), dfs(succ));
    }

    // root of SCC
    if (lowlink(cur) == dfs(cur)) {
        LoopForestNode* node = new LoopForestNode(parent, (int) (pass - first_pass));
        std::vector<Lambda*>& headers = node->headers_;

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
            node->depth_ = std::numeric_limits<int>::min() + node->depth_;
        }
self_loop:
        for (size_t i = ++b; i != e; ++i) {
            Lambda* lambda = stack[i];
            if (lambda == scope.entry())
                headers.push_back(lambda);
            else {
                for_all (pred, scope.preds(lambda)) {
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

        // pop whole scc
        stack.resize(b);
        assert(num != 1 || (node->headers().size() == 1 && node->headers().front() == cur));
    }
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
        o << header->debug << " ";
    o << std::endl;
    for_all (child, node->children())
        o << child;
    return o;
}

//------------------------------------------------------------------------------

} // namespace anydsl2
