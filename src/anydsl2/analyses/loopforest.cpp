#include "anydsl2/analyses/loopforest.h"

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"

#include <algorithm>
#include <iostream>

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
    LoopForestNode* root = new LoopForestNode(0);
    root->headers_.push_back(scope.entry());
    recurse(root);
    root->headers_.clear();
    return root;
}

void LFBuilder::recurse(LoopForestNode* parent) {
    std::cout << "recurse node: " << parent->headers().front()->debug << std::endl;
    pass = world().new_pass();
    if (first_pass == size_t(-1))
        first_pass = pass;

    counter = 0;
    walk_scc(parent->headers().front() /* pick one */, parent);

    for_all (node, parent->children()) {
        for_all (header, node->headers()) {
            std::cout << "mark as header: " << header->debug << std::endl;
            header->flags[IsHeader] = true;
        }
    }

    for_all (node, parent->children()) {
        if (node->depth() != -1) {
            recurse(node);
        }
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
        LoopForestNode* node = new LoopForestNode(parent);
        std::vector<Lambda*>& headers = node->headers_;

        std::cout << "root: " << cur->debug << std::endl;

        size_t num = 0;
        size_t e = stack.size();
        size_t b = e - 1;
        do {
            stack[b]->flags[InSCC] = true;
            ++num;
            std::cout << '\t' << stack[b]->debug << std::endl;
        } while (stack[b--] != cur);

        if (num == 1)
            node->depth_ = -1;

        for (size_t i = ++b; i != e; ++i) {
            Lambda* lambda = stack[i];
            if (lambda == scope.entry())
                goto set_header;
            else {
                for_all (pred, scope.preds(lambda)) {
                    if (!pred->flags[InSCC])
                        goto set_header;
                }
                continue;
            }
set_header:
            std::cout << "header: " << lambda->debug << std::endl;
            headers.push_back(lambda);
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

} // namespace anydsl2
