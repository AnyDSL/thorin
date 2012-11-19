#include "anydsl2/analyses/loopforest.h"

#include "anydsl2/lambda.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"

#include <algorithm>
#include <iostream>

namespace anydsl2 {

class LFBuilder {
public:


    LFBuilder(const Scope& scope) 
        : scope(scope)
        , numbers(scope.size())
        , pass(world().new_pass())
        , counter(0)
    {
        stack.reserve(scope.size());
        walk_scc(scope.entry());
    }

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
        assert(is_visited(lambda) && !lambda->flags[0]);
        stack.push_back(lambda);
        lambda->flags[0] = true;
    }

    Lambda* pop() { 
        Lambda* lambda = stack.back();
        assert(is_visited(lambda) && lambda->flags[0]);
        lambda->flags[0] = false;
        stack.pop_back();
        return lambda;
    }

    bool is_on_stack(Lambda* lambda) {
        assert(is_visited(lambda));
        return lambda->flags[0];
    }

    void visit(Lambda* lambda) {
        assert(!lambda->is_visited(pass));
        lambda->visit(pass);
        lambda->flags[0] = false;
        numbers[lambda->sid()] = Number(counter++);
        push(lambda);
    }
    bool is_visited(Lambda* lambda) { return lambda->is_visited(pass); }

    void walk_scc(Lambda* cur);

    const Scope& scope;
    Array<Number> numbers;
    size_t pass;
    size_t counter;
    std::vector<Lambda*> stack;
};

void LFBuilder::walk_scc(Lambda* cur) {
    visit(cur);

    for_all (succ, scope.succs(cur)) {
        if (!is_visited(succ)) {
            walk_scc(succ);
            lowlink(cur) = std::min(lowlink(cur), lowlink(succ));
        } else if (is_on_stack(succ))
            lowlink(cur) = std::min(lowlink(cur), dfs(succ));
    }

    // root of SCC
    if (lowlink(cur) == dfs(cur)) {
        Lambda* popped;
        LambdaSet scc;
        do {
            popped = pop();
            scc.insert(popped);
            std::cout << popped->debug << std::endl;
        } while (popped != cur);

        if (scc.size() > 1) {
            for_all (lambda, scc) {
                if (lambda == scope.entry())
                    std::cout << "header: " << lambda->debug << std::endl;
                else {
                    for_all (pred, scope.preds(lambda)) {
                        if (scc.find(pred) == scc.end()) {
                            std::cout << "header: " << pred->debug << " -> " << lambda->debug << std::endl;
                        }
                    }
                }
            }
        }

        std::cout << "---" << std::endl;
    }
}

LoopForestNode* create_loop_forest(const Scope& scope) {
    LFBuilder builder(scope);
    return 0;
}

} // namespace anydsl2
