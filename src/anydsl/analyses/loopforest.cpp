#include "anydsl/analyses/loopforest.h"

#include "anydsl/lambda.h"
#include "anydsl/analyses/scope.h"

#include <algorithm>

namespace anydsl2 {

static void on_stack(Lambda* lambda) { lambda->mark(); }
static void not_on_stack(Lambda* lambda) { lambda->unmark(); }
static bool is_on_stack(Lambda* lambda) { return lambda->is_marked(); }
LoopForest::Number& LoopForest::number(Lambda* lambda) { return numbers_[lambda->sid()]; }
size_t& LoopForest::lowlink(Lambda* lambda) { return number(lambda).low; }
size_t& LoopForest::dfs(Lambda* lambda) { return number(lambda).dfs; }
bool LoopForest::visited(Lambda* lambda) { return number(lambda).dfs != size_t(-1); }

void LoopForest::push(Lambda* lambda) { 
    on_stack(lambda); 
    stack_.push_back(lambda); 
}

Lambda* LoopForest::pop() { 
    Lambda* res = stack_.back(); 
    not_on_stack(res); 
    stack_.pop_back(); 
    return res; 
}

LoopForest::LoopForest(const Scope& scope) 
    : scope_(scope)
    , numbers_(scope.size())
{
    assert(stack_.empty());
    counter_ = 0;
    for_all (lambda, scope_.rpo())
        not_on_stack(lambda);

    walk_scc(scope.entry());
}


void LoopForest::walk_scc(Lambda* cur) {
    number(cur) = Number(counter_++);
    push(cur);

    for_all (succ, scope_.succs(cur)) {
        if (!visited(succ)) {
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
                if (lambda == scope_.entry())
                    std::cout << "header: " << lambda->debug << std::endl;
                else {
                    for_all (pred, scope_.preds(lambda)) {
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

} // namespace anydsl2
