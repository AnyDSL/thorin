#ifndef ANALYSES_LOOPS_H
#define ANALYSES_LOOPS_H

#include "anydsl/util/array.h"

#include <vector>

namespace anydsl2 {

class Lambda;
class Scope;

class LoopForest {
public:

    LoopForest(const Scope& scope);

    const Scope& scope() const { return scope_; }

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

    void walk_scc(Lambda* cur);
    bool visited(Lambda* lambda);
    Number& number(Lambda* lambda);
    size_t& lowlink(Lambda* lambda);
    size_t& dfs(Lambda* lambda);
    void push(Lambda* lambda);
    Lambda* pop();

    const Scope& scope_;

    Array<Number> numbers_;
    size_t counter_;
    std::vector<Lambda*> stack_;
    //LambdaSet ignore_;
};

} // namespace anydsl2

#endif // ANALYSES_LOOPS_H
