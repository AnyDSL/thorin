#ifndef THORIN_UTIL_QUEUE_H
#define THORIN_UTIL_QUEUE_H

#include <queue>
#include <stack>

namespace thorin {

template<class T>
T pop(std::stack<T>& stack) {
    auto val = stack.top();
    stack.pop();
    return val;
}

template<class T>
T pop(std::queue<T>& queue) {
    auto val = queue.front();
    queue.pop();
    return val;
}

}

#endif
