#ifndef THORIN_UTIL_QUEUE_H
#define THORIN_UTIL_QUEUE_H

#include <queue>

namespace thorin {

template<class T>
T pop(std::queue<T>& queue) {
    auto val = queue.front();
    queue.pop();
    return val;
}

}

#endif
