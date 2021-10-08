#ifndef THORIN_UTIL_CONTAINER_H
#define THORIN_UTIL_CONTAINER_H

#include <stack>
#include <queue>

namespace thorin {

template<class S>
auto pop(S& s) -> decltype(s.top(), typename S::value_type()) {
    auto val = s.top();
    s.pop();
    return val;
}

template<class Q>
auto pop(Q& q) -> decltype(q.front(), typename Q::value_type()) {
    auto val = q.front();
    q.pop();
    return val;
}

template<class I, class T, class Cmp>
I binary_find(I begin, I end, T val, Cmp cmp) {
    auto i = std::lower_bound(begin, end, val, cmp);
    return (i != end && !(cmp(val, *i))) ? i : end;
}

template<class Set>
class unique_stack {
public:
    using T = typename std::remove_reference_t<Set>::value_type;

    bool push(T val) {
        if (done_.emplace(val).second) {
            stack_.emplace(val);
            return true;
        }
        return false;
    }

    bool empty() const { return stack_.empty(); }
    const T& top() { return stack_.top(); }
    T pop() { return thorin::pop(stack_); }
    void clear() { done_.clear(); stack_ = {}; }

private:
    Set done_;
    std::stack<T> stack_;
};

template<class Set>
class unique_queue {
public:
    using T = typename std::remove_reference_t<Set>::value_type;

    unique_queue() = default;
    unique_queue(Set set)
        : done_(set)
    {}

    bool push(T val) {
        if (done_.emplace(val).second) {
            queue_.emplace(val);
            return true;
        }
        return false;
    }

    bool empty() const { return queue_.empty(); }
    T pop() { return thorin::pop(queue_); }
    T& front() { return queue_.front(); }
    T& back() { return queue_.back(); }
    void clear() { done_.clear(); queue_ = {}; }

private:
    Set done_;
    std::queue<T> queue_;
};

}

#endif
