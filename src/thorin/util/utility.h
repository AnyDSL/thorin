#ifndef THORIN_UTILITY_H
#define THORIN_UTILITY_H

#include <cassert>
#include <memory>
#include <queue>

#ifndef _MSC_VER
#define THORIN_UNREACHABLE do { assert(true && "unreachable"); abort(); } while(0)
#else // _MSC_VER
inline __declspec(noreturn) void thorin_dummy_function() { abort(); }
#define THORIN_UNREACHABLE do { assert(true && "unreachable"); thorin_dummy_function(); } while(0)
#endif

#ifndef NDEBUG
#define THORIN_CALL_ONCE do { static bool once = true; assert(once); once=false; } while(0)
#define assert_unused(x) assert(x)
#else
#define THORIN_CALL_ONCE
#define assert_unused(x) ((void) (0 && (x)))
#endif

#define THORIN_IMPLIES(a, b) (!(a) || ((a) && (b)))

// http://stackoverflow.com/questions/1489932/how-to-concatenate-twice-with-the-c-preprocessor-and-expand-a-macro-as-in-arg
#define THORIN_PASTER(x,y) x ## y

namespace thorin {

/// Use to initialize an \p AutoPtr in a lazy way.
template<class This, class T>
inline T& lazy_init(const This* self, std::unique_ptr<T>& ptr) {
    return *(ptr ? ptr : (ptr.reset(new T(*self)), ptr));
}

template<class T>
T pop(std::queue<T>& queue) {
    auto val = queue.front();
    queue.pop();
    return val;
}

template<class T>
struct Push {
    Push(T& t, T new_val)
        : old_(t)
        , ref_(t)
    {
        t = new_val;
    }
    ~Push() { ref_ = old_; }

private:
    T old_;
    T& ref_;
};

template<class T, class U>
inline Push<T> push(T& t, U new_val) { return Push<T>(t, new_val); }

#define THORIN_LNAME__(name, line) name##__##line
#define THORIN_LNAME_(name, line)  THORIN_LNAME__(name, line)
#define THORIN_LNAME(name)         THORIN_LNAME_(name, __LINE__)

#define THORIN_PUSH(what, with) auto THORIN_LNAME(thorin_push) = thorin::push((what), (with))

/// Determines whether @p i is a power of two.
constexpr size_t is_power_of_2(size_t i) { return ((i != 0) && !(i & (i - 1))); }

constexpr unsigned log2(unsigned n, unsigned p = 0) { return (n <= 1) ? p : log2(n / 2, p + 1); }

constexpr uint32_t round_to_power_of_2(uint32_t i) {
    i--;
    i |= i >> 1;
    i |= i >> 2;
    i |= i >> 4;
    i |= i >> 8;
    i |= i >> 16;
    i++;
    return i;
}

}

#endif
