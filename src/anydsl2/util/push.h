#ifndef ANYDSL2_UTIL_PUSH_H
#define ANYDSL2_UTIL_PUSH_H

namespace anydsl2 {

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

} // namespace anydsl2

#endif // UTIL_PUSH_H
