#ifndef ANYDSL2_UTIL_PUSH_H
#define ANYDSL2_UTIL_PUSH_H

#define ANYDSL2_LNAME__(name, line) name##__##line
#define ANYDSL2_LNAME_(name, line)  ANYDSL2_LNAME__(name, line)
#define ANYDSL2_LNAME(name)         ANYDSL2_LNAME_(name, __LINE__)
 
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

template<class T, class U>
inline Push<T> push(T& t, U new_val) { return Push<T>(t, new_val); }

#define ANYDSL2_PUSH(what, with) auto ANYDSL2_LNAME(anydsl2_push) = anydsl2::push((what), (with))

} // namespace anydsl2

#endif // UTIL_PUSH_H
