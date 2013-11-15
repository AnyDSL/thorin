#ifndef THORIN_UTIL_PUSH_H
#define THORIN_UTIL_PUSH_H

#define THORIN_LNAME__(name, line) name##__##line
#define THORIN_LNAME_(name, line)  THORIN_LNAME__(name, line)
#define THORIN_LNAME(name)         THORIN_LNAME_(name, __LINE__)
 
namespace thorin {

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

#define THORIN_PUSH(what, with) auto THORIN_LNAME(thorin_push) = thorin::push((what), (with))

} // namespace thorin

#endif // UTIL_PUSH_H
