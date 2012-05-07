#ifndef DSLU_SINGLETON_HEADER
#define DSLU_SINGLETON_HEADER

namespace anydsl {

template <class T>
class Singleton {
public:
    static T& This() {
        static T t;
        return t;
    }
};

}

#endif
