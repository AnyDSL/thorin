#ifndef DSLU_FOREACH_H
#define DSLU_FOREACH_H

#include <cstddef>

#include <boost/typeof/typeof.hpp>


namespace anydsl {

template<class T, typename SEL> struct SelIter   { typedef typename T::const_iterator iter; };
template<class T> struct SelIter<T, void (int&)> { typedef typename T::iterator       iter; };

template<class T> typename T::iterator       begin(T& t)       { return t.begin(); }
template<class T> typename T::const_iterator begin(const T& t) { return t.begin(); }
template<class T, size_t N> T*               begin(T (&t)[N])  { return t; }

template<class T> typename T::iterator       end(T& t)       { return t.end(); }
template<class T> typename T::const_iterator end(const T& t) { return t.end(); }
template<class T, size_t N> T*               end(T (&t)[N])  { return t + N; }

} // namespace anydsl

#define LNAME__(name, line) name##__##line
#define LNAME_(name, line)  LNAME__(name, line)
#define LNAME(name)         LNAME_(name, __LINE__)

#define FOREACHX(var, what, step) \
    if (bool LNAME(break) = false) {} else \
        for (anydsl::SelIter<BOOST_TYPEOF((what)), void (int var)>::iter LNAME(iter) = anydsl::begin((what)), LNAME(end) = anydsl::end((what)); !LNAME(break) && LNAME(iter) != LNAME(end); LNAME(break) ? (void)0 : (void)((step), ++LNAME(iter))) \
            if (bool LNAME(once) = (LNAME(break) = true, false)) {} else for (std::iterator_traits<anydsl::SelIter<BOOST_TYPEOF((what)), void (int var)>::iter>::value_type var = *LNAME(iter); !LNAME(once); LNAME(break) = false, LNAME(once) = true)

#define FOREACHTX(var, what, step) \
    if (bool LNAME(break) = false) {} else \
        for (typename anydsl::SelIter<BOOST_TYPEOF((what)), void (int var)>::iter LNAME(iter) = anydsl::begin((what)), LNAME(end) = anydsl::end((what)); !LNAME(break) && LNAME(iter) != LNAME(end); LNAME(break) ? (void)0 : (void)((step), ++LNAME(iter))) \
            if (bool LNAME(once) = (LNAME(break) = true, false)) {} else for (typename std::iterator_traits<typename anydsl::SelIter<BOOST_TYPEOF((what)), void (int var)>::iter>::value_type var = *LNAME(iter); !LNAME(once); LNAME(break) = false, LNAME(once) = true)

#define FOREACH( var, what) FOREACHX( var, what, (void)0)
#define FOREACHT(var, what) FOREACHTX(var, what, (void)0)

#endif // DSLU_FOREACH_H
