#ifndef DSLU_FOREACH_H
#define DSLU_FOREACH_H

#include <cstddef>
#include <utility>

#include <boost/typeof/typeof.hpp>


namespace anydsl {

template<class T, typename SEL> struct SelIter   { typedef typename T::const_iterator iter; };
template<class T> struct SelIter<T, void (int&)> { typedef typename T::iterator       iter; };
template<class T, class U> struct SelIter<std::pair<T, T>, U> { typedef T iter; };

template<class T> typename T::iterator       begin(T& t)       { return t.begin(); }
template<class T> typename T::const_iterator begin(const T& t) { return t.begin(); }
template<class T, size_t N> T*               begin(T (&t)[N])  { return t; }
template<class T> T                          begin(const std::pair<T, T>& t) { return t.first; }

template<class T> typename T::iterator       end(T& t)       { return t.end(); }
template<class T> typename T::const_iterator end(const T& t) { return t.end(); }
template<class T, size_t N> T*               end(T (&t)[N])  { return t + N; }
template<class T> T                          end(const std::pair<T, T>& t) { return t.second; }

template<class T, typename SEL> struct SelCont   { typedef T const cont; };
template<class T> struct SelCont<T, void (int&)> { typedef T cont; };

} // namespace anydsl

#define LNAME__(name, line) name##__##line
#define LNAME_(name, line)  LNAME__(name, line)
#define LNAME(name)         LNAME_(name, __LINE__)

#define for_all_x(var, what, step) \
    if (bool LNAME(break) = false) {} else \
        if (bool LNAME(once_ref) = true) {} else \
            for (anydsl::SelCont<BOOST_TYPEOF((what)), void (int var)>::cont& LNAME(what_ref) = ((what)); LNAME(once_ref); LNAME(once_ref) = false) \
                for (anydsl::SelIter<BOOST_TYPEOF((what)), void (int var)>::iter LNAME(iter) = anydsl::begin(LNAME(what_ref)), LNAME(end) = anydsl::end(LNAME(what_ref)); !LNAME(break) && LNAME(iter) != LNAME(end); LNAME(break) ? (void)0 : (void)((step), ++LNAME(iter))) \
                    if (bool LNAME(once) = (LNAME(break) = true, false)) {} else for (std::iterator_traits<anydsl::SelIter<BOOST_TYPEOF((what)), void (int var)>::iter>::value_type var = *LNAME(iter); !LNAME(once); LNAME(break) = false, LNAME(once) = true)

#if 0
#define for_all_t_x(var, what, step) \
    if (bool LNAME(break) = false) {} else \
        for (typename anydsl::SelIter<BOOST_TYPEOF((what)), void (int var)>::iter LNAME(iter) = anydsl::begin((what)), LNAME(end) = anydsl::end((what)); !LNAME(break) && LNAME(iter) != LNAME(end); LNAME(break) ? (void)0 : (void)((step), ++LNAME(iter))) \
            if (bool LNAME(once) = (LNAME(break) = true, false)) {} else for (typename std::iterator_traits<typename anydsl::SelIter<BOOST_TYPEOF((what)), void (int var)>::iter>::value_type var = *LNAME(iter); !LNAME(once); LNAME(break) = false, LNAME(once) = true)
#endif

#define for_all( var, what) for_all_x( var, what, (void)0)
#define for_all_t(var, what) for_all_t_x(var, what, (void)0)

#endif // DSLU_FOREACH_H
