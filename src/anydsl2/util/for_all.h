#ifndef ANYDSL2_FOREACH_H
#define ANYDSL2_FOREACH_H

#include <cstddef>
#include <utility>

#include <boost/typeof/typeof.hpp>


namespace anydsl2 {

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

} // namespace anydsl2

#define LNAME__(name, line) name##__##line
#define LNAME_(name, line)  LNAME__(name, line)
#define LNAME(name)         LNAME_(name, __LINE__)

#define for_all_x(var, what, step) \
    if (bool LNAME(break) = false) {} else \
        if (bool LNAME(once_ref) = false) {} else \
            for (anydsl2::SelCont<BOOST_TYPEOF((what)), void (int var)>::cont& LNAME(what_ref) = ((what)); !LNAME(once_ref); LNAME(once_ref) = true) \
                for (anydsl2::SelIter<BOOST_TYPEOF((what)), void (int var)>::iter LNAME(iter) = anydsl2::begin(LNAME(what_ref)), LNAME(end) = anydsl2::end(LNAME(what_ref)); !LNAME(break) && LNAME(iter) != LNAME(end); LNAME(break) ? (void)0 : (void)((step), ++LNAME(iter))) \
                    if (bool LNAME(once) = (LNAME(break) = true, false)) {} else for (std::iterator_traits<anydsl2::SelIter<BOOST_TYPEOF((what)), void (int var)>::iter>::value_type var = *LNAME(iter); !LNAME(once); LNAME(break) = false, LNAME(once) = true)

#define for_all2_x(var1, what1, var2, what2, step) \
    if (bool LNAME(break) = false) {} else \
        if (bool LNAME(once_ref) = false) {} else \
            for (anydsl2::SelCont<BOOST_TYPEOF((what1)), void (int var1)>::cont& LNAME(what_ref1) = ((what1)); !LNAME(once_ref); LNAME(once_ref) = true) \
                for (anydsl2::SelCont<BOOST_TYPEOF((what2)), void (int var2)>::cont& LNAME(what_ref2) = ((what2)); !LNAME(once_ref); LNAME(once_ref) = true) \
                    for (anydsl2::SelIter<BOOST_TYPEOF((what2)), void (int var2)>::iter LNAME(iter2) = anydsl2::begin(LNAME(what_ref2)); !LNAME(once_ref); LNAME(once_ref) = true) \
                        for (anydsl2::SelIter<BOOST_TYPEOF((what1)), void (int var1)>::iter LNAME(iter1) = anydsl2::begin(LNAME(what_ref1)), LNAME(end) = anydsl2::end(LNAME(what_ref1)); !LNAME(break) && LNAME(iter1) != LNAME(end); LNAME(break) ? (void)0 : (void)((step), ++LNAME(iter1), ++LNAME(iter2))) \
                            if (bool LNAME(once) = (LNAME(break) = true, false)) {} else for (std::iterator_traits<anydsl2::SelIter<BOOST_TYPEOF((what1)), void (int var1)>::iter>::value_type var1 = *LNAME(iter1); !LNAME(once); LNAME(once) = true) \
                                for (std::iterator_traits<anydsl2::SelIter<BOOST_TYPEOF((what2)), void (int var2)>::iter>::value_type var2 = *LNAME(iter2); !LNAME(once); LNAME(break) = false, LNAME(once) = true)

#define for_all(var, what) for_all_x( var, what, (void)0)
#define for_all2(var1, what1, var2, what2) for_all2_x(var1, what1, var2, what2, (void)0)

#endif // DSLU_FOREACH_H
