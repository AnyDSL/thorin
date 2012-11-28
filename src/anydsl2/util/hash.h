#ifndef ANYDSL2_UTIL_HASH_H
#define ANYDSL2_UTIL_HASH_H

#include <boost/functional/hash.hpp>
#include <boost/tuple/tuple.hpp>

namespace anydsl2 {


template<class T>
inline size_t hash_combine(size_t seed, const T& t) { boost::hash_combine(seed, t); return seed; }

inline size_t hash_tuple(size_t seed, boost::tuples::null_type) { return seed; }

template<class T>
inline size_t hash_tuple(size_t seed, const T& tuple) { 
    return hash_tuple(hash_combine(seed, tuple.get_head()), tuple.get_tail());
}

template<class T>
inline size_t hash_tuple(const T& tuple) { return hash_tuple(0, tuple); }

} // namespace util

#endif
