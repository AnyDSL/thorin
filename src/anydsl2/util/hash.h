#ifndef ANYDSL2_UTIL_HASH_H
#define ANYDSL2_UTIL_HASH_H

#include <boost/functional/hash.hpp>

namespace anydsl2 {

template<class T>
inline size_t hash_combine(size_t seed, const T& t) { boost::hash_combine(seed, t); return seed; }

} // namespace util

#endif
