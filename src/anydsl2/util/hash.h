#ifndef ANYDSL2_UTIL_HASH_H
#define ANYDSL2_UTIL_HASH_H

#include <functional>

namespace anydsl2 {

template<class T> 
inline size_t hash_value(const T& t) { return std::hash<T>()(t); }
template<class T>
inline size_t hash_combine(size_t seed, const T& t) { return seed ^ (hash_value(t) + 0x9e3779b9 + (seed << 6) + (seed >> 2)); }

} // namespace util

#endif
