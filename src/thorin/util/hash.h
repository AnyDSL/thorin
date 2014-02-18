#ifndef THORIN_UTIL_HASH_H
#define THORIN_UTIL_HASH_H

#include <functional>

namespace thorin {

template<class T> 
inline size_t hash_value(const T& t) { return std::hash<T>()(t); }
template<class T>
inline size_t hash_combine(size_t seed, const T& val) { return seed ^ (hash_value(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2)); }

}

#endif
