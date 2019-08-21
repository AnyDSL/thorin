#ifndef THORIN_UTIL_CAST_H
#define THORIN_UTIL_CAST_H

#include <cassert>
#include <cstring>

namespace thorin {

/// A bitcast.
template<class D, class S>
inline D bcast(const S& src) {
    D dst;
    auto s = reinterpret_cast<const void*>(&src);
    auto d = reinterpret_cast<void*>(&dst);

    if constexpr(sizeof(D) == sizeof(S)) memcpy(d, s, sizeof(D));
    if constexpr(sizeof(D)  < sizeof(S)) memcpy(d, s, sizeof(D));
    if constexpr(sizeof(D)  > sizeof(S)) {
        memset(d, 0, sizeof(D));
        memcpy(d, s, sizeof(S));
    }
    return dst;
}

/// Provides a @c dynamic_cast -like feature without resorting to RTTI.
template<class Base>
class RuntimeCast {
public:
    template<class T>       T* isa()       { return static_cast<      Base*>(this)->tag() == T::Tag ? static_cast<      T*>(this) : nullptr; } ///< Dynamic cast.
    template<class T> const T* isa() const { return static_cast<const Base*>(this)->tag() == T::Tag ? static_cast<const T*>(this) : nullptr; } ///< Dynamic cast. @c const version.
    template<class T>       T* as()       { assert(isa<T>()); return static_cast<      T*>(this); }                                            ///< Static cast with debug check.
    template<class T> const T* as() const { assert(isa<T>()); return static_cast<const T*>(this); }                                            ///< Static cast with debug check. @c const version.
};

}

#endif
