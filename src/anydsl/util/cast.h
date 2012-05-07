#ifndef DSLU_CAST_HEADER
#define DSLU_CAST_HEADER

#include <cstring>

#include "anydsl/util/assert.h"

namespace anydsl {

/*
 * Watch out for the order of the template parameters in all of these casts.
 * All the casts use the order <TO, FROM>. 
 * This order may seem irritating at first glance.
 * This is solely for the reason, that C++ type deduction does not work the other way round.
 * However, for these kind of cast you won't have to specify the template parameters explicitly anyway.
 * Thus you do not have to care except for the bitcast -- so be warned.
 *
 * note: inline hint seems to be necessary -- at least on my current version of gcc
 */

/// \c static_cast checked in debug version
template<class TO, class FROM>
inline TO* scast(FROM* u) {
    anydsl_assert( dynamic_cast<TO*>(u), "cast not possible" );
    return static_cast<TO*>(u);
}

/// \c static_cast checked in debug version -- \c const version
template<class TO, class FROM>
inline const TO* scast(const FROM* u) {
    anydsl_assert( dynamic_cast<const TO*>(u), "cast not possible" );
    return static_cast<const TO*>(u);
}

/// shorthand for \c dynamic_cast
template<class TO, class FROM>
inline TO* dcast(FROM* u) {
    return dynamic_cast<TO*>(u);
}

/// shorthand for \c dynamic_cast -- \c const version
template<class TO, class FROM>
inline const TO* dcast(const FROM* u) {
    return dynamic_cast<const TO*>(u);
}

/// shorthand for \c reinterpret_cast
template<class TO, class FROM>
inline TO* rcast(FROM* u) {
    return reinterpret_cast<TO*>(u);
}

/// shorthand for \c reinterpret_cast -- \c const version
template<class TO, class FROM>
inline const TO* rcast(const FROM* u) {
    return reinterpret_cast<const TO*>(u);
}
/// shorthand for \c const_cast -- removes constness
template<class TO, class FROM>
inline TO* ccast(const FROM* u) {
    return const_cast<TO*>(u);
}

/// shorthand for \c const_cast -- adds constness
template<class TO, class FROM>
inline const TO* ccast(FROM* u) {
    return const_cast<const TO*>(u);
}

/** 
 * @brief A bitcast.
 *
 * The bitcast requires both types to be of the same size.
 * Watch out for the order of the template parameters!
 */
template<class TO, class FROM>
inline TO bcast(const FROM& from) {
    anydsl_assert(sizeof(FROM) == sizeof(TO), "size mismatch for bitcast");
    TO to;
    memcpy(&to, &from, sizeof(TO));
    return to;
}

} // namespace anydsl

#endif // DSLU_CAST_HEADER
