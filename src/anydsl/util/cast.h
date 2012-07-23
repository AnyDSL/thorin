#ifndef DSLU_CAST_HEADER
#define DSLU_CAST_HEADER

#include <cstring>

#include "anydsl/util/assert.h"

namespace anydsl {

/*
 * Watch out for the order of the template parameters in all of these casts.
 * All the casts use the order <LEFT, RIGHT>. 
 * This order may seem irritating at first glance, as you read <TO, FROM>.
 * This is solely for the reason, that C++ type deduction does not work the other way round.
 * However, for these kind of cast you won't have to specify the template parameters explicitly anyway.
 * Thus you do not have to care except for the bitcast -- so be warned.
 * Just read as <LEFT, RIGHT> instead, i.e., 
 * LEFT is the thing which is return (the left-hand side of the function call),
 * RIGHT is the thing which goes in (the right-hand side of a call).
 *
 * NOTE: inline hint seems to be necessary -- at least on my current version of gcc
 */

/// \c static_cast checked in debug version
template<class LEFT, class RIGHT>
inline LEFT* scast(RIGHT* u) {
    anydsl_assert( !u || dynamic_cast<LEFT*>(u), "cast not possible" );
    return static_cast<LEFT*>(u);
}

/// \c static_cast checked in debug version -- \c const version
template<class LEFT, class RIGHT>
inline const LEFT* scast(const RIGHT* u) {
    anydsl_assert( !u || dynamic_cast<const LEFT*>(u), "cast not possible" );
    return static_cast<const LEFT*>(u);
}

/// shorthand for \c dynamic_cast
template<class LEFT, class RIGHT>
inline LEFT* dcast(RIGHT* u) {
    return dynamic_cast<LEFT*>(u);
}

/// shorthand for \c dynamic_cast -- \c const version
template<class LEFT, class RIGHT>
inline const LEFT* dcast(const RIGHT* u) {
    return dynamic_cast<const LEFT*>(u);
}

/// shorthand for \c reinterpret_cast
template<class LEFT, class RIGHT>
inline LEFT* rcast(RIGHT* u) {
    return reinterpret_cast<LEFT*>(u);
}

/// shorthand for \c reinterpret_cast -- \c const version
template<class LEFT, class RIGHT>
inline const LEFT* rcast(const RIGHT* u) {
    return reinterpret_cast<const LEFT*>(u);
}
/// shorthand for \c const_cast -- removes constness
template<class LEFT, class RIGHT>
inline LEFT* ccast(const RIGHT* u) {
    return const_cast<LEFT*>(u);
}

/// shorthand for \c const_cast -- adds constness
template<class LEFT, class RIGHT>
inline const LEFT* ccast(RIGHT* u) {
    return const_cast<const LEFT*>(u);
}

/** 
 * @brief A bitcast.
 *
 * The bitcast requires both types to be of the same size.
 * Watch out for the order of the template parameters!
 */
template<class LEFT, class RIGHT>
inline LEFT bcast(const RIGHT& from) {
    anydsl_assert(sizeof(RIGHT) == sizeof(LEFT), "size mismatch for bitcast");
    LEFT to;
    memcpy(&to, &from, sizeof(LEFT));
    return to;
}

/** 
 * @brief Provides handy \p as and \p isa methods.
 *
 * Inherit from this class in order to use 
 * \code
Bar* bar = foo->as<Bar>();
if (Bar* bar = foo->isa<Bar>()) { ... }
 * \endcode
 * instead of more combersume
 * \code
Bar* bar = anydsl::scast<Bar>(foo);
if (Bar* bar = anydsl::dcast<Bar>(foo)) { ... }
 * \endcode
 */
class MagicCast {
public:

    virtual ~MagicCast() {}

    template<class T> T* as()  { return anydsl::scast<T>(this); }
    template<class T> T* isa() { return anydsl::dcast<T>(this); }
    template<class T> const T* as()  const { return anydsl::scast<T>(this); }
    template<class T> const T* isa() const { return anydsl::dcast<T>(this); }
};

} // namespace anydsl

#endif // DSLU_CAST_HEADER
