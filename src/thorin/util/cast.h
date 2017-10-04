#ifndef THORIN_UTIL_CAST_H
#define THORIN_UTIL_CAST_H

#include <type_traits>

#include "thorin/util/utility.h"

namespace thorin {

/*
 * Watch out for the order of the template parameters in all of these casts.
 * All the casts use the order <L, R>.
 * This order may seem irritating at first glance, as you read <TO, FROM>.
 * This is solely for the reason, that C++ type deduction does not work the other way round.
 * However, for these tags of cast you won't have to specify the template parameters explicitly anyway.
 * Thus, you do not have to care except for the bitcast -- so be warned.
 * Just read as <L, R> instead, i.e.,
 * L is the thing which is returned (the left-hand side of the function call),
 * R is the thing which goes in (the right-hand side of a call).
 */

/// @c static_cast checked in debug version
template<class L, class R>
inline L* scast(R* r) {
    static_assert(std::is_base_of<R, L>::value, "R is not a base type of L");
    assert((!r || dynamic_cast<L*>(r)) && "cast not possible" );
    return static_cast<L*>(r);
}

/// @c static_cast checked in debug version -- @c const version
template<class L, class R>
inline const L* scast(const R* r) { return const_cast<const L*>(scast<L, R>(const_cast<R*>(r))); }

/// shorthand for @c dynamic_cast
template<class L, class R>
inline L* dcast(R* u) {
    static_assert(std::is_base_of<R, L>::value, "R is not a base type of L");
    return dynamic_cast<L*>(u);
}

/// shorthand for @c dynamic_cast -- @c const version
template<class L, class R>
inline const L* dcast(const R* r) { return const_cast<const L*>(dcast<L, R>(const_cast<R*>(r))); }

/**
 * A bitcast.
 * The bitcast requires both types to be of the same size.
 * Watch out for the order of the template parameters!
 */
template<class L, class R>
inline L bcast(const R& from) {
    static_assert(sizeof(R) == sizeof(L), "size mismatch for bitcast");
    L to;
    memcpy(&to, &from, sizeof(L));
    return to;
}

/**
 * Provides handy @p as and @p isa methods.
 * Inherit from this class in order to use
 * @code
Bar* bar = foo->as<Bar>();
if (Bar* bar = foo->isa<Bar>()) { ... }
 * @endcode
 * instead of more combersume
 * @code
Bar* bar = thorin::scast<Bar>(foo);
if (Bar* bar = thorin::dcast<Bar>(foo)) { ... }
 * @endcode
 */
template<class Base>
class RuntimeCast {
public:
    /**
     * Acts as static cast -- checked for correctness in the debug version.
     * Use if you @em know that @p this is of type @p To.
     * It is a program error (an assertion is raised) if this does not hold.
     */
    template<class To> To* as() { return thorin::scast<To>(static_cast<Base*>(this)); }

    /**
     * Acts as dynamic cast.
     * @return @p this cast to @p To if @p this is a @p To, 0 otherwise.
     */
    template<class To> To* isa() { return thorin::dcast<To>(static_cast<Base*>(this)); }

    ///< @c const version of @see RuntimeCast#as.
    template<class To>
    const To* as()  const { return thorin::scast<To>(static_cast<const Base*>(this)); }

    ///< @c const version of @see RuntimeCast#isa.
    template<class To>
    const To* isa() const { return thorin::dcast<To>(static_cast<const Base*>(this)); }
};

}

#endif
