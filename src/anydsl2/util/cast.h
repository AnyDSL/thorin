#ifndef ANYDSL2_CAST_HEADER
#define ANYDSL2_CAST_HEADER

#include <cstring>
#include <type_traits>

#include "anydsl2/util/assert.h"

namespace anydsl2 {

/*
 * Watch out for the order of the template parameters in all of these casts.
 * All the casts use the order <L, R>. 
 * This order may seem irritating at first glance, as you read <TO, FROM>.
 * This is solely for the reason, that C++ type deduction does not work the other way round.
 * However, for these kinds of cast you won't have to specify the template parameters explicitly anyway.
 * Thus, you do not have to care except for the bitcast -- so be warned.
 * Just read as <L, R> instead, i.e., 
 * L is the thing which is returned (the left-hand side of the function call),
 * R is the thing which goes in (the right-hand side of a call).
 *
 * NOTE: inline hint seems to be necessary -- at least on my current version of gcc
 */

/// \c static_cast checked in debug version
template<class L, class R>
inline L* scast(R* r) {
    static_assert(std::is_base_of<R, L>(), "R is not a base type of L");
    assert((!r || dynamic_cast<L*>(r)) && "cast not possible" );
    return static_cast<L*>(r);
}

/// \c static_cast checked in debug version -- \c const version
template<class L, class R>
inline const L* scast(const R* r) { return const_cast<const L*>(scast<L, R>(const_cast<R*>(r))); }

/// shorthand for \c dynamic_cast
template<class L, class R>
inline L* dcast(R* u) { 
    static_assert(std::is_base_of<R, L>(), "R is not a base type of L");
    return dynamic_cast<L*>(u); 
}

/// shorthand for \c dynamic_cast -- \c const version
template<class L, class R>
inline const L* dcast(const R* r) { return const_cast<const L*>(dcast<L, R>(const_cast<R*>(r))); }

/** 
 * @brief A bitcast.
 *
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
 * @brief Provides handy \p as and \p isa methods.
 *
 * Inherit from this class in order to use 
 * @code
Bar* bar = foo->as<Bar>();
if (Bar* bar = foo->isa<Bar>()) { ... }
 * @endcode
 * instead of more combersume
 * @code
Bar* bar = anydsl::scast<Bar>(foo);
if (Bar* bar = anydsl::dcast<Bar>(foo)) { ... }
 * @endcode
 */
class MagicCast {
public:
    virtual ~MagicCast() {}

    /**
     * Acts as static cast -- checked for correctness in the debug version.
     * Use if you \em know that \p this is of type \p U.
     * It is a program error (an assertion is raised) if this does not hold.
     */
    template<class U> U* as()  { return anydsl2::scast<U>(this); }
    /** 
     * @brief Acts as dynamic cast.
     * @return \p this cast to \p U if \p this is a \p U, 0 otherwise.
     */
    template<class U> U* isa() { return anydsl2::dcast<U>(this); }
    /// const version of @see MagicCast#as
    template<class U> const U* as()  const { return anydsl2::scast<U>(this); }
    /// const version of @see MagicCast#isa
    template<class U> const U* isa() const { return anydsl2::dcast<U>(this); }
};

} // namespace anydsl2

#endif // DSLU_CAST_HEADER
