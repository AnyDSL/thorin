#ifndef ANYDSL2_ANALYSES_VERIFIER_H
#define ANYDSL2_ANALYSES_VERIFIER_H

#include "anydsl2/lambda.h"

namespace anydsl2 {

class World;

bool verify(World& world, Lambdas& invalid);

template<bool dump>
bool verify(World& world) {
    Lambdas invalid;
    bool result = verify(world, invalid);
    if(dump && !result) {
        // dump all invalid entries
        for_all(lambda, invalid)
            lambda->dump();
    }
    return result;
}

inline bool pureVerify(World& world) {
    return verify<false>(world);
}

inline bool verifyAndDump(World& world) {
    return verify<true>(world);
}

} // namespace anydsl2

#endif
