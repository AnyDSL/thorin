#ifndef ANYDSL2_ANALYSES_VERIFIER_H
#define ANYDSL2_ANALYSES_VERIFIER_H

#include "anydsl2/lambda.h"

namespace anydsl2 {

class World;

enum InvalidEntryType {
    INVALID_CE = 0,

    INVALID_NOT_IN_WORLD,
    INVALID_TYPES,
    INVALID_TYPES_GENERICS,
    INVALID_STRUCTURE,
    INVALID_CYCLIC_DEPENDENCY
};

class InvalidEntry {
public:
    InvalidEntry(InvalidEntryType type, const Def* def);
    InvalidEntry(const Def* def, const Def* source);

    const Def* def() const { return def_; }
    const Def* source() const { return source_; }
    InvalidEntryType type() const { return type_; }

    bool isConsequentialError() const { return source_ != def_; }

    void dump();

private:
    InvalidEntryType type_;
    const Def* def_;
    const Def* source_;
};

typedef std::vector<InvalidEntry> InvalidEntries;

bool verify(World& world, InvalidEntries& invalid);

template<bool dump>
bool verify(World& world) {
    InvalidEntries invalid;
    bool result = verify(world, invalid);
    if(dump && !result) {
        // dump all invalid entries
        for_all(invalidOne, invalid)
            invalidOne.dump();
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
