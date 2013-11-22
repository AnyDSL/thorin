#ifndef THORIN_DEFSET_H
#define THORIN_DEFSET_H

#include <unordered_set>

#include "thorin/def.h"

namespace thorin {

class DefSet : public std::unordered_set<const DefNode*, DefNodeHash, DefNodeEqual> {
public:
    typedef std::unordered_set<const DefNode*, DefNodeHash, DefNodeEqual> Super;

    bool contains(const DefNode* def) { return Super::find(def) != Super::end(); }
};

}

#endif

