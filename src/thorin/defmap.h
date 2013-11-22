#ifndef THORIN_DEFMAP_H
#define THORIN_DEFMAP_H

#include <unordered_map>

#include "thorin/def.h"

namespace thorin {

template<class Value>
class DefMap : public std::unordered_map<const DefNode*, Value, DefNodeHash, DefNodeEqual> {
public:
    typedef std::unordered_map<const DefNode*, Value, DefNodeHash, DefNodeEqual> Super;
};

template<class Value>
class DefMap<Value*> : public std::unordered_map<const DefNode*, Value*, DefNodeHash, DefNodeEqual> {
public:
    typedef std::unordered_map<const DefNode*, Value*, DefNodeHash, DefNodeEqual> Super;

    Value* find(const DefNode* def) const {
        auto i = Super::find(def);
        return i == Super::end() ? nullptr : i->first.second;
    }
};

}

#endif
