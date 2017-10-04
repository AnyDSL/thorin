#ifndef THORIN_TRANSFORM_IMPORT_H
#define THORIN_TRANSFORM_IMPORT_H

#include "thorin/world.h"

namespace thorin {

class Importer {
public:
    Importer(std::string name)
        : world_(name)
    {}

    World& world() { return world_; }
    const Type* import(const Type*);
    const Def* import(Tracker);
    bool todo() const { return todo_; }

public:
    Type2Type type_old2new_;
    Def2Def def_old2new_;
    World world_;
    bool todo_ = false;
};

}

#endif
