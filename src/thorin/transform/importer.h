#ifndef THORIN_TRANSFORM_IMPORT_H
#define THORIN_TRANSFORM_IMPORT_H

#include "thorin/world.h"
#include "thorin/config.h"

namespace thorin {

class Importer {
public:
    Importer(World& src);

    World& world() { return world_; }
    const Def* import(Tracker);
    bool todo() const { return todo_; }

public:
    Def2Def old2new_;
    World world_;
    bool todo_ = false;
};

}

#endif
