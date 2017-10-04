#ifndef THORIN_TRANSFORM_IMPORT_H
#define THORIN_TRANSFORM_IMPORT_H

#include "thorin/world.h"

namespace thorin {

class Importer {
public:
    Importer(World& src)
        : world_(src.name())
    {
        if  (src.is_pe_done())
            world_.mark_pe_done();
#ifndef NDEBUG
        if (src.track_history())
            world_.enable_history(true);
#endif
    }

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
