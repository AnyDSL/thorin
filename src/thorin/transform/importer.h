#ifndef THORIN_TRANSFORM_IMPORT_H
#define THORIN_TRANSFORM_IMPORT_H

#include "thorin/world.h"
#include "thorin/config.h"

namespace thorin {

class Importer {
public:
    Importer(World& src)
        : world_(src)
    {
        if (src.is_pe_done())
            world_.mark_pe_done();
#if THORIN_ENABLE_CHECKS
        if (src.track_history())
            world_.enable_history(true);
#endif
    }

    World& world() { return world_; }
    const Type* import(const Type*);
    const Def* import(const Def*);
    void import_plugin_intrinsic(const Continuation* cont, unique_plugin_intrinsic impl);
    bool todo() const { return todo_; }

public:
    Type2Type type_old2new_;
    Def2Def def_old2new_;
    World world_;
    bool todo_ = false;
};

}

#endif
