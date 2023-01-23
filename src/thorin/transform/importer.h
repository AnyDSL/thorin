#ifndef THORIN_TRANSFORM_IMPORT_H
#define THORIN_TRANSFORM_IMPORT_H

#include "thorin/world.h"
#include "thorin/config.h"

namespace thorin {

class Importer {
public:
    explicit Importer(World& src, World& dst)
        : src(src)
        , dst(dst)
    {
        if (src.is_pe_done())
            world().mark_pe_done();
#if THORIN_ENABLE_CHECKS
        if (src.track_history())
            world().enable_history(true);
#endif
    }

    World& world() { return dst; }
    //const Type* import(const Type*);
    const Def* import(const Def*);
    bool todo() const { return todo_; }

public:
    Def2Def def_old2new_;
    World& src;
    World& dst;
    bool todo_ = false;
};

}

#endif
