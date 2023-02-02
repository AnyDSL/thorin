#ifndef THORIN_TRANSFORM_IMPORT_H
#define THORIN_TRANSFORM_IMPORT_H

#include "thorin/world.h"
#include "thorin/config.h"
#include "thorin/transform/rewrite.h"

namespace thorin {

class Importer : Rewriter {
public:
    explicit Importer(World& src, World& dst)
        : Rewriter(src, dst)
    {

        if (src.is_pe_done())
            dst.mark_pe_done();
#if THORIN_ENABLE_CHECKS
        if (src.track_history())
            dst.enable_history(true);
#endif
    }

    const Def* import(const Def* odef) { return instantiate(odef); }
    bool todo() const { return todo_; }

protected:
    const Def* rewrite(const Def* odef) override;

private:
    bool todo_ = false;
};

}

#endif
