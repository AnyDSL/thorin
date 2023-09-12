#ifndef THORIN_TRANSFORM_IMPORT_H
#define THORIN_TRANSFORM_IMPORT_H

#include "thorin/world.h"
#include "thorin/config.h"
#include "thorin/transform/rewrite.h"
#include "thorin/analyses/scope.h"

namespace thorin {

class Importer : Rewriter {
public:
    explicit Importer(World& src, World& dst)
        : Rewriter(src, dst), forest_(std::make_unique<ScopesForest>(src))
    {
        assert(&src != &dst);
        if (src.is_pe_done())
            dst.mark_pe_done();
    }

    const Def* import(const Def* odef) { return instantiate(odef); }
    bool todo() const { return todo_; }

protected:
    const Def* rewrite(const Def* odef) override;

private:
    std::unique_ptr<ScopesForest> forest_;
    bool todo_ = false;
};

}

#endif
