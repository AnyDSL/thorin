#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/stream.h"

namespace thorin {

struct ScopedWorld : public Streamable<ScopedWorld> {
    ScopedWorld(World& w) : world_(w), forest_(w) {}

    World& world_;
    mutable ScopesForest forest_;

    mutable DefSet done_;
    mutable ContinuationMap<std::unique_ptr<std::vector<const Def*>>> scopes_to_defs_;
    mutable std::vector<const Def*> top_lvl_;

    Stream& stream(Stream&) const;
private:

    void stream_cont(thorin::Stream& s, Continuation* cont) const;
    void prepare_def(Continuation* in, const Def* def) const;
    static void stream_op(thorin::Stream&, const Def* op);
    static void stream_ops(thorin::Stream& s, Defs defs);
    static void stream_def(thorin::Stream& s, const Def* def);
    static void stream_defs(thorin::Stream& s, std::vector<const Def*>& defs);
};

std::unique_ptr<ScopedWorld> scoped_world(World& w);

}
