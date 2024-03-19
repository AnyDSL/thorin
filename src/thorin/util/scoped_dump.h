#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/stream.h"

namespace thorin {

#define COLORS(C) \
C(Black, "\u001b[30m") \
C(Red, "\u001b[31m") \
C(Green, "\u001b[32m") \
C(Yellow, "\u001b[33m") \
C(Blue, "\u001b[34m") \
C(Magenta, "\u001b[35m") \
C(Cyan, "\u001b[36m") \
C(White, "\u001b[37m") \
C(Reset, "\u001b[0m")  \

struct ScopedWorld : public Streamable<ScopedWorld> {
    struct Config {
        bool use_color;
    };

    ScopedWorld(World& w, Config cfg = { getenv("THORIN_NO_COLOR") ? false : true }) : world_(w), forest_(w), config_(cfg) {
#define T(n, c) n = cfg.use_color ? c : "";
        COLORS(T)
#undef T
    }

    World& world_;
    mutable ScopesForest forest_;

    mutable DefSet done_;
    mutable ContinuationMap<std::unique_ptr<std::vector<const Def*>>> scopes_to_defs_;
    mutable std::vector<const Def*> top_lvl_;
    Config config_;

#define T(n, c) const char* n;
    COLORS(T)
#undef T

    Stream& stream(Stream&) const;
private:

    void stream_cont(thorin::Stream& s, Continuation* cont) const;
    void prepare_def(Continuation* in, const Def* def) const;
    void stream_op(thorin::Stream&, const Def* op) const;
    void stream_ops(thorin::Stream& s, Defs defs) const;
    void stream_def(thorin::Stream& s, const Def* def) const;
    void stream_defs(thorin::Stream& s, std::vector<const Def*>& defs) const;
};

}
