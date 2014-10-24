#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"

/*
 * see "Fast liveness checking for ssa-form programs"
 * http://dl.acm.org/citation.cfm?id=1356064
 */

namespace thorin {

class Liveness {
public:
    Liveness(const Schedule& schedule)
        : schedule_(schedule)
        , scope_(schedule.scope())
        , domtree_(*scope_.domtree())
        , reduced_preds_(scope_.size())
        , reduced_succs_(scope_.size())
    {}

    void compute_reduced_cfg(Lambda* prev, Lambda* cur);

    const Schedule& schedule() const { return schedule_; }
    const Scope& scope() const { return scope_; }
    const DomTree& domtree() const { return domtree_; }
    size_t rpo_id(Lambda* lambda) const { return scope_.rpo_id(lambda); }
    void reduced_link(Lambda* src, Lambda* dst) {
        if (src) {
            reduced_succs_[rpo_id(src)].push_back(dst);
            reduced_preds_[rpo_id(dst)].push_back(src);
        }
    }

private:
    class ReducedCFGBuilder {
    public:
        enum class Color : uint8_t { White, Gray, Black };

        ReducedCFGBuilder(Liveness& liveness)
            : liveness_(liveness)
            , colors_(liveness.scope().size(), Color::White)
        {
            visit(nullptr, liveness_.scope().entry());
        }

        void visit(Lambda* prev, Lambda* cur);
        Color& color(Lambda* lambda) { return colors_[liveness_.rpo_id(lambda)]; }

    private:
        Liveness& liveness_;
        std::vector<Color> colors_;
    };

    const Schedule& schedule_;
    const Scope& scope_;
    const DomTree& domtree_;
    std::vector<std::vector<Lambda*>> reduced_preds_;
    std::vector<std::vector<Lambda*>> reduced_succs_;
};

void Liveness::ReducedCFGBuilder::visit(Lambda* prev, Lambda* cur) {
    auto& col = color(cur);
    switch (col) {
        case Color::White:              // white: not yet visited
            col = Color::Gray;          // mark gray: is on recursion stack
            for (auto succ : liveness_.scope_.succs(cur))
                visit(cur, succ);
            col = Color::Black;         // mark black: done
            break;
        case Color::Gray: return;       // back edge: do nothing
        case Color::Black: break;       // cross or forward edge: link
    }
    liveness_.reduced_link(prev, cur);
}

}
