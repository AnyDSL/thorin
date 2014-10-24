#include "thorin/primop.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"

/*
 * see "Fast liveness checking for ssa-form programs"
 * http://dl.acm.org/citation.cfm?id=1356064
 */

namespace thorin {

class Liveness {
private:
    enum class Color : uint8_t { White, Gray, Black };

public:
    Liveness(const Schedule& schedule);

    const Schedule& schedule() const { return schedule_; }
    const Scope& scope() const { return scope_; }
    size_t size() const { return scope_.size(); }
    const DomTree& domtree() const { return domtree_; }
    size_t rpo_id(Lambda* lambda) const { return scope_.rpo_id(lambda); }
    void reduced_link(Lambda* src, Lambda* dst) {
        if (src) {
            reduced_succs_[rpo_id(src)].push_back(dst);
            reduced_preds_[rpo_id(dst)].push_back(src);
        }
    }
    void reduced_visit(std::vector<Color>& colors, Lambda* prev, Lambda* cur);
    bool is_live_in(Def def, Lambda* lambda);

private:
    const Schedule& schedule_;
    const Scope& scope_;
    const DomTree& domtree_;
    DefMap<Lambda*> def2lambda_;
    std::vector<std::vector<Lambda*>> reduced_preds_;
    std::vector<std::vector<Lambda*>> reduced_succs_;
    //std::vector<
};

Liveness::Liveness(const Schedule& schedule)
    : schedule_(schedule)
    , scope_(schedule.scope())
    , domtree_(*scope().domtree())
    , reduced_preds_(size())
    , reduced_succs_(size())
{
    // compute for each definition its defining lambda block
    for (auto lambda : scope()) {
        for (auto param : lambda->params())
            def2lambda_[param] = lambda;
        for (auto primop : schedule[lambda])
            def2lambda_[primop] = lambda;
    }

    // compute reduced CFG
    std::vector<Color> colors(size(), Color::White);
    reduced_visit(colors, nullptr, scope().entry());

    // compute reduced reachable set
    for (auto lambda : scope()) {
    }
}

void Liveness::reduced_visit(std::vector<Color>& colors, Lambda* prev, Lambda* cur) {
    auto& col = colors[rpo_id(cur)];
    switch (col) {
        case Color::White:              // white: not yet visited
            col = Color::Gray;          // mark gray: is on recursion stack
            for (auto succ : scope_.succs(cur))
                reduced_visit(colors, cur, succ);
            col = Color::Black;         // mark black: done
            break;                      // link
        case Color::Gray: return;       // back edge: do nothing
        case Color::Black: break;       // cross or forward edge: link
    }
    reduced_link(prev, cur);
}

bool Liveness::is_live_in(Def def, Lambda* lambda) {
    size_t d_rpo = rpo_id(def2lambda_[def]);
    size_t l_rpo = rpo_id(lambda);
    //if (l_rpo < 
}

}
