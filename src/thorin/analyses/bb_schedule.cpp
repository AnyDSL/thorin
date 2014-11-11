#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"

namespace thorin {


static int count_children(const DomNode* n, LambdaMap<int>& lambda2num) {
    int num = 0;
    for (auto child : n->children())
        num += count_children(child, lambda2num);
    return (lambda2num[n->lambda()] = num) + 1;
}

static void bb_schedule(const F_CFG& cfg, const DomNode* n, const LoopTree& looptree, 
        std::vector<Lambda*>& bbs, const LambdaMap<int>& lambda2num) {
    auto lambda = n->lambda();
    bbs.push_back(lambda);
    auto children = n->children();
    std::sort(children.begin(), children.end(), [&] (const DomNode* n1, const DomNode* n2) {
        auto l1 = n1->lambda();
        auto l2 = n2->lambda();

        // handle loops first
        auto depth1 = looptree.cfg_node2leaf(cfg.lookup(l1))->depth();
        auto depth2 = looptree.cfg_node2leaf(cfg.lookup(l2))->depth();
        if (depth1 > depth2) return true;
        if (depth1 < depth2) return false;

        // if this fails - use the one with more children in the domtree
        auto num1 = lambda2num.find(l1)->second;
        auto num2 = lambda2num.find(l2)->second;
        if (num1 > num2) return true;
        if (num1 < num2) return false;

        // if this fails use the one which is a direct succ of lambda
        const auto& succs = cfg.succs(cfg.lookup(lambda));
        auto is_succ1 = std::find(succs.begin(), succs.end(), cfg.lookup(l1)) != succs.end();
        auto is_succ2 = std::find(succs.begin(), succs.end(), cfg.lookup(l2)) != succs.end();
        if ( is_succ1 && !is_succ2) return true;
        if (!is_succ1 &&  is_succ2) return false;

        // if this still fails - simply use smaller gid
        return l1->gid() < l2->gid();
    });

    for (auto child : children)
        bb_schedule(cfg, child, looptree, bbs, lambda2num);
}

std::vector<Lambda*> bb_schedule(const Scope& scope) {
    auto domtree = scope.cfg()->domtree();
    LambdaMap<int> lambda2num;
    count_children(domtree->root(), lambda2num);
    std::vector<Lambda*> bbs;
    bb_schedule(*scope.cfg()->f_cfg(), domtree->root(), *scope.cfg()->looptree(), bbs, lambda2num);
    assert(bbs.size() == scope.cfg()->f_cfg()->size());
    return bbs;
}

}
