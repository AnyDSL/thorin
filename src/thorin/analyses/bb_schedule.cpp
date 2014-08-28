#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"

namespace thorin {

static int count_children(const DomNode* n, LambdaMap<int>& lambda2num) {
    int num = 0;
    for (auto child : n->children())
        num += count_children(child, lambda2num);
    return (lambda2num[n->lambda()] = num) + 1;
}

static void bb_schedule(const DomNode* n, const LoopTree& looptree, std::vector<Lambda*>& bbs, const LambdaMap<int>& lambda2num) {
    bbs.push_back(n->lambda());
    auto children = n->children();
    std::sort(children.begin(), children.end(), [&] (const DomNode* n1, const DomNode* n2) {
        auto l1 = n1->lambda();
        auto l2 = n2->lambda();
        auto d1 = looptree.depth(l1);
        auto d2 = looptree.depth(l2);
        if (d1 > d2) return true;
        if (d1 < d2) return false;
        auto i1 = lambda2num.find(l1)->second;
        auto i2 = lambda2num.find(l2)->second;
        if (i1 > i2) return true;
        if (i1 < i2) return false;
        return l1->gid() < l2->gid();
    });

    for (auto child : children)
        bb_schedule(child, looptree, bbs, lambda2num);
}

std::vector<Lambda*> bb_schedule(const Scope& scope) {
    const DomTree domtree(scope);
    const LoopTree looptree(scope);
    LambdaMap<int> lambda2num;
    count_children(domtree.root(), lambda2num);
    std::vector<Lambda*> bbs;
    bb_schedule(domtree.root(), looptree, bbs, lambda2num);
    assert(bbs.size() == scope.size());
    return bbs;
}

}
