#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"

namespace thorin {

#if 0

static int count_children(const DomNode* n, LambdaMap<int>& lambda2num) {
    int num = 0;
    for (auto child : n->children())
        num += count_children(child, lambda2num);
    return (lambda2num[n->lambda()] = num) + 1;
}

static void bb_schedule(const Scope& scope, const DomNode* n, std::vector<Lambda*>& bbs, const LambdaMap<int>& lambda2num) {
    auto looptree = scope.looptree();
    auto lambda = n->lambda();
    bbs.push_back(lambda);
    auto children = n->children();
    std::sort(children.begin(), children.end(), [&] (const DomNode* n1, const DomNode* n2) {
        auto l1 = n1->lambda();
        auto l2 = n2->lambda();

        // handle loops first
        auto depth1 = looptree->depth(l1);
        auto depth2 = looptree->depth(l2);
        if (depth1 > depth2) return true;
        if (depth1 < depth2) return false;

        // if this fails - use the one with more children in the domtree
        auto num1 = lambda2num.find(l1)->second;
        auto num2 = lambda2num.find(l2)->second;
        if (num1 > num2) return true;
        if (num1 < num2) return false;

        // if this fails use the one which is a direct succ of lambda
        const auto& succs = scope.succs(lambda);
        auto is_succ1 = std::find(succs.begin(), succs.end(), l1) != succs.end();
        auto is_succ2 = std::find(succs.begin(), succs.end(), l2) != succs.end();
        if ( is_succ1 && !is_succ2) return true;
        if (!is_succ1 &&  is_succ2) return false;

        // if this still fails - simply use smaller gid
        return l1->gid() < l2->gid();
    });

    for (auto child : children)
        bb_schedule(scope, child, bbs, lambda2num);
}

std::vector<Lambda*> bb_schedule(const Scope& scope) {
    auto domtree = scope.domtree();
    LambdaMap<int> lambda2num;
    count_children(domtree->root(), lambda2num);
    std::vector<Lambda*> bbs;
    bb_schedule(scope, domtree->root(), bbs, lambda2num);
    assert(bbs.size() == scope.size());
    return bbs;
}
#endif

}
