#include "anydsl2/analyses/domtree.h"

#include <limits>
#include <queue>

#include "anydsl2/lambda.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

DomNode::DomNode(Lambda* lambda) 
    : lambda_(lambda) 
    , idom_(0)
{}

int DomNode::depth() const {
    int result = 0;

    for (const DomNode* i = this; !i->entry(); i = i->idom())
        ++result;

    return result;
};

//------------------------------------------------------------------------------

DomTree::DomTree(const Scope& scope, bool forwards)
    : scope_(scope)
    , nodes_(size())
    , forwards_(forwards)
{
    create();
}

DomTree::~DomTree() {
    for_all (node, nodes_)
        delete node;
}

void DomTree::create() {
    for_all (lambda, rpo())
        nodes_[index(lambda)] = new DomNode(lambda);

    for (size_t i = 0; i < size(); ++i)
        assert(i == index(nodes_[i]));

    // map entries' initial idoms to themselves
    for_all (entry,  entries()) {
        DomNode* entry_node = lookup(entry);
        entry_node->idom_ = entry_node;
    }

    // all others' idoms are set to their first found dominating pred
    for_all (lambda, body()) {
        for_all (pred, preds(lambda)) {
            if (index(pred) < index(lambda)) {
                lookup(lambda)->idom_ = lookup(pred);
                goto outer_loop;
            }
        }
        ANYDSL2_UNREACHABLE;
outer_loop:;
    }

    if (is_postdomtree()) {
        for_all (lambda, rpo())
            std::cout << lambda->unique_name() << std::endl;

        std::cout << "---" << std::endl;
        std::cout << entries().size() << std::endl;
        std::cout << "---" << std::endl;

        for_all (lambda, body())
            std::cout << lambda->unique_name() << std::endl;

        for_all (lambda, rpo())
            std::cout << index(lambda) << ": " << lambda->unique_name() << " -> " << lookup(lambda)->idom()->lambda()->unique_name() << std::endl;
    }

    std::cout << "---" << std::endl;
    std::cout << "---" << std::endl;
    std::cout << "---" << std::endl;
    std::cout << "---" << std::endl;

    for (bool changed = true; changed;) {
        changed = false;

        for_all (lambda, body()) {
            DomNode* lambda_node = lookup(lambda);

            std::cout << "\ncur: " << lambda->unique_name() << " -> " << lookup(lambda)->idom()->lambda()->unique_name() << std::endl;

            DomNode* new_idom = 0;
            for_all (pred, preds(lambda)) {
                std::cout << "succ: " << pred->unique_name() << std::endl;
                DomNode* pred_node = lookup(pred);
                assert(pred_node);
                new_idom = new_idom ? lca(new_idom, pred_node) : pred_node;
            }
            assert(new_idom);
            std::cout << "new_idom: " << new_idom->lambda()->unique_name() << std::endl;
            if (lambda_node->idom() != new_idom) {
                lambda_node->idom_ = new_idom;
                changed = true;
            }
        }
    }

    for_all (lambda, body()) {
        const DomNode* n = lookup(lambda);
        n->idom_->children_.push_back(n);
    }
}

DomNode* DomTree::lca(DomNode* i, DomNode* j) {
    while (!is_entry(i, j) && index(i) != index(j)) {
        while (!is_entry(i, j) && index(i) < index(j)) 
            j = j->idom_;
        while (!is_entry(i, j) && index(j) < index(i)) 
            i = i->idom_;
    }

    return i;
}

size_t DomTree::size() const { return scope_.size(); }

} // namespace anydsl2
