#ifndef ANYDSL2_ANALYSES_DOMTREE_H
#define ANYDSL2_ANALYSES_DOMTREE_H

#include <boost/unordered_set.hpp>

#include "anydsl2/lambda.h"
#include "anydsl2/util/array.h"
#include "anydsl2/analyses/scope_analysis.h"

namespace anydsl2 {

template<bool> class DomNodeBase;
template<bool> class DomTreeBase;
class Def;
class Lambda;
class Scope;
class World;

template<bool forwards>
class DomNodeBase {
public:

    explicit DomNodeBase(Lambda* lambda) 
        : lambda_(lambda) 
        , idom_(0)
    {}

    Lambda* lambda() const { return lambda_; }
    const DomNodeBase* idom() const { return idom_; }
    const std::vector<const DomNodeBase*>& children() const { return children_; }
    bool entry() const { return idom_ == this; }
    int depth() const;

private:

    Lambda* lambda_;
    DomNodeBase* idom_;
    std::vector<const DomNodeBase*> children_;

    friend class DomTreeBase<forwards>;
};

template<bool forwards>
class DomTreeBase : public ScopeAnalysis<DomNodeBase<forwards>, forwards> {
public:

    typedef DomNodeBase<forwards> DomNode;
    typedef ScopeAnalysis<DomNodeBase<forwards>, forwards> Super;

    explicit DomTreeBase(const Scope& scope)
        : Super(scope)
    {
        create();
    }

    int depth(Lambda* lambda) const { return Super::node(lambda)->depth(); }
    /// Returns the least common ancestor of \p i and \p j.
    Lambda* lca(Lambda* i, Lambda* j) const { return lca(Super::lookup(i), Super::lookup(j))->lambda(); }
    const DomNode* lca(const DomNode* i, const DomNode* j) const { 
        return const_cast<DomTreeBase*>(this)->lca(const_cast<DomNode*>(i), const_cast<DomNode*>(j)); 
    }
    Lambda* idom(Lambda* lambda) const { return Super::lookup(lambda)->idom()->lambda(); }

private:

    void create();
    DomNode* lca(DomNode* i, DomNode* j);
};

typedef DomNodeBase< true>      DomNode;
typedef DomNodeBase<false>  PostDomNode;
typedef DomTreeBase< true>      DomTree;
typedef DomTreeBase<false>  PostDomTree;

} // namespace anydsl2

#endif
