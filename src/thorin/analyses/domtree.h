#ifndef THORIN_ANALYSES_DOMTREE_H
#define THORIN_ANALYSES_DOMTREE_H

#include "thorin/lambda.h"
#include "thorin/util/array.h"

namespace thorin {

template<bool> class DomTreeBase;
class Def;
class Lambda;
template<bool> class Scope;
class World;

class DomNode {
public:
    explicit DomNode(Lambda* lambda) 
        : lambda_(lambda) 
        , idom_(0)
    {}

    Lambda* lambda() const { return lambda_; }
    const DomNode* idom() const { return idom_; }
    const std::vector<const DomNode*>& children() const { return children_; }
    bool entry() const { return idom_ == this; }
    int depth() const;

private:
    Lambda* lambda_;
    DomNode* idom_;
    std::vector<const DomNode*> children_;

    //template<bool forwards> 
    //friend class DomTreeBase<forwards>;
};

template<bool forwards>
class DomTreeBase {
public:
    explicit DomTreeBase(const Scope<forwards>& scope)
        : scope_(scope)
    {
        create();
    }

    int depth(Lambda* lambda) const { return lookup(lambda)->depth(); }
    /// Returns the least common ancestor of \p i and \p j.
    Lambda* lca(Lambda* i, Lambda* j) const { return lca(lookup(i), lookup(j))->lambda(); }
    const DomNode* lca(const DomNode* i, const DomNode* j) const { 
        return const_cast<DomTreeBase*>(this)->lca(const_cast<DomNode*>(i), const_cast<DomNode*>(j)); 
    }
    Lambda* idom(Lambda* lambda) const { return Super::lookup(lambda)->idom()->lambda(); }
    const DomNode* lookup(Lambda* lambda) const { return map_[lambda]; }

private:
    DomNode* lookup(Lambda* lambda) { return map_[lambda]; }
    void create();
    DomNode* lca(DomNode* i, DomNode* j);

    typename Scope<forwards> scope_;
    LambdaMap<const DomNode*> map_;
};

typedef DomTreeBase< true>      DomTree;
typedef DomTreeBase<false>  PostDomTree;

} // namespace thorin

#endif
