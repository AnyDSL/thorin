#ifndef ANYDSL2_ANALYSES_DOMTREE_H
#define ANYDSL2_ANALYSES_DOMTREE_H

#include <boost/unordered_set.hpp>

#include "anydsl2/lambda.h"
#include "anydsl2/util/array.h"
#include "anydsl2/analyses/scope.h"

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
class DomTreeBase {
public:

    typedef DomNodeBase<forwards> DomNode;

    explicit DomTreeBase(const Scope& scope)
        : scope_(scope)
        , nodes_(size())
    {
        create();
    }

    ~DomTreeBase();

    const Scope& scope() const { return scope_; }
    ArrayRef<const DomNode*> nodes() const { return ArrayRef<const DomNode*>(nodes_.begin(), nodes_.size()); }
    size_t size() const;
    const DomNode* node(Lambda* lambda) const { assert(scope().contains(lambda)); return nodes_[index(lambda)]; }
    int depth(Lambda* lambda) const { return node(lambda)->depth(); }
    /// Returns the least common ancestor of \p i and \p j.
    Lambda* lca(Lambda* i, Lambda* j) const { return lca(lookup(i), lookup(j))->lambda(); }
    const DomNode* lca(const DomNode* i, const DomNode* j) const { 
        return const_cast<DomTreeBase*>(this)->lca(const_cast<DomNode*>(i), const_cast<DomNode*>(j)); 
    }
    Lambda* idom(Lambda* lambda) const { return lookup(lambda)->idom()->lambda(); }
    size_t index(DomNode* n) const { return index(n->lambda()); }
    /// Returns \p lambda%'s \p backwards_sid() in the case this a postdomtree 
    /// or \p lambda%'s sid() if this is an ordinary domtree.
    size_t index(Lambda* lambda) const { return forwards ? lambda->sid() : lambda->backwards_sid(); }
    ArrayRef<Lambda*> rpo() const { return forwards ? scope_.rpo() : scope_.backwards_rpo(); }
    ArrayRef<Lambda*> entries() const { return forwards ? scope_.entries() : scope_.exits(); }
    ArrayRef<Lambda*> body() const { return forwards ? scope_.body() : scope_.backwards_body(); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return forwards ? scope_.preds(lambda) : scope_.succs(lambda); }
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return forwards ? scope_.succs(lambda) : scope_.preds(lambda); }
    bool is_entry(DomNode* i, DomNode* j) const { return forwards 
        ? (scope_.is_entry(i->lambda()) && scope_.is_entry(j->lambda()))
        : (scope_.is_exit (i->lambda()) && scope_.is_exit (j->lambda())); }

private:

    void create();
    DomNode* lca(DomNode* i, DomNode* j);
    DomNode* lookup(Lambda* lambda) { assert(scope().contains(lambda)); return nodes_[index(lambda)]; }
    const DomNode* lookup(Lambda* lambda) const { return const_cast<DomTreeBase*>(this)->lookup(lambda); }

    const Scope& scope_;
    Array<DomNode*> nodes_;
};

typedef DomNodeBase< true>      DomNode;
typedef DomNodeBase<false>  PostDomNode;
typedef DomTreeBase< true>      DomTree;
typedef DomTreeBase<false>  PostDomTree;

} // namespace anydsl2

#endif
