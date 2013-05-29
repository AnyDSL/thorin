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

    explicit DomTreeBase(const Scope& scope)
        : scope_(scope)
        , nodes_(size())
    {
        create();
    }

    ~DomTreeBase();

    const Scope& scope() const { return scope_; }
    ArrayRef<const DomNodeBase<forwards>*> nodes() const { return ArrayRef<const DomNodeBase<forwards>*>(nodes_.begin(), nodes_.size()); }
    size_t size() const;
    const DomNodeBase<forwards>* node(Lambda* lambda) const { assert(scope().contains(lambda)); return nodes_[index(lambda)]; }
    int depth(Lambda* lambda) const { return node(lambda)->depth(); }
    /// Returns the least common ancestor of \p i and \p j.
    Lambda* lca(Lambda* i, Lambda* j) const { return lca(lookup(i), lookup(j))->lambda(); }
    const DomNodeBase<forwards>* lca(const DomNodeBase<forwards>* i, const DomNodeBase<forwards>* j) const { 
        return const_cast<DomTreeBase*>(this)->lca(const_cast<DomNodeBase<forwards>*>(i), const_cast<DomNodeBase<forwards>*>(j)); 
    }
    Lambda* idom(Lambda* lambda) const { return lookup(lambda)->idom()->lambda(); }
    size_t index(DomNodeBase<forwards>* n) const { return index(n->lambda()); }
    /// Returns \p lambda%'s \p backwards_sid() in the case this a postdomtree 
    /// or \p lambda%'s sid() if this is an ordinary domtree.
    size_t index(Lambda* lambda) const { return forwards ? lambda->sid() : lambda->backwards_sid(); }
    ArrayRef<Lambda*> rpo() const { return forwards ? scope_.rpo() : scope_.backwards_rpo(); }
    ArrayRef<Lambda*> entries() const { return forwards ? scope_.entries() : scope_.exits(); }
    ArrayRef<Lambda*> body() const { return forwards ? scope_.body() : scope_.backwards_body(); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return forwards ? scope_.preds(lambda) : scope_.succs(lambda); }
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return forwards ? scope_.succs(lambda) : scope_.preds(lambda); }
    bool is_entry(DomNodeBase<forwards>* i, DomNodeBase<forwards>* j) const { return forwards 
        ? (scope_.is_entry(i->lambda()) && scope_.is_entry(j->lambda()))
        : (scope_.is_exit (i->lambda()) && scope_.is_exit (j->lambda())); }

private:

    void create();
    DomNodeBase<forwards>* lca(DomNodeBase<forwards>* i, DomNodeBase<forwards>* j);
    DomNodeBase<forwards>* lookup(Lambda* lambda) { assert(scope().contains(lambda)); return nodes_[index(lambda)]; }
    const DomNodeBase<forwards>* lookup(Lambda* lambda) const { return const_cast<DomTreeBase*>(this)->lookup(lambda); }

    const Scope& scope_;
    Array<DomNodeBase<forwards>*> nodes_;
};

typedef DomNodeBase< true>      DomNode;
typedef DomNodeBase<false>  PostDomNode;
typedef DomTreeBase< true>      DomTree;
typedef DomTreeBase<false>  PostDomTree;

} // namespace anydsl2

#endif
