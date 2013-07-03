#ifndef ANYDSL2_ANALYSES_SCOPE_ANALYSIS_H
#define ANYDSL2_ANALYSES_SCOPE_ANALYSIS_H

#include "anydsl2/analyses/scope.h"
#include "anydsl2/util/autoptr.h"

namespace anydsl2 {

template<class T, bool forwards>
class ScopeAnalysis {
protected:

    ScopeAnalysis(const Scope& scope)
        : scope_(scope)
        , nodes_(size())
    {}

public:

    const Scope& scope() const { return scope_; }
    size_t size() const { return scope().size();}
    ArrayRef<const T*> nodes() const { return ArrayRef<const T*>(nodes_.begin(), nodes_.size()); }
    const T* node(Lambda* lambda) const { assert(scope().contains(lambda)); return nodes_[index(lambda)]; }
    size_t index(T* n) const { return index(n->lambda()); }

    /// Returns \p lambda's scope id if this is a forwards analysis and lambda%'s \p backwards_sid() in the case of a backwards analysis.
    size_t index(Lambda* lambda) const { return forwards ? lambda->sid() : lambda->backwards_sid(); }

    /// Returns \p lambda's rpo if this is a forwards analysis and lambda%'s \p backwards_rpo() in the case of a backwards analysis.
    ArrayRef<Lambda*> rpo() const { return forwards ? scope().rpo() : scope().backwards_rpo(); }

    /// Returns \p lambda's entries() if if this is a forwards analysis and lambda%'s \p exits() in the case of a backwards analysis.
    ArrayRef<Lambda*> entries() const { return forwards ? scope().entries() : scope().exits(); }

    /// Returns \p lambda's body() if if this is a forwards analysis and lambda%'s \p backwards_body() in the case of a backwards analysis.
    ArrayRef<Lambda*> body() const { return forwards ? scope().body() : scope().backwards_body(); }

    /// Returns \p lambda's preds() if if this is a forwards analysis and lambda%'s \p succs() in the case of a backwards analysis.
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return forwards ? scope().preds(lambda) : scope().succs(lambda); }

    /// Returns \p lambda's succs() in this scope if if this is a forwards analysis and lambda%'s \p preds() in this scope in the case of a backwards analysis.
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return forwards ? scope().succs(lambda) : scope().preds(lambda); }

    /// Are both \p i and \p j entries (exits) of this \p Scope in the case of a forwards analysis (backwards analysis).
    bool is_entry(const T* i, const T* j) const { return forwards 
        ? (scope().is_entry(i->lambda()) && scope().is_entry(j->lambda()))
        : (scope().is_exit (i->lambda()) && scope().is_exit (j->lambda())); }

    T* lookup(Lambda* lambda) { assert(scope().contains(lambda)); return nodes_[index(lambda)]; }
    const T* lookup(Lambda* lambda) const { return const_cast<ScopeAnalysis*>(this)->lookup(lambda); }

protected:

    const Scope& scope_;
    AutoVector<T*> nodes_;
};

} // namespace analyses

#endif // ANALYSES_SCOPE_ANALYSIS_H
