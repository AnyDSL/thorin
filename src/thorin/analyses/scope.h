#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"

namespace thorin {

template<bool> class DomTreeBase;
typedef DomTreeBase<true>  DomTree;
typedef DomTreeBase<false> PostDomTree;

class LoopTree;

//------------------------------------------------------------------------------

class Scope {
public:
    Scope(const Scope&) = delete;
    Scope& operator= (Scope) = delete;

    /// Builds a unique meta \p Lambda as dummy entry if necessary.
    explicit Scope(World& world, ArrayRef<Lambda*> entries);
    /// Does not build a meta \p Lambda
    explicit Scope(Lambda* entry)
        : Scope(entry->world(), {entry})
    {}
    ~Scope();

    /// All lambdas within this scope in reverse post-order.
    ArrayRef<Lambda*> rpo() const { return rpo_; }
    /// All lambdas within this scope in reverse rpo (this is \em no post-order).
    ArrayRef<Lambda*> rev_rpo() const { return rev_rpo_; }
    Lambda* rev_rpo(size_t i) const { return rev_rpo_[i]; }
    Lambda* rpo(size_t i) const { return rpo_[i]; }
    Lambda* entry() const { return rpo().front(); }
    Lambda* exit()  const { return rev_rpo_.front(); }
    /// Like \p rpo() but without \p entry()
    ArrayRef<Lambda*> body() const { return rpo().slice_from_begin(1); }
    const DefSet& in_scope() const { return in_scope_; }
    bool contains(Def def) const { return in_scope_.contains(def); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return preds_[rpo_id(lambda)]; }
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return succs_[rpo_id(lambda)]; }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    size_t rev_rpo_id(Lambda* lambda) const { return lambda->find(this)->rev_rpo_id; }
    size_t rpo_id(Lambda* lambda) const { return lambda->find(this)->rpo_id; }
    uint32_t sid() const { return sid_; }
    size_t size() const { return rpo_.size(); }
    World& world() const { return world_; }

    const DomTree* domtree() const;
    const PostDomTree* postdomtree() const;
    const LoopTree* looptree() const;

    typedef ArrayRef<Lambda*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

    typedef ArrayRef<Lambda*>::const_reverse_iterator const_reverse_iterator;
    const_reverse_iterator rbegin() const { return rpo().rbegin(); }
    const_reverse_iterator rend() const { return rpo().rend(); }

private:
    static bool is_candidate(Def def) { return def->candidate_ == candidate_counter_; }
    static void set_candidate(Def def) { def->candidate_ = candidate_counter_; }
    static void unset_candidate(Def def) { assert(is_candidate(def)); --def->candidate_; }

    void identify_scope(ArrayRef<Lambda*> entries);
    void number(ArrayRef<Lambda*> entries);
    size_t number(Lambda* lambda, size_t i);
    void build_cfg(ArrayRef<Lambda*> entries);
    std::vector<Lambda*> find_exits(Array<Lambda*> entries);
    void rev_number(ArrayRef<Lambda*> exits);
    size_t rev_number(Lambda* lambda, size_t i);

    void build_in_scope();
    void link(Lambda* src, Lambda* dst) {
        assert(is_candidate(src) && is_candidate(dst));
        succs_[rpo_id(src)].push_back(dst);
        preds_[rpo_id(dst)].push_back(src);
    }

    template<class T> T* lazy(T*& ptr) const { return ptr ? ptr : ptr = new T(*this); }

    World& world_;
    DefSet in_scope_;
    uint32_t sid_;
    std::vector<Lambda*> rpo_;
    std::vector<Lambda*>rev_rpo_;
    std::vector<std::vector<Lambda*>> preds_;
    std::vector<std::vector<Lambda*>> succs_;
    mutable const DomTree* domtree_ = 0;
    mutable const PostDomTree* postdomtree_ = 0;
    mutable const LoopTree* looptree_ = 0;

    static uint32_t candidate_counter_;
    static uint32_t sid_counter_;

    template<bool> friend class ScopeView;
};

//------------------------------------------------------------------------------

template<bool forward = true>
class ScopeView {
public:
    explicit ScopeView(const Scope& scope)
        : scope_(scope)
    {}

    const Scope& scope() const { return scope_; }
    /// All lambdas within this scope in reverserev_rpost-order.
    ArrayRef<Lambda*> rpo() const { return forward ? scope().rpo_ : scope().rev_rpo_; }
    Lambda* entry() const { return rpo().front(); }
    Lambda* exit()  const { return (forward ? scope().rev_rpo_ : scope().rpo_).front(); }
    /// Like \p rpo() but without \p entry()
    ArrayRef<Lambda*> body() const { return rpo().slice_from_begin(1); }
    const DefSet& in_scope() const { return scope().in_scope_; }
    bool contains(Def def) const { return scope().in_scope_.contains(def); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return forward ? scope().preds(lambda) : scope().succs(lambda); }
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return forward ? scope().succs(lambda) : scope().preds(lambda); }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    size_t rpo_id(Lambda* lambda) const { return forward ? scope().rpo_id(lambda) : scope().rev_rpo_id(lambda); }
    size_t rev_rpo_id(Lambda* lambda) const { return forward ? scope().rev_rpo_id(lambda) : scope().rpo_id(lambda); }
    size_t size() const { return scope().size(); }
    World& world() const { return scope().world(); }

    typedef ArrayRef<Lambda*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

private:
    const Scope& scope_;

    friend class Scope;
};

//------------------------------------------------------------------------------

template<bool elide_empty = true>
void top_level_scopes(World&, std::function<void(const Scope&)>);

//------------------------------------------------------------------------------

}

#endif
