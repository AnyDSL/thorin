#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"

namespace thorin {

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

    /// All lambdas within this scope in post-order.
    ArrayRef<Lambda*>  po() const { return  po_; }
    /// All lambdas within this scope in reverse post-order.
    ArrayRef<Lambda*> rpo() const { return rpo_; }
    Lambda*  po(size_t i) const { return  po_[i]; }
    Lambda* rpo(size_t i) const { return rpo_[i]; }
    Lambda* entry() const { return rpo().front(); }
    Lambda* exit()  const { return po_.front(); }
    /// Like \p rpo() but without \p entry()
    ArrayRef<Lambda*> body() const { return rpo().slice_from_begin(1); }
    const DefSet& in_scope() const { return in_scope_; }
    bool contains(Def def) const { return in_scope_.contains(def); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return preds_[rpo_index(lambda)]; }
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return succs_[rpo_index(lambda)]; }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    size_t po_index(Lambda* lambda) const { return lambda->find(this)->po_index; }
    size_t rpo_index(Lambda* lambda) const { return lambda->find(this)->rpo_index; }
    uint32_t sid() const { return sid_; }
    size_t size() const { return rpo_.size(); }
    World& world() const { return world_; }

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
    void build_in_scope();
    void link(Lambda* src, Lambda* dst) {
        assert(is_candidate(src) && is_candidate(dst));
        succs_[rpo_index(src)].push_back(dst);
        preds_[rpo_index(dst)].push_back(src);
    }

    World& world_;
    DefSet in_scope_;
    uint32_t sid_;
    std::vector<Lambda*> rpo_;
    std::vector<Lambda*> po_;
    std::vector<std::vector<Lambda*>> preds_;
    std::vector<std::vector<Lambda*>> succs_;

    static uint32_t candidate_counter_;
    static uint32_t sid_counter_;

    friend class ScopeView;
};

//------------------------------------------------------------------------------

class ScopeView {
public:
    explicit ScopeView(const Scope& scope, const bool is_forward = true)
        : scope_(scope)
        , is_forward_(is_forward)
    {}

    const Scope& scope() const { return scope_; }
    bool is_forward() const { return is_forward_; }
    /// All lambdas within this scope in reverse post-order.
    ArrayRef<Lambda*> rpo() const { return is_forward() ? scope().rpo_ : scope().po_; }
    Lambda* entry() const { return rpo().front(); }
    Lambda* exit()  const { return (is_forward() ? scope().po_ : scope().rpo_).front(); }
    /// Like \p rpo() but without \p entry()
    ArrayRef<Lambda*> body() const { return rpo().slice_from_begin(1); }
    const DefSet& in_scope() const { return scope().in_scope_; }
    bool contains(Def def) const { return scope().in_scope_.contains(def); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return is_forward() ? scope().preds(lambda) : scope().succs(lambda); }
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return is_forward() ? scope().succs(lambda) : scope().preds(lambda); }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    size_t  po_index(Lambda* lambda) const { return is_forward() ? scope(). po_index(lambda) : scope().rpo_index(lambda); }
    size_t rpo_index(Lambda* lambda) const { return is_forward() ? scope().rpo_index(lambda) : scope(). po_index(lambda); }
    size_t size() const { return scope().size(); }
    World& world() const { return scope().world(); }

    typedef ArrayRef<Lambda*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

    typedef ArrayRef<Lambda*>::const_reverse_iterator const_reverse_iterator;
    const_reverse_iterator rbegin() const { return rpo().rbegin(); }
    const_reverse_iterator rend() const { return rpo().rend(); }

private:
    const Scope& scope_;
    const bool is_forward_;

    friend class Scope;
};

//------------------------------------------------------------------------------

}

#endif
