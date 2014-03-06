#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#define private public

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

    /// Always builds a unique meta \p Lambda as entry.
    explicit Scope(World& world, ArrayRef<Lambda*> entries);
    /// Does not build a meta \p Lambda
    explicit Scope(Lambda* entry)
        : Scope(entry->world(), {entry})
    {}
    ~Scope();

    /// All lambdas within this scope in reverse postorder.
    ArrayRef<Lambda*> rpo() const { return rpo_; }
    Lambda* entry() const { return rpo().front(); }
    Lambda* exit()  const { return reverse_rpo_.front(); }
    /// Like \p rpo() but without \p entry()
    ArrayRef<Lambda*> body() const { return rpo().slice_from_begin(1); }
    const DefSet& in_scope() const { return in_scope_; }
    bool contains(Def def) const { return in_scope_.contains(def); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return preds_[lambda]; }
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return succs_[lambda]; }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    int sid(Lambda* lambda) const { assert(contains(lambda)); return sid_[lambda]; }
    size_t size() const { return rpo_.size(); }
    World& world() const { return world_; }

    typedef ArrayRef<Lambda*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

    typedef ArrayRef<Lambda*>::const_reverse_iterator const_reverse_iterator;
    const_reverse_iterator rbegin() const { return rpo().rbegin(); }
    const_reverse_iterator rend() const { return rpo().rend(); }

private:
    void identify_scope(ArrayRef<Lambda*> entries);
    void build_succs();
    void build_preds();
    void uce(Lambda* entry);
    Lambda* find_exit();
    void link_exit(Lambda* entry, Lambda* exit);
    void post_order_visit(LambdaSet&, LambdaSet&, Lambda* cur, Lambda* exit);
    template<bool forward> void rpo_numbering(Lambda* entry);
    template<bool forward> int po_visit(LambdaSet& set, Lambda* cur, int i);
    void link_succ(Lambda* src, Lambda* dst) { assert(contains(src) && contains(dst)); succs_[src].push_back(dst); };
    void link_pred(Lambda* src, Lambda* dst) { assert(contains(src) && contains(dst)); preds_[dst].push_back(src); };

    World& world_;
    DefSet in_scope_;
    std::vector<Lambda*> rpo_;
    std::vector<Lambda*> reverse_rpo_;
    mutable LambdaMap<std::vector<Lambda*>> preds_;
    mutable LambdaMap<std::vector<Lambda*>> succs_;
    mutable LambdaMap<int> sid_;
    mutable LambdaMap<int> reverse_sid_;

    friend class ScopeView;
};

//------------------------------------------------------------------------------

class ScopeView {
public:
    explicit ScopeView(const Scope& scope, const bool is_forward = true)
        : scope_(scope)
        , ptr(&const_cast<Scope&>(scope))
        , foo(*ptr)
        , is_forward_(is_forward)
    {}

    const Scope& scope() const { return scope_; }
    bool is_forward() const { return is_forward_; }
    /// All lambdas within this scope in reverse postorder.
    ArrayRef<Lambda*> rpo() const { return is_forward() ? scope().rpo_ : scope().reverse_rpo_; }
    Lambda* entry() const { return rpo().front(); }
    Lambda* exit()  const { return (is_forward() ? scope().reverse_rpo_ : scope().rpo_).front(); }
    /// Like \p rpo() but without \p entry()
    ArrayRef<Lambda*> body() const { return rpo().slice_from_begin(1); }
    const DefSet& in_scope() const { return scope().in_scope_; }
    bool contains(Def def) const { return scope().in_scope_.contains(def); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return is_forward() ? scope().preds(lambda) : scope().succs(lambda); }
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return is_forward() ? scope().succs(lambda) : scope().preds(lambda); }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    int sid(Lambda* lambda) const { 
        assert(contains(lambda)); 
        return (is_forward() ? scope().sid_ : scope().reverse_sid_)[lambda]; 
    }
    size_t size() const { return scope().size(); }
    World& world() const { return scope().world(); }

    typedef ArrayRef<Lambda*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

    typedef ArrayRef<Lambda*>::const_reverse_iterator const_reverse_iterator;
    const_reverse_iterator rbegin() const { return rpo().rbegin(); }
    const_reverse_iterator rend() const { return rpo().rend(); }

private:
    LambdaSet reachable(Lambda* entry);

    const Scope& scope_;
    Scope* ptr;
    Scope& foo;
    const bool is_forward_;

    friend class Scope;
};

//------------------------------------------------------------------------------

}

#endif
