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
    explicit Scope(Lambda* entry, bool forwards = true);
    ~Scope();

    /// All lambdas within this scope in reverse postorder.
    ArrayRef<Lambda*> rpo() const { return forwards() ? rpo_ : reverse_rpo_; }
    Lambda* entry() const { return rpo().front(); }
    Lambda* exit()  const { return rpo().back(); }
    /// Like \p rpo() but without \p entry()
    ArrayRef<Lambda*> body() const { return rpo().slice_from_begin(1); }
    const DefSet& in_scope() const { return in_scope_; }
    bool contains(Def def) const { return in_scope_.contains(def); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return (forwards() ? preds_ : succs_)[lambda]; }
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return (forwards() ? succs_ : preds_)[lambda]; }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    int sid(Lambda* lambda) const { return (forwards() ? sid_ : reverse_sid_)[lambda]; }
    size_t size() const { return rpo_.size(); }
    World& world() const { return world_; }
    bool forwards() const { return forwards_; }

    typedef ArrayRef<Lambda*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

    typedef ArrayRef<Lambda*>::const_reverse_iterator const_reverse_iterator;
    const_reverse_iterator rbegin() const { return rpo().rbegin(); }
    const_reverse_iterator rend() const { return rpo().rend(); }

private:
    void identify_scope(Lambda* entry);
    void build_cfg(Lambda* entry);
    void uce(Lambda* entry);
    void find_exits(Lambda* entry);
    void rpo_numbering(Lambda* entry);
    int po_visit(LambdaSet& set, Lambda* cur, int i);
    void link(Lambda* src, Lambda* dst) {
        succs_[src].push_back(dst);
        preds_[dst].push_back(src);
    };
    void assign_sid(Lambda* lambda, int i) { (forwards() ? sid_ : reverse_sid_)[lambda] = i; }

    World& world_;
    bool forwards_;
    DefSet in_scope_;
    std::vector<Lambda*> rpo_;
    std::vector<Lambda*> reverse_rpo_;
    LambdaMap<std::vector<Lambda*>> preds_;
    LambdaMap<std::vector<Lambda*>> succs_;
    LambdaMap<int> sid_;
    LambdaMap<int> reverse_sid_;
};

//------------------------------------------------------------------------------

Array<Lambda*> top_level_lambdas(World& world);
Lambda* top_lambda(World& world);

//------------------------------------------------------------------------------

} // namespace thorin

#endif
