#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"

namespace thorin {

Array<Lambda*> top_level_lambdas(World& world);

template<bool forwards = true>
class ScopeBase {
public:
    explicit ScopeBase(Lambda* entry);
    ScopeBase(World& world)
        : ScopeBase(world, top_level_lambdas(world))
    {}
    ScopeBase(World& world, ArrayRef<Lambda*> entries);

    /// All lambdas within this scope in reverse postorder.
    ArrayRef<Lambda*> rpo() const { return rpo_; }
    Lambda* entry() const { return rpo_.front(); }
    Lambda* exit()  const { return rpo_.back(); }
    /// Like \p rpo() but without \p entry() and \p exit()
    ArrayRef<Lambda*> body() const { return rpo().slice_from_begin(1).slice_num_from_end(1); }
    const DefSet& in_scope() const { return in_scope_; }
    bool contains(Def def) const { return in_scope_.contains(def); }

    typedef ArrayRef<Lambda*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

    const std::vector<Lambda*>& preds(Lambda* lambda) const { return preds_.find(lambda)->second; }
    const std::vector<Lambda*>& succs(Lambda* lambda) const { return succs_.find(lambda)->second; }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    int sid(Lambda* lambda) const { return sid_.find(lambda)->second; }

    size_t size() const { return rpo_.size(); }
    World& world() const { return world_; }

private:
    void identify_scope(ArrayRef<Lambda*> entries);
    void build_cfg(ArrayRef<Lambda*> entries);
    void uce(Lambda* entry);
    void find_exits();
    void rpo_numbering(Lambda* entry);
    void link(Lambda* src, Lambda* dst) {
        succs_[src].push_back(dst);
        preds_[dst].push_back(src);
    };

    World& world_;
    DefSet in_scope_;
    std::vector<Lambda*> rpo_;
    LambdaMap<std::vector<Lambda*>> preds_;
    LambdaMap<std::vector<Lambda*>> succs_;
    LambdaMap<int> sid_;
};

typedef ScopeBase<true> Scope;
typedef ScopeBase<false> BackwardsScope;

} // namespace thorin

#endif
