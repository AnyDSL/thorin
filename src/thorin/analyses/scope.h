#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"

namespace thorin {

Array<Lambda*> top_level_lambdas(World& world);

class Scope {
public:
    explicit Scope(Lambda* entry);
    Scope(World& world, ArrayRef<Lambda*> entries);
    Scope(ArrayRef<Lambda*> entries)
        : Scope(entries[0]->world(), entries)
    {}

    bool contains(Def def) const { return in_scope_.contains(def); }

    /// All lambdas within this scope in reverse postorder.
    ArrayRef<Lambda*> rpo() const { return rpo_; }
    ArrayRef<Lambda*> entries() const { return ArrayRef<Lambda*>(rpo_).slice_to_end(num_entries()); }
    /// Like \p rpo() but without \p entries().
    ArrayRef<Lambda*> body() const { return rpo().slice_from_begin(num_entries()); }
    ArrayRef<Lambda*> backwards_rpo() const;
    ArrayRef<Lambda*> exits() const { return backwards_rpo().slice_to_end(num_exits()); }
    /// Like \p backwards_rpo() but without \p exits().
    ArrayRef<Lambda*> backwards_body() const { return backwards_rpo().slice_from_begin(num_exits()); }
    const DefSet& in_scope() const { return in_scope_; }

    Lambda* rpo(size_t i) const { return rpo_[i]; }
    Lambda* operator [] (size_t i) const { return rpo(i); }

    typedef ArrayRef<Lambda*>::const_iterator const_iterator;
    const_iterator begin() const { return rpo().begin(); }
    const_iterator end() const { return rpo().end(); }

    ArrayRef<Lambda*> preds(Lambda* lambda) const;
    ArrayRef<Lambda*> succs(Lambda* lambda) const;

    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    size_t num_entries() const { return num_entries_; }
    size_t num_exits() const { if (num_exits_ == size_t(-1)) backwards_rpo(); return num_exits_; }

    size_t size() const { return rpo_.size(); }
    World& world() const { return world_; }

    bool is_entry(Lambda* lambda) const;
    bool is_exit(Lambda* lambda) const;

    size_t sid(Lambda* lambda) const;
    size_t backwards_sid(Lambda* lambda) const;

private:
    void identify_scope(ArrayRef<Lambda*> entries);
    void rpo_numbering(ArrayRef<Lambda*> entries);
    void collect(LambdaSet& entries, Lambda* lambda);
    template<bool forwards> size_t po_visit(LambdaSet&, Lambda* cur, size_t i) const;
    template<bool forwards> size_t number(LambdaSet&, Lambda* cur, size_t i) const;

    World& world_;
    std::vector<Lambda*> rpo_;
    DefSet in_scope_;
    size_t num_entries_;

    struct LambdaSidInfo {
        size_t sid;
        size_t backwards_sid;

        LambdaSidInfo()
            : sid(-1)
            , backwards_sid(-1)
        {}
    };

    mutable LambdaMap<LambdaSidInfo> sid_;

    mutable size_t num_exits_;
    mutable AutoPtr<Array<Lambda*>> backwards_rpo_;
    mutable Array<Array<Lambda*>> preds_;
    mutable Array<Array<Lambda*>> succs_;
};

} // namespace thorin

#endif
