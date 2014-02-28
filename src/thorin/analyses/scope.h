#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"

namespace thorin {

class Scope {
public:
    enum {
        Forward                             = 1 << 0,
        UniqueExit                          = 1 << 1,
        Forward_UniqueExit                  = Forward | UniqueExit,
        Forward_No_UniqueExit               = Forward,
        Backward /**Implies unique exist*/  =           UniqueExit,
    };

    /// Always builds a unique meta \p Lambda as entry.
    explicit Scope(World& world, ArrayRef<Lambda*> entries, int mode = Forward_No_UniqueExit);
    /// Does not build a meta \p Lambda
    explicit Scope(Lambda* entry, int mode = Forward_No_UniqueExit);
    ~Scope();

    /// All lambdas within this scope in reverse postorder.
    ArrayRef<Lambda*> rpo() const { return is_forward() ? rpo_ : reverse_rpo_; }
    Lambda* entry() const { return rpo().front(); }
    Lambda* exit()  const { assert(has_unique_exit()); return rpo().back(); }
    /// Like \p rpo() but without \p entry()
    ArrayRef<Lambda*> body() const { return rpo().slice_from_begin(1); }
    const DefSet& in_scope() const { return in_scope_; }
    bool contains(Def def) const { return in_scope_.contains(def); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const { return (is_forward() ? preds_ : succs_)[lambda]; }
    ArrayRef<Lambda*> succs(Lambda* lambda) const { return (is_forward() ? succs_ : preds_)[lambda]; }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    int sid(Lambda* lambda) const { assert(contains(lambda)); return (is_forward() ? sid_ : reverse_sid_)[lambda]; }
    size_t size() const { return rpo_.size(); }
    World& world() const { return world_; }
    bool is_forward() const { return mode_ & Forward; ; }
    bool has_unique_exit() const { return mode_ & UniqueExit; }
    int mode() const { return mode_; }
    Scope& reverse() { assert(has_unique_exit()); mode_ ^= Forward; return *this; }

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
    template<bool forward> void rpo_numbering(Lambda* entry, Lambda* exit);
    template<bool forward> int po_visit(LambdaSet& set, Lambda* cur, int i);
    void link_succ(Lambda* src, Lambda* dst) { assert(contains(src) && contains(dst)); succs_[src].push_back(dst); };
    void link_pred(Lambda* src, Lambda* dst) { assert(contains(src) && contains(dst)); preds_[dst].push_back(src); };

    World& world_;
    int mode_;
    DefSet in_scope_;
    std::vector<Lambda*> rpo_;
    std::vector<Lambda*> reverse_rpo_;
    LambdaMap<std::vector<Lambda*>> preds_;
    LambdaMap<std::vector<Lambda*>> succs_;
    LambdaMap<int> sid_;
    LambdaMap<int> reverse_sid_;
};

}

#endif
