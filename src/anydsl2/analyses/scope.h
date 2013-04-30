#ifndef ANYDSL2_ANALYSES_SCOPE_H
#define ANYDSL2_ANALYSES_SCOPE_H

#include <vector>

#include "anydsl2/lambda.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/autoptr.h"

namespace anydsl2 {

class DomTree;
class LoopForestNode;
class LoopInfo;

class Scope {
public:

    explicit Scope(Lambda* entry);
    explicit Scope(ArrayRef<Lambda*> entries);
    explicit Scope(World& world);
    ~Scope();

    bool contains(Lambda* lambda) const { return lambda->scope() == this; }
    /// All bodies with this scope in reverse postorder.
    ArrayRef<Lambda*> rpo() const { return rpo_; }
    const std::vector<Lambda*>& entries() const { return entries_; }
    /// Like \p rpo() but without \p entries().
    ArrayRef<Lambda*> body() const { return rpo().slice_back(num_entries()); }
    Lambda* rpo(size_t i) const { return rpo_[i]; }
    Lambda* operator [] (size_t i) const { return rpo(i); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const;
    ArrayRef<Lambda*> succs(Lambda* lambda) const;
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    Lambda* entry() const { return rpo_[0]; }
    size_t num_entries() const { return entries().size(); }
    size_t size() const { return rpo_.size(); }
    World& world() const { return entry()->world(); }

    Lambda* clone(const GenericMap& generic_map = GenericMap());
    Lambda* drop(ArrayRef<const Def*> with);
    Lambda* drop(ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, 
                 const GenericMap& generic_map = GenericMap());
    Lambda* lift(ArrayRef<const Def*> to_lift, 
                 const GenericMap& generic_map = GenericMap());
    Lambda* mangle(ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, 
                   ArrayRef<const Def*> to_lift, 
                   const GenericMap& generic_map = GenericMap());

    const DomTree& domtree() const;
    const LoopForestNode* loopforest() const;
    const LoopInfo& loopinfo() const;

private:

    void analyze();
    void process();
    void jump_to_param_users(const size_t pass, Lambda* lambda);
    void up(const size_t pass, Lambda* lambda);
    void find_user(const size_t pass, const Def* def);
    size_t number(const size_t pass, Lambda* cur, size_t i);
    void insert(const size_t pass, Lambda* lambda) { 
        lambda->visit_first(pass); 
        lambda->scope_ = this; 
        rpo_.push_back(lambda); 
    }
    template<class T>
    void fill_succ_pred(const Lambdas& lsp, T& sp);

    std::vector<Lambda*> entries_;
    std::vector<Lambda*> rpo_;
    Array< Array<Lambda*> > preds_;
    Array< Array<Lambda*> > succs_;
    mutable AutoPtr<DomTree> domtree_;
    mutable AutoPtr<LoopForestNode> loopforest_;
    mutable AutoPtr<LoopInfo> loopinfo_;
    Lambda* hack_;
};

} // namespace anydsl2

#endif
