#ifndef ANYDSL2_ANALYSES_SCOPE_H
#define ANYDSL2_ANALYSES_SCOPE_H

#include <vector>

#include "anydsl2/lambda.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/autoptr.h"

namespace anydsl2 {

class DomTree;
class LoopTreeNode;
class LoopInfo;

class Scope {
public:

    explicit Scope(Lambda* entry);
    explicit Scope(World& world, ArrayRef<Lambda*> entries);
    explicit Scope(World& world);
    ~Scope();

    bool contains(Lambda* lambda) const { return lambda->scope() == this; }
    /// All bodies with this scope in reverse postorder.
    ArrayRef<Lambda*> rpo() const { return rpo_; }
    const std::vector<Lambda*>& entries() const { return entries_; }
    const std::vector<Lambda*>& exits() const;
    Array<Lambda*> copy_entries() const { return Array<Lambda*>(entries_); }
    /// Like \p rpo() but without \p entries().
    ArrayRef<Lambda*> body() const { return rpo().slice_back(num_entries()); }
    Lambda* rpo(size_t i) const { return rpo_[i]; }
    Lambda* operator [] (size_t i) const { return rpo(i); }
    ArrayRef<Lambda*> preds(Lambda* lambda) const;
    ArrayRef<Lambda*> succs(Lambda* lambda) const;
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    size_t num_entries() const { return entries().size(); }
    size_t size() const { return rpo_.size(); }
    World& world() const { return world_; }
    bool is_entry(Lambda* lambda) const { assert(contains(lambda)); return lambda->sid() < num_entries(); }

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
    const DomTree& postdomtree() const;
    const LoopTreeNode* looptree() const;
    const LoopInfo& loopinfo() const;

private:

    void analyze();
    void process();
    void jump_to_param_users(const size_t pass, Lambda* lambda, Lambda* limit);
    void up(const size_t pass, Lambda* lambda, Lambda* limit);
    void find_user(const size_t pass, const Def* def, Lambda* limit);
    size_t number(const size_t pass, Lambda* cur, size_t i);
    void insert(const size_t pass, Lambda* lambda) { 
        lambda->visit_first(pass); 
        lambda->scope_ = this; 
        rpo_.push_back(lambda); 
    }

    World& world_;
    std::vector<Lambda*> entries_;
    std::vector<Lambda*> rpo_;
    mutable AutoPtr< std::vector<Lambda*> > exits_;
    mutable Array< Array<Lambda*> > preds_;
    mutable Array< Array<Lambda*> > succs_;
    mutable AutoPtr<DomTree> domtree_;
    mutable AutoPtr<DomTree> postdomtree_;
    mutable AutoPtr<LoopTreeNode> looptree_;
    mutable AutoPtr<LoopInfo> loopinfo_;
};

} // namespace anydsl2

#endif
