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

typedef std::vector<const Def*> FreeVariables;

LambdaSet find_scope(Lambda* entry);

class Scope {
public:

    typedef Array<Lambda*> Lambdas;

    explicit Scope(Lambda* entry);
    ~Scope();

    bool contains(Lambda* lambda) const { return lambdas_.find(lambda) != lambdas_.end(); }
    const LambdaSet& lambdas() const { return lambdas_; }
    const Lambdas& rpo() const { return rpo_; }
    Lambda* rpo(size_t i) const { return rpo_[i]; }
    const Lambdas& preds(Lambda* lambda) const;
    const Lambdas& succs(Lambda* lambda) const;
    Lambda* entry() const { return rpo_[0]; }
    size_t size() const { return lambdas_.size(); }
    void reassign_sids();
    World& world() const { return entry()->world(); }
    FreeVariables free_variables() const;

    Lambda* clone(bool self = true, const GenericMap& generic_map = GenericMap());
    Lambda* drop(ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, 
                 bool self = true, const GenericMap& generic_map = GenericMap());
    Lambda* lift(ArrayRef<const Def*> to_lift, 
                 bool self = true, const GenericMap& generic_map = GenericMap());
    Lambda* mangle(ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, 
                   ArrayRef<const Def*> to_lift, 
                   bool self = true, const GenericMap& generic_map = GenericMap());

    const DomTree& domtree() const;
    const LoopForestNode* loopforest() const;
    const LoopInfo& loopinfo() const;

private:

    static size_t number(const LambdaSet& lambdas, Lambda* cur, size_t i);

    LambdaSet lambdas_;
    Lambdas rpo_;
    Array<Lambdas> preds_;
    Array<Lambdas> succs_;
    mutable AutoPtr<DomTree> domtree_;
    mutable AutoPtr<LoopForestNode> loopforest_;
    mutable AutoPtr<LoopInfo> loopinfo_;
};

} // namespace anydsl2

#endif

