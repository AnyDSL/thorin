#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/indexmap.h"

namespace thorin {

//------------------------------------------------------------------------------

class LoopTree;

template<bool> class DomTreeBase;
typedef DomTreeBase<true>  DomTree;
typedef DomTreeBase<false> PostDomTree;

template<bool> class CFG;
typedef CFG<true>  F_CFG;
typedef CFG<false> B_CFG;

class InCFNode;
class OutCFNode;

//------------------------------------------------------------------------------

/**
 * @brief A Control-Flow Node.
 *
 * Managed by @p CFA.
 */
class CFNode : public MagicCast<CFNode> {
protected:
    CFNode(Def def)
        : def_(def)
    {}

public:
    Def def() const { return def_; }
    virtual const InCFNode* in_node() const = 0;

private:
    ArrayRef<const CFNode*> preds() const { return preds_; }
    ArrayRef<const CFNode*> succs() const { return succs_; }
    void link(const CFNode* other) const {
        this->succs_.push_back(other);
        other->preds_.push_back(this);
    }

    Def def_;
    mutable size_t f_index_ = -1; ///< RPO index in a forward @p CFG.
    mutable size_t b_index_ = -1; ///< RPO index in a backwards @p CFG.
    mutable std::vector<const CFNode*> preds_;
    mutable std::vector<const CFNode*> succs_;

    friend class CFABuilder;
    friend class CFA;
    template<bool> friend class CFG;
};

/// This node represents a @p CFNode within its underlying @p Scope.
class InCFNode : public CFNode {
public:
    InCFNode(Lambda* lambda)
        : CFNode(lambda)
    {}
    virtual ~InCFNode();

    Lambda* lambda() const { return def()->as_lambda(); }
    const DefMap<const OutCFNode*>& out_nodes() const { return out_nodes_; }
    virtual const InCFNode* in_node() const override { return this; }

private:
    Lambda* lambda_;
    mutable DefMap<const OutCFNode*> out_nodes_;

    friend class CFABuilder;
};

/// Any jumps targeting a @p Lambda or @p Param outside the @p CFA's underlying @p Scope target this node.
class OutCFNode : public CFNode {
public:
    OutCFNode(const InCFNode* parent, Def def)
        : CFNode(def)
        , parent_(parent)
    {
        assert(def->isa<Param>() || def->isa<Lambda>());
    }
    virtual ~OutCFNode() {}

    const InCFNode* parent() const { return parent_; }
    virtual const InCFNode* in_node() const override { return parent_; }

private:
    const InCFNode* parent_;
};

//------------------------------------------------------------------------------

/**
 * @brief Control Flow Analysis.
 *
 * This class maintains information obtained by a 0-CFA run on a @p Scope.
 */
class CFA {
public:
    CFA(const CFA&) = delete;
    CFA& operator= (CFA) = delete;

    explicit CFA(const Scope& scope);
    ~CFA();

    const Scope& scope() const { return scope_; }
    size_t num_cf_nodes() const { return num_cf_nodes_; }
    const Scope::Map<const InCFNode*>& in_nodes() const { return in_nodes_; }
    ArrayRef<const CFNode*> preds(Lambda* lambda) const { return in_nodes_[lambda]->preds(); }
    ArrayRef<const CFNode*> succs(Lambda* lambda) const { return in_nodes_[lambda]->succs(); }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    const InCFNode* entry() const { return in_nodes_.entry(); }
    const InCFNode* exit() const { return in_nodes_.exit(); }
    const F_CFG* f_cfg() const;
    const B_CFG* b_cfg() const;
    const DomTree* domtree() const;
    const PostDomTree* postdomtree() const;
    const LoopTree* looptree() const;
    const InCFNode* lookup(Lambda* lambda) const { return find(in_nodes_, lambda); }

private:
    const Scope& scope_;
    Scope::Map<const InCFNode*> in_nodes_; ///< Maps lambda in scope to InCFNode. 
    mutable AutoPtr<const F_CFG> f_cfg_;
    mutable AutoPtr<const B_CFG> b_cfg_;
    mutable AutoPtr<const LoopTree> looptree_;
    size_t num_cf_nodes_ = 0;

    friend class CFABuilder;
};

//------------------------------------------------------------------------------

/** 
 * @brief A Control-Flow Graph.
 *
 * A small wrapper for the information obtained by a @p CFA.
 * The template parameter @p forward determines the direction of the edges.
 * @c true means a conventional @p CFG.
 * @c false means that all edges in this @p CFG are reverted.
 * Thus, a dominance analysis, for example, becomes a post-dominance analysis.
 * @see DomTreeBase
 */
template<bool forward>
class CFG {
public:
    template<class Value>
    using Map = IndexMap<CFG<forward>, const CFNode*, Value>;
    using Set = IndexSet<CFG<forward>, const CFNode*>;

    CFG(const CFG&) = delete;
    CFG& operator= (CFG) = delete;

    explicit CFG(const CFA&);

    const CFA& cfa() const { return cfa_; }
    size_t size() const { return cfa_.num_cf_nodes(); }
    ArrayRef<const CFNode*> preds(const CFNode* n) const { return forward ? n->preds() : n->succs(); }
    ArrayRef<const CFNode*> succs(const CFNode* n) const { return forward ? n->succs() : n->preds(); }
    size_t num_preds(const CFNode* n) const { return preds(n).size(); }
    size_t num_succs(const CFNode* n) const { return succs(n).size(); }
    const InCFNode* entry() const { return forward ? cfa().entry() : cfa().exit();  }
    const InCFNode* exit()  const { return forward ? cfa().exit()  : cfa().entry(); }
    /// All lambdas within this scope in reverse post-order.
    ArrayRef<const CFNode*> rpo() const { return rpo_.array(); }
    const CFNode* rpo(size_t i) const { return rpo_.array()[i]; }
    /// Range of @p InCFNode%s, i.e., all @p OutCFNode%s will be skipped during iteration.
    Range<filter_iterator<ArrayRef<const CFNode*>::const_iterator, bool (*)(const CFNode*), const InCFNode*>> in_rpo() const { 
        return range<const InCFNode*>(rpo().begin(), rpo().end(), is_in_node);
    }
    Range<filter_iterator<ArrayRef<const CFNode*>::const_reverse_iterator, 
          bool (*)(const CFNode*), const InCFNode*>> reverse_in_rpo() const { 
        return range<const InCFNode*>(rpo().rbegin(), rpo().rend(), is_in_node);
    }
    /// Like @p rpo() but without @p entry()
    ArrayRef<const CFNode*> body() const { return rpo().slice_from_begin(1); }
    const InCFNode* lookup(Lambda* lambda) const { return cfa().lookup(lambda); }
    const DomTreeBase<forward>* domtree() const;

    static size_t index(const CFNode* n) { return forward ? n->f_index_ : n->b_index_; }
    static bool is_in_node(const CFNode* n) { return n->isa<InCFNode>(); }

private:
    size_t post_order_visit(const CFNode* n, size_t i);

    const CFA& cfa_;
    Map<const CFNode*> rpo_;
    mutable AutoPtr<const DomTreeBase<forward>> domtree_;
};

//------------------------------------------------------------------------------

}

#endif
