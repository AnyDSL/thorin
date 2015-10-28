#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <iostream>
#include <vector>

#include "thorin/lambda.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/graphs.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/indexmap.h"
#include "thorin/util/stream.h"

namespace thorin {

//------------------------------------------------------------------------------

template<bool> class LoopTree;
template<bool> class DomTreeBase;
template<bool> class DFGBase;
class CFNode;
class InNode;
class OutNode;

struct CFNodeHash {
    uint64_t operator() (const CFNode* n) const;
};

typedef thorin::HashSet<const CFNode*, CFNodeHash> CFNodeSet;

//------------------------------------------------------------------------------

/**
 * @brief A Control-Flow Node.
 *
 * Managed by @p CFA.
 */
class CFNode : public MagicCast<CFNode>, public Streamable {
protected:
    CFNode(Def def)
        : def_(def)
    {}

public:
    Def def() const { return def_; }
    virtual const InNode* in_node() const = 0;

private:
    const CFNodeSet& preds() const { return preds_; }
    const CFNodeSet& succs() const { return succs_; }
    void link(const CFNode* other) const;

    static const size_t Fresh     = size_t(-1);
    static const size_t Unfresh   = size_t(-2);
    static const size_t Reachable = size_t(-3);
    static const size_t Visited   = size_t(-4);

    Def def_;
    mutable size_t f_index_ = Fresh;     ///< RPO index in a forward @p CFG.
    mutable size_t b_index_ = Reachable; ///< RPO index in a backwards @p CFG.
    mutable CFNodeSet preds_;
    mutable CFNodeSet succs_;

    friend class CFABuilder;
    friend class CFA;
    template<bool> friend class CFG;
};

/// This node represents a @p CFNode within its underlying @p Scope.
class InNode : public CFNode {
public:
    InNode(Lambda* lambda)
        : CFNode(lambda)
    {}
    virtual ~InNode();

    Lambda* lambda() const { return def()->as_lambda(); }
    const DefMap<const OutNode*>& out_nodes() const { return out_nodes_; }
    virtual const InNode* in_node() const override { return this; }
    virtual std::ostream& stream(std::ostream&) const override;

private:
    mutable DefMap<const OutNode*> out_nodes_;

    friend class CFABuilder;
};

typedef std::function<void(const InNode*)> YieldIn;

/// Any jumps targeting a @p Lambda or @p Param outside the @p CFA's underlying @p Scope target this node.
class OutNode : public CFNode {
public:
    OutNode(const InNode* context, Def def)
        : CFNode(def)
        , context_(context)
    {
        assert(def->isa<Param>() || def->isa<Lambda>());
    }

    virtual ~OutNode() {}

    const InNode* context() const { return context_; }
    const CFNodeSet& ancestors() const { return ancestors_; }
    virtual const InNode* in_node() const override { return context_; }
    virtual std::ostream& stream(std::ostream&) const override;

private:
    const InNode* context_;
    mutable CFNodeSet ancestors_;

    friend class CFABuilder;
};

//------------------------------------------------------------------------------

/**
 * @brief Control Flow Analysis.
 *
 * This class maintains information obtained by local control-flow analysis run on a @p Scope.
 * See "Shallow Embedding of DSLs via Online Partial Evaluation", Leißa et.al. for details.
 */
class CFA {
public:
    CFA(const CFA&) = delete;
    CFA& operator= (CFA) = delete;

    explicit CFA(const Scope& scope);
    ~CFA();

    const Scope& scope() const { return scope_; }
    size_t num_in_nodes() const { return num_in_nodes_; }
    size_t num_out_nodes() const { return num_out_nodes_; }
    size_t size() const { return num_in_nodes() + num_out_nodes(); }
    const Scope::Map<const InNode*>& in_nodes() const { return in_nodes_; }
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    const InNode* operator [] (Lambda* lambda) const { return find(in_nodes_, lambda); }

private:
    const CFNodeSet& preds(Lambda* lambda) const { return in_nodes_[lambda]->preds(); }
    const CFNodeSet& succs(Lambda* lambda) const { return in_nodes_[lambda]->succs(); }
    size_t num_preds(Lambda* lambda) const { return preds(lambda).size(); }
    size_t num_succs(Lambda* lambda) const { return succs(lambda).size(); }
    const InNode* entry() const { return in_nodes_.array().front(); }
    const InNode* exit() const { return in_nodes_.array().back(); }
    void error_dump() const;

    const Scope& scope_;
    Scope::Map<const InNode*> in_nodes_; ///< Maps lambda in scope to InNode.
    mutable AutoPtr<const F_CFG> f_cfg_;
    mutable AutoPtr<const B_CFG> b_cfg_;
    size_t num_in_nodes_ = 0;
    size_t num_out_nodes_ = 0;

    friend class CFABuilder;
    template<bool> friend class CFG;
};

//------------------------------------------------------------------------------

namespace detail {
    void in_next(const InNode* in, YieldIn f, std::function<const CFNodeSet&(const CFNode*)> next);
}

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
    const Scope& scope() const { return cfa().scope(); }
    size_t size() const { return cfa().size(); }
    const CFNodeSet& preds(const CFNode* n) const { return forward ? n->preds() : n->succs(); }
    const CFNodeSet& succs(const CFNode* n) const { return forward ? n->succs() : n->preds(); }
    void in_preds(const InNode* in, YieldIn f) const {
        detail::in_next(in, f, [&] (const CFNode* n) -> const CFNodeSet& { return preds(n); });
    }
    void in_succs(const InNode* in, YieldIn f) const {
        detail::in_next(in, f, [&] (const CFNode* n) -> const CFNodeSet& { return succs(n); });
    }
    void in_preds(Lambda* lambda, YieldIn f) const { in_preds(cfa()[lambda], f); }
    void in_succs(Lambda* lambda, YieldIn f) const { in_succs(cfa()[lambda], f); }
    size_t num_preds(const CFNode* n) const { return preds(n).size(); }
    size_t num_succs(const CFNode* n) const { return succs(n).size(); }
    const InNode* entry() const { return forward ? cfa().entry() : cfa().exit();  }
    const InNode* exit()  const { return forward ? cfa().exit()  : cfa().entry(); }

    ArrayRef<const CFNode*> rpo() const { return rpo_.array(); }        ///< All @p CFNode%s within this @p CFG in reverse post-order.
    ArrayRef<const CFNode*> body() const { return rpo().skip_front(); } ///< Like @p rpo() but without @p entry()

    /// All @p CFNode%s within this @p CFG in post-order.
    Range<ArrayRef<const CFNode*>::const_reverse_iterator> po() const { return reverse_range(rpo_.array()); }

    const CFNode* rpo(size_t i) const { return rpo_.array()[i]; }           ///< Maps from reverse post-order index to @p CFNode.
    const CFNode* po(size_t i) const { return rpo_.array()[size()-1-i]; }   ///< Maps from post-order index to @p CFNode.
    const InNode* operator [] (Lambda* l) const { return cfa()[l]; }        ///< Maps from @p l to @p InNode.

    /// Range of @p InNode%s in reverse post-order. All @p OutNode%s will be skipped during iteration.
    Range<filter_iterator<ArrayRef<const CFNode*>::const_iterator, bool (*)(const CFNode*), const InNode*>> in_rpo() const {
        return range<const InNode*>(rpo().begin(), rpo().end(), is_in_node);
    }

    /// Range of @p InNode%s in post-order. All @p OutNode%s will be skipped during iteration.
    Range<filter_iterator<ArrayRef<const CFNode*>::const_reverse_iterator, bool (*)(const CFNode*), const InNode*>> in_po() const {
        return range<const InNode*>(rpo().rbegin(), rpo().rend(), is_in_node);
    }

    const DomTreeBase<forward>& domtree() const;
    const LoopTree<forward>& looptree() const;
    const DFGBase<forward>& dfg() const;

    static size_t index(const CFNode* n) { return forward ? n->f_index_ : n->b_index_; }
    static bool is_in_node(const CFNode* n) { return n->isa<InNode>(); }

    static const CFG& get(const Scope& scope) { return scope.cfg<forward>(); }

    void ycomp(std::ostream& ostream = std::cout) const {
        emit_ycomp(ostream, scope(), range(rpo()),
            [] (const CFNode* n) { return range(n->succs()); },
            YComp_Orientation::TopToBottom
        );
    }

    static void emit_world(World& world, std::ostream& out = std::cout) {
        emit_ycomp(out, world, &CFG<forward>::ycomp);
    }

private:
    size_t post_order_visit(const CFNode* n, size_t i);

    const CFA& cfa_;
    Map<const CFNode*> rpo_;
    mutable AutoPtr<const DomTreeBase<forward>> domtree_;
    mutable AutoPtr<const LoopTree<forward>> looptree_;
    mutable AutoPtr<const DFGBase<forward>> dfg_;
};

//------------------------------------------------------------------------------

}

#endif
