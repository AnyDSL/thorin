#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <iostream>
#include <vector>

#include "thorin/lambda.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/indexmap.h"
#include "thorin/util/stream.h"
#include "thorin/util/ycomp.h"

namespace thorin {

//------------------------------------------------------------------------------

template<bool> class LoopTree;
template<bool> class DomTreeBase;
template<bool> class DFGBase;
class CFNode;
typedef std::vector<const CFNode*> CFNodes;

/**
 * @brief A Control-Flow Node.
 *
 * Managed by @p CFA.
 */
class CFNodeBase : public MagicCast<CFNodeBase>, public Streamable {
protected:
    CFNodeBase(Def def)
        : def_(def)
    {}

public:
    Def def() const { return def_; }

protected:
    static const size_t Fresh     = size_t(-1);
    static const size_t Unfresh   = size_t(-2);
    static const size_t Reachable = size_t(-3);
    static const size_t Visited   = size_t(-4);

    mutable size_t f_index_ = Fresh;     ///< RPO index in a forward @p CFG.
    mutable size_t b_index_ = Reachable; ///< RPO index in a backwards @p CFG.

private:
    Def def_;

    friend class CFABuilder;
    template<bool> friend class CFG;
};

/// This node represents a @p CFNode within its underlying @p Scope.
class CFNode : public CFNodeBase {
public:
    CFNode(Lambda* lambda)
        : CFNodeBase(lambda)
    {}

    Lambda* lambda() const { return def()->as_lambda(); }
    virtual std::ostream& stream(std::ostream&) const override;

private:
    const CFNodes& preds() const { return preds_; }
    const CFNodes& succs() const { return succs_; }
    void link(const CFNode* other) const;

    mutable CFNodes preds_;
    mutable CFNodes succs_;

    friend class CFA;
    friend class CFABuilder;
    template<bool> friend class CFG;
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
    size_t size() const { return size_; }
    const Scope::Map<const CFNode*>& nodes() const { return nodes_; }
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    const CFNode* operator [] (Lambda* lambda) const { return find(nodes_, lambda); }

private:
    const CFNodes& preds(Lambda* lambda) const { return nodes_[lambda]->preds(); }
    const CFNodes& succs(Lambda* lambda) const { return nodes_[lambda]->succs(); }
    const CFNode* entry() const { return nodes_.array().front(); }
    const CFNode* exit() const { return nodes_.array().back(); }
    void error_dump() const;

    const Scope& scope_;
    Scope::Map<const CFNode*> nodes_;
    size_t size_ = 0;
    mutable AutoPtr<const F_CFG> f_cfg_;
    mutable AutoPtr<const B_CFG> b_cfg_;

    friend class CFABuilder;
    template<bool> friend class CFG;
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
class CFG : public YComp {
public:
    template<class Value>
    using Map = IndexMap<CFG<forward>, const CFNode*, Value>;
    using Set = IndexSet<CFG<forward>, const CFNode*>;

    CFG(const CFG&) = delete;
    CFG& operator= (CFG) = delete;

    explicit CFG(const CFA&);
    static const CFG& create(const Scope& scope) { return scope.cfg<forward>(); }

    const CFA& cfa() const { return cfa_; }
    size_t size() const { return cfa().size(); }
    const CFNodes& preds(Lambda* lambda) const { return preds(cfa()[lambda]); }
    const CFNodes& succs(Lambda* lambda) const { return succs(cfa()[lambda]); }
    const CFNodes& preds(const CFNode* n) const { return forward ? n->preds() : n->succs(); }
    const CFNodes& succs(const CFNode* n) const { return forward ? n->succs() : n->preds(); }
    size_t num_preds(const CFNode* n) const { return preds(n).size(); }
    size_t num_succs(const CFNode* n) const { return succs(n).size(); }
    const CFNode* entry() const { return forward ? cfa().entry() : cfa().exit();  }
    const CFNode* exit()  const { return forward ? cfa().exit()  : cfa().entry(); }

    ArrayRef<const CFNode*> rpo() const { return rpo_.array(); }        ///< All @p CFNode%s within this @p CFG in reverse post-order.
    ArrayRef<const CFNode*> body() const { return rpo().skip_front(); } ///< Like @p rpo() but without @p entry()

    /// All @p CFNode%s within this @p CFG in post-order.
    Range<ArrayRef<const CFNode*>::const_reverse_iterator> po() const { return reverse_range(rpo_.array()); }

    const CFNode* rpo(size_t i) const { return rpo_.array()[i]; }           ///< Maps from reverse post-order index to @p CFNode.
    const CFNode* po(size_t i) const { return rpo_.array()[size()-1-i]; }   ///< Maps from post-order index to @p CFNode.
    const CFNode* operator [] (Lambda* l) const { return cfa()[l]; }        ///< Maps from @p l to @p CFNode.

    const DomTreeBase<forward>& domtree() const;
    const LoopTree<forward>& looptree() const;
    const DFGBase<forward>& dfg() const;

    virtual void stream_ycomp(std::ostream& out) const override {
        thorin::ycomp(out, scope(), range(rpo()),
            [] (const CFNode* n) { return range(n->succs()); },
            YComp_Orientation::TopToBottom
        );
    }

    static size_t index(const CFNode* n) { return forward ? n->f_index_ : n->b_index_; }

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
