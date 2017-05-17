#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <vector>

#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"
#include "thorin/util/indexmap.h"
#include "thorin/util/indexset.h"
#include "thorin/util/stream.h"
#include "thorin/util/ycomp.h"

namespace thorin {

//------------------------------------------------------------------------------

template<bool> class LoopTree;
template<bool> class DomTreeBase;
template<bool> class DomFrontierBase;

/**
 * A Control-Flow Node.
 * Managed by @p CFA.
 */
class CFNodeBase : public MagicCast<CFNodeBase>, public Streamable {
public:
    CFNodeBase(const Def* def)
        : def_(def)
        , gid_(gid_counter_++)
    {}

    uint64_t gid() const { return gid_; }
    const Def* def() const { return def_; }

private:
    const Def* def_;
    size_t gid_;
    static uint64_t gid_counter_;
};

class RealCFNode : public CFNodeBase {
protected:
    RealCFNode(const Def* def)
        : CFNodeBase(def)
    {}

protected:
    mutable size_t f_index_ = -1; ///< RPO index in a forward @p CFG.
    mutable size_t b_index_ = -1; ///< RPO index in a backwards @p CFG.

    friend class CFABuilder;
    template<bool> friend class CFG;
};

class CFNode;
typedef GIDSet<const CFNode*> CFNodes;

/// This node represents a @p CFNode within its underlying @p Scope.
class CFNode : public RealCFNode {
public:
    CFNode(Continuation* continuation)
        : RealCFNode(continuation)
    {}

    Continuation* continuation() const { return def()->as_continuation(); }
    virtual std::ostream& stream(std::ostream&) const override;

private:
    const CFNodes& preds() const { return preds_; }
    const CFNodes& succs() const { return succs_; }
    void link(const CFNode* other) const;

    mutable CFNodes preds_;
    mutable CFNodes succs_;

    friend class CFABase;
    friend class CFABuilder;
    friend class CFA;
    template<bool> friend class CFG;
};

//------------------------------------------------------------------------------

/**
 * Control Flow Analysis.
 * Base class for all CFA aglorithms.
 */
class CFABase {
public:
    CFABase(const CFABase&) = delete;
    CFABase& operator= (CFABase) = delete;

    explicit CFABase(const Scope& scope);
    ~CFABase();

    const Scope& scope() const { return scope_; }
    size_t size() const { return nodes().size(); }
    const ContinuationMap<const CFNode*>& nodes() const { return nodes_; }
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    const CFNode* operator [] (Continuation* continuation) const { return find(nodes_, continuation); }

protected:
    /// invoke in the derived class after everything else has been done
    void init();

private:
    void link_to_exit();
    void verify();
    const CFNodes& preds(Continuation* continuation) const { auto cn = nodes_.find(continuation)->second; assert(cn); return cn->preds(); }
    const CFNodes& succs(Continuation* continuation) const { auto cn = nodes_.find(continuation)->second; assert(cn); return cn->succs(); }
    const CFNode* entry() const { return entry_; }
    const CFNode* exit() const { return exit_; }

    const Scope& scope_;
    ContinuationMap<const CFNode*> nodes_;
    const CFNode* entry_;
    const CFNode* exit_;
    mutable std::unique_ptr<const F_CFG> f_cfg_;
    mutable std::unique_ptr<const B_CFG> b_cfg_;

    friend class CFABuilder;
    friend class CFA;
    template<bool> friend class CFG;
};

class CFA : public CFABase {
public:
    explicit CFA(const Scope& scope);

private:
    const CFNode* node(Continuation*);
};

/**
 * Control Flow Analysis.
 * This class maintains information obtained by local control-flow analysis run on a @p Scope.
 * See "Shallow Embedding of DSLs via Online Partial Evaluation", Leißa et.al. for details.
 */
class CFASmart : public CFABase {
public:
    explicit CFASmart(const Scope& scope);


    friend class CFABuilder;
};

//------------------------------------------------------------------------------

/**
 * A Control-Flow Graph.
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

    explicit CFG(const CFABase&);
    static const CFG& create(const Scope& scope) { return scope.cfg<forward>(); }

    const CFABase& cfa() const { return cfa_; }
    size_t size() const { return cfa().size(); }
    const CFNodes& preds(const CFNode* n) const { return n ? (forward ? n->preds() : n->succs()) : empty_; }
    const CFNodes& succs(const CFNode* n) const { return n ? (forward ? n->succs() : n->preds()) : empty_; }
    const CFNodes& preds(Continuation* continuation) const { return preds(cfa()[continuation]); }
    const CFNodes& succs(Continuation* continuation) const { return succs(cfa()[continuation]); }
    size_t num_preds(const CFNode* n) const { return preds(n).size(); }
    size_t num_succs(const CFNode* n) const { return succs(n).size(); }
    size_t num_preds(Continuation* continuation) const { return num_preds(cfa()[continuation]); }
    size_t num_succs(Continuation* continuation) const { return num_succs(cfa()[continuation]); }
    const CFNode* entry() const { return forward ? cfa().entry() : cfa().exit();  }
    const CFNode* exit()  const { return forward ? cfa().exit()  : cfa().entry(); }

    ArrayRef<const CFNode*> reverse_post_order() const { return rpo_.array(); }
    Range<ArrayRef<const CFNode*>::const_reverse_iterator> post_order() const { return reverse_range(rpo_.array()); }
    const CFNode* reverse_post_order(size_t i) const { return rpo_.array()[i]; }  ///< Maps from reverse post-order index to @p CFNode.
    const CFNode* post_order(size_t i) const { return rpo_.array()[size()-1-i]; } ///< Maps from post-order index to @p CFNode.
    const CFNode* operator [] (Continuation* continuation) const { return cfa()[continuation]; }    ///< Maps from @p l to @p CFNode.
    const DomTreeBase<forward>& domtree() const;
    const LoopTree<forward>& looptree() const;
    const DomFrontierBase<forward>& domfrontier() const;
    virtual void stream_ycomp(std::ostream& out) const override;

    static size_t index(const CFNode* n) { return forward ? n->f_index_ : n->b_index_; }

private:
    size_t post_order_visit(const CFNode* n, size_t i);

    static CFNodes empty_;

    const CFABase& cfa_;
    Map<const CFNode*> rpo_;
    mutable std::unique_ptr<const DomTreeBase<forward>> domtree_;
    mutable std::unique_ptr<const LoopTree<forward>> looptree_;
    mutable std::unique_ptr<const DomFrontierBase<forward>> domfrontier_;
};

//------------------------------------------------------------------------------

}

#endif
