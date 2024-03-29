#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <vector>

#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"
#include "thorin/util/indexmap.h"
#include "thorin/util/indexset.h"
#include "thorin/util/stream.h"

namespace thorin {

//------------------------------------------------------------------------------

template<bool> class LoopTree;
template<bool> class DomTreeBase;
template<bool> class DomFrontierBase;

typedef GIDSet<const CFNode*> CFNodes;

/**
 * A Control-Flow Node.
 * Managed by @p CFA.
 */
class CFNode : public RuntimeCast<CFNode>, public Streamable<CFNode> {
public:
    CFNode(Continuation* continuation)
        : continuation_(continuation)
        , gid_(gid_counter_++)
    {}

    uint64_t gid() const { return gid_; }
    Continuation* continuation() const { return continuation_; }
    Stream& stream(Stream&) const;

private:
    const CFNodes& preds() const { return preds_; }
    const CFNodes& succs() const { return succs_; }
    void link(const CFNode* other) const;

    mutable size_t f_index_ = -1; ///< RPO index in a forward @p CFG.
    mutable size_t b_index_ = -1; ///< RPO index in a backwards @p CFG.

    Continuation* continuation_;
    size_t gid_;
    static uint64_t gid_counter_;
    mutable CFNodes preds_;
    mutable CFNodes succs_;

    friend class CFA;
    template<bool> friend class CFG;
};

//------------------------------------------------------------------------------

/// Control Flow Analysis.
class CFA {
public:
    CFA(const CFA&) = delete;
    CFA& operator= (CFA) = delete;

    explicit CFA(const Scope& scope);
    ~CFA();

    const Scope& scope() const { return scope_; }
    size_t size() const { return nodes().size(); }
    const ContinuationMap<const CFNode*>& nodes() const { return nodes_; }
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    const CFNode* operator[](Continuation* cont) const { return nodes_.lookup(cont).value_or(nullptr); }

private:
    void link_to_exit();
    void verify();
    const CFNodes& preds(Continuation* continuation) const { auto cn = nodes_.find(continuation)->second; assert(cn); return cn->preds(); }
    const CFNodes& succs(Continuation* continuation) const { auto cn = nodes_.find(continuation)->second; assert(cn); return cn->succs(); }
    const CFNode* entry() const { return entry_; }
    const CFNode* exit() const { return exit_; }
    const CFNode* node(Continuation*);

    const Scope& scope_;
    ContinuationMap<const CFNode*> nodes_;
    const CFNode* entry_;
    const CFNode* exit_;
    mutable std::unique_ptr<const F_CFG> f_cfg_;
    mutable std::unique_ptr<const B_CFG> b_cfg_;

    template<bool> friend class CFG;
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
class CFG {
public:
    template<class Value>
    using Map = IndexMap<CFG<forward>, const CFNode*, Value>;
    using Set = IndexSet<CFG<forward>, const CFNode*>;

    CFG(const CFG&) = delete;
    CFG& operator= (CFG) = delete;

    explicit CFG(const CFA&);

    const CFA& cfa() const { return cfa_; }
    size_t size() const { return cfa().size(); }
    const CFNodes& preds(const CFNode* n) const;
    const CFNodes& succs(const CFNode* n) const;
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

    static size_t index(const CFNode* n) { return forward ? n->f_index_ : n->b_index_; }

private:
    size_t post_order_visit(const CFNode* n, size_t i);

    const CFA& cfa_;
    Map<const CFNode*> rpo_;
    mutable std::unique_ptr<const DomTreeBase<forward>> domtree_;
    mutable std::unique_ptr<const LoopTree<forward>> looptree_;
};

//------------------------------------------------------------------------------

}

#endif
