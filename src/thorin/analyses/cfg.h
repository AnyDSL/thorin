#ifndef THORIN_ANALYSES_CFG_H
#define THORIN_ANALYSES_CFG_H

#include <vector>

#include "thorin/analyses/scope.h"
#include "thorin/util/array.h"
#include "thorin/util/indexmap.h"
#include "thorin/util/indexset.h"
#include "thorin/util/streamf.h"

namespace thorin {

//------------------------------------------------------------------------------

class CFNode;
template<bool> class LoopTree;
template<bool> class DomTreeBase;
template<bool> class DomFrontierBase;

typedef GIDSet<const CFNode*> CFNodes;

/**
 * A Control-Flow Node.
 * Managed by @p CFA.
 */
class CFNode : public Streamable {
public:
    CFNode(Def* nom)
        : nominal_(nom)
        , gid_(gid_counter_++)
    {}

    uint64_t gid() const { return gid_; }
    Def* nominal() const { return nominal_; }
    std::ostream& stream(std::ostream& os) const override;

private:
    const CFNodes& preds() const { return preds_; }
    const CFNodes& succs() const { return succs_; }
    void link(const CFNode* other) const;

    mutable size_t f_index_ = -1; ///< RPO index in a forward @p CFG.
    mutable size_t b_index_ = -1; ///< RPO index in a backwards @p CFG.

    Def* nominal_;
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
    const NomMap<const CFNode*>& nodes() const { return nodes_; }
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    const CFNode* operator[](Def* nom) const { return nodes_.lookup(nom).value_or(nullptr); }

private:
    void link_to_exit();
    void verify();
    const CFNodes& preds(Def* nom) const { auto cn = nodes_.find(nom)->second; assert(cn); return cn->preds(); }
    const CFNodes& succs(Def* nom) const { auto cn = nodes_.find(nom)->second; assert(cn); return cn->succs(); }
    const CFNode* entry() const { return entry_; }
    const CFNode* exit() const { return exit_; }
    const CFNode* node(Def*);

    const Scope& scope_;
    NomMap<const CFNode*> nodes_;
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
    const CFNodes& preds(Def* nom) const { return preds(cfa()[nom]); }
    const CFNodes& succs(Def* nom) const { return succs(cfa()[nom]); }
    size_t num_preds(const CFNode* n) const { return preds(n).size(); }
    size_t num_succs(const CFNode* n) const { return succs(n).size(); }
    size_t num_preds(Def* nom) const { return num_preds(cfa()[nom]); }
    size_t num_succs(Def* nom) const { return num_succs(cfa()[nom]); }
    const CFNode* entry() const { return forward ? cfa().entry() : cfa().exit();  }
    const CFNode* exit()  const { return forward ? cfa().exit()  : cfa().entry(); }

    ArrayRef<const CFNode*> reverse_post_order() const { return rpo_.array(); }
    Range<ArrayRef<const CFNode*>::const_reverse_iterator> post_order() const { return reverse_range(rpo_.array()); }
    const CFNode* reverse_post_order(size_t i) const { return rpo_.array()[i]; }  ///< Maps from reverse post-order index to @p CFNode.
    const CFNode* post_order(size_t i) const { return rpo_.array()[size()-1-i]; } ///< Maps from post-order index to @p CFNode.
    const CFNode* operator [] (Def* nom) const { return cfa()[nom]; }    ///< Maps from @p l to @p CFNode.
    const DomTreeBase<forward>& domtree() const;
    const LoopTree<forward>& looptree() const;
    const DomFrontierBase<forward>& domfrontier() const;

    static size_t index(const CFNode* n) { return forward ? n->f_index_ : n->b_index_; }

private:
    size_t post_order_visit(const CFNode* n, size_t i);

    const CFA& cfa_;
    Map<const CFNode*> rpo_;
    mutable std::unique_ptr<const DomTreeBase<forward>> domtree_;
    mutable std::unique_ptr<const LoopTree<forward>> looptree_;
    mutable std::unique_ptr<const DomFrontierBase<forward>> domfrontier_;
};

//------------------------------------------------------------------------------

}

#endif
