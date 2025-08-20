#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/continuation.h"
#include "thorin/util/array.h"
#include "thorin/util/stream.h"

namespace thorin {

template<bool> class CFG;
typedef CFG<true>  F_CFG;
typedef CFG<false> B_CFG;

class CFA;
class CFNode;

class ScopesForest;

/**
 * A @p Scope represents a region of @p Continuation%s which are live from the view of an @p entry @p Continuation.
 * Transitively, all user's of the @p entry's parameters are pooled into this @p Scope.
 * @p entry() will be first, @p exit() will be last.
 * @warning All other @p Continuation%s are in no particular order.
 */
class Scope : public Streamable<Scope> {
public:
    Scope(const Scope&) = delete;
    Scope& operator=(Scope) = delete;

    explicit Scope(Continuation* entry);
    explicit Scope(Continuation* entry, ScopesForest&);
    ~Scope();

    /// Invoke if you have modified sth in this Scope.
    Scope& update();

    //@{ misc getters
    World& world() const { return world_; }
    Continuation* entry() const { return entry_; }
    ScopesForest& forest() const { return forest_; }
    //@}

    //@{ get Def%s contained in this Scope
    const DefSet& defs() const { return defs_; }
    const DefSet& free_frontier() const { return free_frontier_; }
    bool contains(const Def* def) const { return defs_.contains(def); }
    //@}

    /// All @p Param%s that appear free in this @p Scope.
    const ParamSet& free_params() const;
    /// Are there any free @p Param%s within this @p Scope.
    bool has_free_params() const;
    const Param* first_free_param() const;

    Continuation* parent_scope() const;
    ContinuationSet children_scopes() const;

    //@{ simple CFA to construct a CFG
    const CFA& cfa() const;
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    //@}

    /// @name logging
    //@{
    Stream& stream(Stream&) const;                  ///< Streams thorin to file @p out.
    //@}

    void verify();
private:
    void run();
    DefSet potentially_contained() const;

    template<bool stop_after_first>
    std::tuple<ParamSet, bool> search_free_params() const;

    World& world_;
    std::unique_ptr<ScopesForest> root;
    ScopesForest& forest_;
    Continuation* entry_ = nullptr;
    DefSet defs_;
    DefSet free_frontier_;
    mutable std::optional<const Param*> first_free_param_;
    mutable std::unique_ptr<ParamSet> free_params_;
    mutable std::unique_ptr<const CFA> cfa_;
    mutable std::optional<Continuation*> parent_scope_;

    friend ScopesForest;
};

class ScopesForest {
public:
    ScopesForest(World& world) : world_(world) {}

    Scope& get_scope(Continuation* entry);

    ContinuationSet top_level_scopes();

    std::vector<Continuation*> parent_scopes_path(Continuation*);
    Continuation* least_common_ancestor(Continuation* a, Continuation* b);

    /**
     * Transitively visits all @em reachable Scope%s in @p world that do not have free variables.
     * We call these Scope%s @em top-level Scope%s.
     * Select with @p elide_empty whether you want to visit trivial Scope%s of Continuation%s without body.
     */
    template<bool elide_empty = true>
    void for_each(std::function<void(Scope&)>);

private:
    World& world_;
    std::vector<Continuation*> stack_;
    ContinuationMap<std::unique_ptr<Scope>> scopes_;

    friend Scope;
};

}

#endif
