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
    explicit Scope(Continuation* entry, std::shared_ptr<ScopesForest>);
    ~Scope();

    /// Invoke if you have modified sth in this Scope.
    Scope& update();

    //@{ misc getters
    World& world() const { return world_; }
    Continuation* entry() const { return entry_; }
    //@}

    //@{ get Def%s contained in this Scope
    const DefSet& defs() const { return defs_; }
    bool contains(const Def* def) const { return defs_.contains(def); }
    /// All @p Param%s that appear free in this @p Scope.
    const ParamSet& free_params() const;
    /// Are there any free @p Param%s within this @p Scope.
    bool has_free_params() const { return !free_params().empty(); }
    //@}

    //@{ simple CFA to construct a CFG
    const CFA& cfa() const;
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    //@}

    /// @name logging
    //@{
    Stream& stream(Stream&) const;                  ///< Streams thorin to file @p out.
    //@}

    /**
     * Transitively visits all @em reachable Scope%s in @p world that do not have free variables.
     * We call these Scope%s @em top-level Scope%s.
     * Select with @p elide_empty whether you want to visit trivial Scope%s of Continuation%s without body.
     */
    template<bool elide_empty = true>
    static void for_each(const World&, std::function<void(Scope&)>);

//private:
    void run();
    DefSet potentially_contained() const;

    ParamSet search_free_variables_nonrec(bool) const;

    World& world_;
    mutable std::shared_ptr<ScopesForest> forest_;
    DefSet defs_;
    Continuation* entry_ = nullptr;
    DefSet free_frontier_;
    mutable std::unique_ptr<ParamSet> free_params_;
    mutable std::unique_ptr<const CFA> cfa_;

    friend ScopesForest;
};

class ScopesForest {
public:
    ScopesForest() {}

    Scope& get_scope(Continuation* entry, std::shared_ptr<ScopesForest>& self);

    std::vector<Continuation*> stack_;
    ContinuationMap<std::unique_ptr<Scope>> scopes_;
};

}

#endif
