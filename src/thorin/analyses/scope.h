#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/lambda.h"
#include "thorin/util/array.h"
#include "thorin/util/stream.h"

namespace thorin {

template<bool> class CFG;
typedef CFG<true>  F_CFG;
typedef CFG<false> B_CFG;

class CFA;
class CFNode;

/**
 * A @p Scope represents a region of @p Lam%s which are live from the view of an @p entry @p Lam.
 * Transitively, all user's of the @p entry's parameters are pooled into this @p Scope.
 * @p entry() will be first, @p exit() will be last.
 * @warning All other @p Lam%s are in no particular order.
 */
class Scope : public Streamable<Scope> {
public:
    Scope(const Scope&) = delete;
    Scope& operator=(Scope) = delete;

    explicit Scope(Lam* entry);
    ~Scope();

    /// Invoke if you have modified sth in this Scope.
    Scope& update();

    //@{ misc getters
    World& world() const { return world_; }
    Lam* entry() const { return entry_; }
    Lam* exit() const { return exit_; }
    //@}

    //@{ get Def%s contained in this Scope
    const DefSet& defs() const { return defs_; }
    bool contains(const Def* def) const { return defs_.contains(def); }
    /// All @p Def%s referenced but @em not contained in this @p Scope.
    const DefSet& free() const;
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
     * Select with @p elide_empty whether you want to visit trivial Scope%s of Lam%s without body.
     */
    template<bool elide_empty = true>
    static void for_each(const World&, std::function<void(Scope&)>);

private:
    void run();

    World& world_;
    DefSet defs_;
    Lam* entry_ = nullptr;
    Lam* exit_ = nullptr;
    mutable std::unique_ptr<DefSet> free_;
    mutable std::unique_ptr<ParamSet> free_params_;
    mutable std::unique_ptr<const CFA> cfa_;
};

}

#endif
