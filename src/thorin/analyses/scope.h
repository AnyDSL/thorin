#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include "thorin/def.h"
#include "thorin/util/stream.h"

namespace thorin {

class CFA;
template<bool> class CFG;
using F_CFG = CFG<true >;
using B_CFG = CFG<false>;

/**
 * A @p Scope represents a region of @p Def%s that are live from the view of an @p entry's @p Var.
 * Transitively, all user's of the @p entry's @p Var are pooled into this @p Scope (see @p defs()).
 * Both @p entry() and @p exit() are @em NOT part of the @p Scope itself.
 * The @p exit() is just a virtual dummy to have a unique exit dual to @p entry().
 */
class Scope : public Streamable<Scope> {
public:
    Scope(const Scope&) = delete;
    Scope& operator=(Scope) = delete;

    explicit Scope(Def* entry);
    ~Scope();

    /// @name getters
    //@{
    World& world() const { return world_; }
    Def* entry() const { return entry_; }
    Def* exit() const { return exit_; }
    std::string name() const { return entry_->debug().name; }
    //@}

    /// @name Def%s bound/free in this Scope
    //@{
    bool bound(const Def* def) const { return bound().contains(def); }
    const DefSet& bound()     const { calc_bound(); return bound_;     } ///< All @p Def%s within this @p Scope.
    const DefSet& free_defs() const { calc_bound(); return free_defs_; } ///< All @em non-const @p Def%s @em directly referenced but @em not @p bound within this @p Scope. May also include @p Var%s or @em noms.
    const VarSet& free_vars() const { calc_free (); return free_vars_; } ///< All @p Var%s that occurr free in this @p Scope. Does @em not transitively contain any free @p Var%s from @p noms.
    const NomSet& free_noms() const { calc_free (); return free_noms_; } ///< All @em noms that occurr free in this @p Scope.
    //@}

    /// @name simple CFA to construct a CFG
    //@{
    const CFA& cfa() const;
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    //@}

    Stream& stream(Stream&) const;

private:
    void run();
    void calc_bound() const;
    void calc_free() const;

    World& world_;
    Def* entry_ = nullptr;
    Def* exit_  = nullptr;
    mutable bool has_bound_ = false;
    mutable bool has_free_  = false;
    mutable DefSet bound_;
    mutable DefSet free_defs_;
    mutable VarSet free_vars_;
    mutable NomSet free_noms_;
    mutable std::unique_ptr<const CFA> cfa_;
};

/// Does @p var occurr free in @p def?
bool is_free(const Var* var, const Def* def);

}

#endif
