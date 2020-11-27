#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/def.h"
#include "thorin/util/array.h"
#include "thorin/util/stream.h"

namespace thorin {

class CFA;
template<bool> class CFG;
using F_CFG = CFG<true >;
using B_CFG = CFG<false>;

/**
 * A @p Scope represents a region of @em nominals which are live from the view of an @p entry @em nominal.
 * Transitively, all user's of the @p entry's @p Param%s are pooled into this @p Scope.
 * Both @p entry() and @p exit() are @em NOT part of the @p Scope itself - but their @p Param%s.
 */
class Scope : public Streamable<Scope> {
public:
    Scope(const Scope&) = delete;
    Scope& operator=(Scope) = delete;

    explicit Scope(Def* entry);
    ~Scope();

    /// Invoke if you have modified sth in this Scope.
    Scope& update();
    /// @name getters
    //@{
    World& world() const { return world_; }
    Def* entry() const { return entry_; }
    Def* exit() const { return exit_; }
    std::string name() const { return entry_->debug().name; }
    //@}
    /// @name get Def%s contained in this Scope
    //@{
    const DefSet& defs() const { return defs_; }
    bool contains(const Def* def) const { return defs_.contains(def); }
    /// All @p Def%s referenced but @em not contained in this @p Scope.
    const DefSet& free() const;
    /// All @p Param%s that appear free in this @p Scope.
    const ParamSet& free_params() const;
    /// All @em nominals that appear free in this @p Scope.
    const NomSet& free_noms() const;
    /// Are there any free @p Param%s within this @p Scope.
    bool has_free_params() const { return !free_params().empty(); }
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

    World& world_;
    DefSet defs_;
    Def* entry_ = nullptr;
    Def* exit_ = nullptr;
    mutable std::unique_ptr<DefSet> free_;
    mutable std::unique_ptr<ParamSet> free_params_;
    mutable std::unique_ptr<NomSet> free_noms_;
    mutable std::unique_ptr<const CFA> cfa_;
};

bool is_free(const Param* param, const Def* def);

}

#endif
