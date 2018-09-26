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

/**
 * A @p Scope represents a region of @p Continuation%s which are live from the view of an @p entry @p Continuation.
 * Transitively, all user's of the @p entry's parameters are pooled into this @p Scope.
 * Use @p continuations() to retrieve a vector of @p Continuation%s in this @p Scope.
 * @p entry() will be first, @p exit() will be last.
 * @warning All other @p Continuation%s are in no particular order.
 */
class Scope : public Streamable {
public:
    Scope(const Scope&) = delete;
    Scope& operator=(Scope) = delete;

    explicit Scope(Continuation* entry);
    ~Scope();

    /// Invoke if you have modified sth in this Scope.
    Scope& update();

    //@{ misc getters
    World& world() const { return world_; }
    Continuation* entry() const { return continuations().front(); }
    Continuation* exit() const { return continuations().back(); }
    /**
     * All continuations in this Scope.
     * entry is first, exit ist last.
     * @attention { All other Continuation%s are in @em no special order. }
     */
    ArrayRef<Continuation*> continuations() const { return continuations_; }
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

    size_t size() const { return continuations_.size(); }

    //@{ simple CFA to construct a CFG
    const CFA& cfa() const;
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    //@}

    //@{ dump
    // Note that we don't use overloading for the following methods in order to have them accessible from gdb.
    virtual std::ostream& stream(std::ostream&) const override;  ///< Streams thorin to file @p out.
    void write_thorin(const char* filename) const;               ///< Dumps thorin to file with name @p filename.
    void thorin() const;                                         ///< Dumps thorin to a file with an auto-generated file name.
    //@}

    typedef ArrayRef<Continuation*>::const_iterator const_iterator;
    const_iterator begin() const { return continuations().begin(); }
    const_iterator end() const { return continuations().end(); }

    /**
     * Transitively visits all @em reachable Scope%s in @p world that do not have free variables.
     * We call these Scope%s @em top-level Scope%s.
     * Select with @p elide_empty whether you want to visit trivial Scope%s of Continuation%s without body.
     */
    template<bool elide_empty = true>
    static void for_each(const World&, std::function<void(Scope&)>);

private:
    void run(Continuation* entry);

    World& world_;
    DefSet defs_;
    std::vector<Continuation*> continuations_;
    mutable std::unique_ptr<DefSet> free_;
    mutable std::unique_ptr<ParamSet> free_params_;
    mutable std::unique_ptr<const CFA> cfa_;
};

}

#endif
