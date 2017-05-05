#ifndef THORIN_ANALYSES_SCOPE_H
#define THORIN_ANALYSES_SCOPE_H

#include <vector>

#include "thorin/continuation.h"
#include "thorin/util/array.h"
#include "thorin/util/iterator.h"
#include "thorin/util/stream.h"

namespace thorin {

template<bool> class CFG;
typedef CFG<true>  F_CFG;
typedef CFG<false> B_CFG;

class CFA;
class CFNode;
class Nest;

/**
 * A @p Scope represents a region of @p Continuation%s which are live from the view of an @p entry @p Continuation.
 * Transitively, all user's of the @p entry's parameters are pooled into this @p Scope.
 */
class Scope : public Streamable {
public:
    Scope(const Scope&) = delete;
    Scope& operator=(Scope) = delete;

    explicit Scope(Continuation* entry);
    ~Scope();

    /// Invoke if you have modified sth in this Scope.
    const Scope& update();

    //@{ misc getters
    World& world() const { return world_; }
    Continuation* entry() const { return continuations_.front(); }
    Continuation* exit() const { return continuations_.back(); }
    /**
     * All continuations in this Scope.
     * entry is first, exit ist last.
     * @attention { All other Continuation%s are in @em no special order. }
     */
    ArrayRef<Continuation*> continuations() const { return continuations_; }
    const Nest& nest() const;
    //@}

    //@{ get Def%s contained in this Scope
    const DefSet& defs() const { return defs_; }
    bool contains(const Def* def) const { return defs_.contains(def); }
    bool inner_contains(Continuation* continuation) const { return continuation != entry() && contains(continuation); }
    bool inner_contains(const Param* param) const { return inner_contains(param->continuation()); }
    //@}

    //@{ get CFG
    const CFA& cfa() const;
    const CFNode* cfa(Continuation*) const;
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    template<bool forward> const CFG<forward>& cfg() const;
    //@}

    //@{ dump
    // Note that we don't use overloading for the following methods in order to have them accessible from gdb.
    virtual std::ostream& stream(std::ostream&) const override;  ///< Streams thorin to file @p out.
    void write_thorin(const char* filename) const;               ///< Dumps thorin to file with name @p filename.
    void thorin() const;                                         ///< Dumps thorin to a file with an auto-generated file name.
    //@}

    /**
     * Transitively visits all @em reachable Scope%s in @p world that do not have free variables.
     * We call these Scope%s @em top-level Scope%s.
     * Select with @p elide_empty whether you want to visit trivial Scope%s of Continuation%s without body.
     */
    template<bool elide_empty = true>
    static void for_each(const World& world, std::function<void(Scope&)>);

private:
    void run(Continuation* entry);
    void nesting();

    World& world_;
    DefSet defs_;
    std::vector<Continuation*> continuations_;
    mutable std::unique_ptr<const Nest> nest_;
    mutable std::unique_ptr<const CFA> cfa_;
};

template<> inline const CFG< true>& Scope::cfg< true>() const { return f_cfg(); }
template<> inline const CFG<false>& Scope::cfg<false>() const { return b_cfg(); }

}

#endif
