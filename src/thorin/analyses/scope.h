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

    const Scope& update();
    ArrayRef<Continuation*> continuations() const { return continuations_; }
    Continuation* operator[](size_t i) const { return continuations_[i]; }
    Continuation* entry() const { return continuations().front(); }
    Continuation* exit() const { return continuations().back(); }
    ArrayRef<Continuation*> body() const { return continuations().skip_front(); } ///< Like @p continuations() but without \p entry()

    const DefSet& defs() const { return defs_; }
    bool contains(const Def* def) const { return defs_.contains(def); }
    bool inner_contains(Continuation* continuation) const { return continuation != entry() && contains(continuation); }
    bool inner_contains(const Param* param) const { return inner_contains(param->continuation()); }
    size_t size() const { return continuations_.size(); }
    World& world() const { return world_; }
    const CFA& cfa() const;
    const CFNode* cfa(Continuation*) const;
    const F_CFG& f_cfg() const;
    const B_CFG& b_cfg() const;
    template<bool forward> const CFG<forward>& cfg() const;

    // Note that we don't use overloading for the following methods in order to have them accessible from gdb.
    virtual std::ostream& stream(std::ostream&) const override;  ///< Streams thorin to file @p out.
    void write_thorin(const char* filename) const;               ///< Dumps thorin to file with name @p filename.
    void thorin() const;                                         ///< Dumps thorin to a file with an auto-generated file name.

    typedef ArrayRef<Continuation*>::const_iterator const_iterator;
    const_iterator begin() const { return continuations().begin(); }
    const_iterator end() const { return continuations().end(); }

    template<bool elide_empty = true>
    static void for_each(const World&, std::function<void(Scope&)>);

private:
    void run(Continuation* entry);

    World& world_;
    DefSet defs_;
    std::vector<Continuation*> continuations_;
    mutable std::unique_ptr<const CFA> cfa_;
};

template<> inline const CFG< true>& Scope::cfg< true>() const { return f_cfg(); }
template<> inline const CFG<false>& Scope::cfg<false>() const { return b_cfg(); }

}

#endif
