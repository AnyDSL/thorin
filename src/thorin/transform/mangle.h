#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/lam.h"
#include "thorin/analyses/scope.h"

namespace thorin {

class Mangler {
public:
    Mangler(const Scope& scope, Defs args, Defs lift);

    const Scope& scope() const { return scope_; }
    World& world() const { return scope_.world(); }
    std::optional<const Def*> old2new(const Def* def) { return old2new_.lookup(def); }
    Lam* mangle();
    Lam* old_entry() const { return old_entry_; }
    Lam* new_entry() const { return new_entry_; }

private:
    const Def* mangle(const Def*);
    bool within(const Def* def) { return scope().contains(def) || defs_.contains(def); }

    const Scope& scope_;
    Defs args_;
    Lam* old_entry_;
    Lam* new_entry_;
    DefSet defs_;
    Def2Def old2new_;
};

Lam* mangle(const Scope&, Defs args, Defs lift);

inline Lam* drop(const Scope& scope, Defs args) {
    return mangle(scope, args, Array<const Def*>());
}

Lam* drop(const App*);

inline Lam* lift(const Scope& scope, Defs defs) {
    return mangle(scope, Array<const Def*>(scope.entry()->num_vars()), defs);
}

inline Lam* clone(const Scope& scope) {
    return mangle(scope, Array<const Def*>(scope.entry()->num_vars()), Defs());
}

}

#endif
