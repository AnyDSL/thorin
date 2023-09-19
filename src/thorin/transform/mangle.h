#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/type.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/rewrite.h"

namespace thorin {

class Mangler : Rewriter {
public:
    Mangler(const Scope& scope, Continuation* entry, Defs args, Defs lift);

    const Scope& scope() const { return scope_; }
    Continuation* mangle();
    Continuation* old_entry() const { return old_entry_; }
    Continuation* new_entry() const { return new_entry_; }

private:
    const Def* rewrite(const Def* odef) override;
    bool within(const Def* def) { return scope().contains(def) || defs_.contains(def); }

    bool is_dropping_;

    const Scope& scope_;
    Defs args_;
    Defs lift_;
    Continuation* old_entry_;
    Continuation* new_entry_;
    DefSet defs_;
};


Continuation* mangle(const Scope&, Continuation* entry, Defs args, Defs lift);

inline Continuation* drop(const Scope& scope, Defs args) {
    return mangle(scope, scope.entry(), args, Array<const Def*>());
}

Continuation* drop(const Def* callee, const Defs specialized_args);

inline Continuation* lift(const Scope& scope, Continuation* entry, Defs defs) {
    return mangle(scope, entry, Array<const Def*>(entry->num_params()), defs);
}

inline Continuation* clone(const Scope& scope) {
    return mangle(scope, scope.entry(), Array<const Def*>(scope.entry()->num_params()), Defs());
}

}

#endif
