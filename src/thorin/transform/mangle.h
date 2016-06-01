#ifndef THORIN_TRANSFORM_MANGLE_H
#define THORIN_TRANSFORM_MANGLE_H

#include "thorin/type.h"
#include "thorin/analyses/scope.h"

namespace thorin {

class Mangler {
public:
    Mangler(const Scope& scope, Types type_args, Defs args, Defs lift);

    const Scope& scope() const { return scope_; }
    World& world() const { return scope_.world(); }
    const Def* def2def(const Def* def) { return find(def2def_, def); }
    Continuation* mangle();
    Continuation* old_entry() const { return old_entry_; }
    Continuation* new_entry() const { return new_entry_; }

private:
    void mangle_body(Continuation* ocontinuation, Continuation* ncontinuation);
    Continuation* mangle_head(Continuation* ocontinuation);
    const Def* mangle(const Def* odef);
    bool within(const Def* def) { return scope().contains(def) || defs_.contains(def); }

    const Scope& scope_;
    Def2Def def2def_;
    Types type_args_;
    Defs args_;
    Defs lift_;
    Type2Type type2type_;
    Continuation* old_entry_;
    Continuation* new_entry_;
    DefSet defs_;
};


Continuation* mangle(const Scope&, Types type_args, Defs args, Defs lift);

inline Continuation* drop(const Scope& scope, Types type_args, Defs args) {
    return mangle(scope, type_args, args, Array<const Def*>());
}

Continuation* drop(const Call&);

inline Continuation* lift(const Scope& scope, Types type_args, Defs defs) {
    return mangle(scope, type_args, Array<const Def*>(scope.entry()->num_params()), defs);
}

inline Continuation* clone(const Scope& scope) {
    return mangle(scope, Array<const Type*>(scope.entry()->num_type_params()),
                         Array<const Def*>(scope.entry()->num_params()), Defs());
}

}

#endif
