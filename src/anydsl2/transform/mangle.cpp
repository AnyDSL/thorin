#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

class Mangler {
public:

    Mangler(const Scope& scope, ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, 
           ArrayRef<const Def*> to_lift, const GenericMap& generic_map)
        : scope(scope)
        , to_drop(to_drop)
        , drop_with(drop_with)
        , to_lift(to_lift)
        , generic_map(generic_map)
        , world(scope.world())
        , pass(world.new_pass())
    {}

    Lambda* mangle();
    void mangle_body(Lambda* olambda, Lambda* nlambda);
    Lambda* mangle_head(Lambda* olambda);
    const Def* mangle(const Def* odef);
    const Def* map(const Def* def, const Def* to) {
        def->visit_first(pass);
        def->cptr = to;
        return to;
    }
    const Def* lookup(const Def* def) {
        assert(def->is_visited(pass));
        return (const Def*) def->cptr;
    }

    const Scope& scope;
    ArrayRef<size_t> to_drop;
    ArrayRef<const Def*> drop_with;
    ArrayRef<const Def*> to_lift;
    GenericMap generic_map;
    World& world;
    const size_t pass;
    Lambda* nentry;
    Lambda* oentry;
};

Lambda* Mangler::mangle() {
    assert(scope.num_entries() == 1 && "TODO");
    oentry = scope.entries()[0];
    const Pi* o_pi = oentry->pi();
    Array<const Type*> nelems = o_pi->elems().cut(to_drop, to_lift.size());
    size_t offset = o_pi->elems().size() - to_drop.size();

    for (size_t i = offset, e = nelems.size(), x = 0; i != e; ++i, ++x)
        nelems[i] = to_lift[x]->type();

    const Pi* n_pi = world.pi(nelems)->specialize(generic_map)->as<Pi>();
    nentry = world.lambda(n_pi, oentry->name);

    // put in params for entry (oentry)
    // op -> iterates over old params
    // np -> iterates over new params
    //  i -> iterates over to_drop
    for (size_t op = 0, np = 0, i = 0, e = o_pi->size(); op != e; ++op) {
        const Param* oparam = oentry->param(op);
        if (i < to_drop.size() && to_drop[i] == op)
            map(oparam, drop_with[i++]);
        else {
            const Param* nparam = nentry->param(np++);
            nparam->name = oparam->name;
            map(oparam, nparam);
        }
    }

    for (size_t i = offset, e = nelems.size(), x = 0; i != e; ++i, ++x) {
        map(to_lift[x], nentry->param(i));
        nentry->param(i)->name = to_lift[x]->name;
    }

    map(oentry, oentry);
    mangle_body(oentry, nentry);

    for (auto cur : scope.rpo().slice_back(1)) {
        if (cur->is_visited(pass))
            mangle_body(cur, lookup(cur)->as_lambda());
    }

    return nentry;
}

Lambda* Mangler::mangle_head(Lambda* olambda) {
    assert(!olambda->is_visited(pass));

    Lambda* nlambda = olambda->stub(generic_map, olambda->name);
    map(olambda, nlambda);

    for (size_t i = 0, e = olambda->num_params(); i != e; ++i)
        map(olambda->param(i), nlambda->param(i));

    return nlambda;
}

void Mangler::mangle_body(Lambda* olambda, Lambda* nlambda) {
    Array<const Def*> ops(olambda->ops().size());
    for (size_t i = 1, e = ops.size(); i != e; ++i)
        ops[i] = mangle(olambda->op(i));

    // fold branch if possible
    if (const Select* select = olambda->to()->isa<Select>()) {
        const Def* cond = mangle(select->cond());
        if (const PrimLit* lit = cond->isa<PrimLit>())
            ops[0] = mangle(lit->value().get_u1().get() ? select->tval() : select->fval());
        else
            ops[0] = world.select(cond, mangle(select->tval()), mangle(select->fval()));
    } else
        ops[0] = mangle(olambda->to());

    ArrayRef<const Def*> nargs(ops.slice_back(1));  // new args of nlambda
    const Def* ntarget = ops.front();               // new target of nlambda

    // check whether we can optimize tail recursion
    if (ntarget == oentry) {
        bool substitute = true;
        for (size_t i = 0, e = to_drop.size(); i != e && substitute; ++i)
            substitute &= nargs[to_drop[i]] == drop_with[i];

        if (substitute)
            return nlambda->jump(nentry, nargs.cut(to_drop));
    }

    nlambda->jump(ntarget, nargs);
}

const Def* Mangler::mangle(const Def* odef) {
    if (odef->is_visited(pass))
        return lookup(odef);

    if (Lambda* olambda = odef->isa_lambda()) {
        if (scope.contains(olambda))
            return mangle_head(olambda);
        else
            return map(odef, odef);
    } else if (odef->isa<Param>())
        return map(odef, odef);

    bool is_new = false;
    const PrimOp* oprimop = odef->as<PrimOp>();
    Array<const Def*> nops(oprimop->size());
    for (size_t i = 0, e = oprimop->size(); i != e; ++i) {
        const Def* nop = nops[i];
        const Def* op = oprimop->op(i);
        nop = mangle(op);
        is_new |= nop != op;
    }

    return map(oprimop, is_new ? world.rebuild(oprimop, nops) : oprimop);
}

//------------------------------------------------------------------------------

Lambda* mangle(const Scope& scope, ArrayRef<size_t> to_drop, ArrayRef<const Def*> drop_with, 
                       ArrayRef<const Def*> to_lift, const GenericMap& generic_map) {
    return Mangler(scope, to_drop, drop_with, to_lift, generic_map).mangle();
}

Lambda* drop(const Scope& scope, ArrayRef<const Def*> with) {
    size_t size = with.size();
    Array<size_t> to_drop(size);
    for (size_t i = 0; i != size; ++i)
        to_drop[i] = i;

    return mangle(scope, to_drop, with, Array<const Def*>(), GenericMap());
}

//------------------------------------------------------------------------------

}
