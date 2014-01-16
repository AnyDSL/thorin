#include "thorin/literal.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

#include <iostream>

namespace thorin {

class Mangler {
public:
    Mangler(const Scope& scope,
            Def2Def& old2new,
            ArrayRef<size_t> to_drop,
            ArrayRef<Def> drop_with,
            ArrayRef<Def> to_lift,
            const GenericMap& generic_map)
        : scope(scope)
        , to_drop(to_drop)
        , drop_with(drop_with)
        , to_lift(to_lift)
        , generic_map(generic_map)
        , world(scope.world())
        , set(scope.in_scope())
        , map(old2new)
    {
        std::queue<Def> queue;
        for (auto def : to_lift)
            queue.push(def);

        while (!queue.empty()) {
            auto def = queue.front();
            queue.pop();

            for (auto use : def->uses()) {
                if (!use->isa_lambda() && !set.visit(use))
                    queue.push(use);
            }
        }
    }

    Lambda* mangle();
    void mangle_body(Lambda* olambda, Lambda* nlambda);
    Lambda* mangle_head(Lambda* olambda);
    Def mangle(Def odef);
    Def lookup(Def def) {
        assert(map.contains(def));
        return map[def];
    }

    const Scope& scope;
    ArrayRef<size_t> to_drop;
    ArrayRef<Def> drop_with;
    ArrayRef<Def> to_lift;
    GenericMap generic_map;
    World& world;
    DefSet set;
    Def2Def& map;
    Lambda* nentry;
    Lambda* oentry;
};

Lambda* Mangler::mangle() {
    oentry = scope.entry();
    assert(!oentry->empty());
    const Pi* o_pi = oentry->pi();
    auto nelems = o_pi->elems().cut(to_drop, to_lift.size());
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
            map[oparam] = drop_with[i++];
        else {
            const Param* nparam = nentry->param(np++);
            nparam->name = oparam->name;
            map[oparam] = nparam;
        }
    }

    for (size_t i = offset, e = nelems.size(), x = 0; i != e; ++i, ++x) {
        map[to_lift[x]] = nentry->param(i);
        nentry->param(i)->name = to_lift[x]->name;
    }

    map[oentry] = oentry;
    mangle_body(oentry, nentry);

    for (auto cur : scope.rpo().slice_from_begin(1)) {
        if (map.contains(cur))
            mangle_body(cur, lookup(cur)->as_lambda());
        else
            map[cur] = cur;
    }

    return nentry;
}

Lambda* Mangler::mangle_head(Lambda* olambda) {
    assert(!map.contains(olambda));
    assert(!olambda->empty());
    Lambda* nlambda = olambda->stub(generic_map, olambda->name);
    map[olambda] = nlambda;
    //std::cout << "map: " << olambda->unique_name() << " -> " << nlambda->unique_name() << std::endl;

    for (size_t i = 0, e = olambda->num_params(); i != e; ++i)
        map[olambda->param(i)] = nlambda->param(i);

    return nlambda;
}

void Mangler::mangle_body(Lambda* olambda, Lambda* nlambda) {
    assert(!olambda->empty());
    Array<Def> ops(olambda->ops().size());
    for (size_t i = 1, e = ops.size(); i != e; ++i)
        ops[i] = mangle(olambda->op(i));

    // fold branch if possible
    if (auto select = olambda->to()->isa<Select>()) {
        Def cond = mangle(select->cond());
        if (auto lit = cond->isa<PrimLit>())
            ops[0] = mangle(lit->value().get_bool() ? select->tval() : select->fval());
        else
            ops[0] = mangle(select); //world.select(cond, mangle(select->tval()), mangle(select->fval()));
    } else
        ops[0] = mangle(olambda->to());

    ArrayRef<Def> nargs(ops.slice_from_begin(1));// new args of nlambda
    Def ntarget = ops.front();                   // new target of nlambda

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

enum class Eval { Run, Infer, Halt };

Def Mangler::mangle(Def odef) {
    if (!set.contains(odef) && !map.contains(odef))
        return odef;
    if (map.contains(odef))
        return lookup(odef);

    if (auto olambda = odef->isa_lambda()) {
        assert(scope.contains(olambda));
        return mangle_head(olambda);
    } else if (auto param = odef->isa<Param>()) {
        assert(scope.contains(param->lambda()));
        return map[odef] = odef;
    }

    auto oprimop = odef->as<PrimOp>();
    Array<Def> nops(oprimop->size());
    Def nprimop;

    if (oprimop->isa<Aggregate>()) {
        for (size_t i = 0, e = oprimop->size(); i != e; ++i)
            nops[i] = mangle(oprimop->op(i));
        nprimop = world.rebuild(oprimop, nops);
    } else {
        Eval eval = Eval::Infer;
        for (size_t i = 0, e = oprimop->size(); i != e; ++i) {
            auto op = mangle(oprimop->op(i));

            if (auto evalop = op->isa<EvalOp>()) {
                if (evalop->isa<Run>()) {
                    if (eval == Eval::Run || eval == Eval::Infer)
                        eval = Eval::Run;
                    else
                        goto halt_mode;
                } else {
                halt_mode:
                    assert(evalop->isa<Halt>());
                    eval = Eval::Halt;
                }
                op = evalop->def();
            }

            nops[i] = op;
        }

        nprimop = world.rebuild(oprimop, nops);
        if (eval == Eval::Run)
            nprimop = world.run(nprimop);
        else if (eval == Eval::Halt)
            nprimop = world.halt(nprimop);
    }
    return map[oprimop] = nprimop;
}

//------------------------------------------------------------------------------

Lambda* mangle(const Scope& scope,
               Def2Def& old2new,
               ArrayRef<size_t> to_drop,
               ArrayRef<Def> drop_with,
               ArrayRef<Def> to_lift,
               const GenericMap& generic_map) {
    return Mangler(scope, old2new, to_drop, drop_with, to_lift, generic_map).mangle();
}

Lambda* drop(const Scope& scope, Def2Def& old2new, ArrayRef<Def> with) {
    size_t size = with.size();
    Array<size_t> to_drop(size);
    for (size_t i = 0; i != size; ++i)
        to_drop[i] = i;

    return mangle(scope, old2new, to_drop, with, Array<Def>(), GenericMap());
}

//------------------------------------------------------------------------------

}
