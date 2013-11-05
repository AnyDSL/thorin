#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/scope.h"

namespace anydsl2 {

class Mangler {
public:
    Mangler(const Scope& scope, ArrayRef<size_t> to_drop, ArrayRef<Def> drop_with, 
           ArrayRef<Def> to_lift, const GenericMap& generic_map)
        : scope(scope)
        , to_drop(to_drop)
        , drop_with(drop_with)
        , to_lift(to_lift)
        , generic_map(generic_map)
        , world(scope.world())
        , pass1(scope.mark())
        , pass2(world.new_pass())
    {}

    Lambda* mangle();
    void mangle_body(Lambda* olambda, Lambda* nlambda);
    Lambda* mangle_head(Lambda* olambda);
    Def mangle(Def odef);
    Def map(Def def, Def to) {
        def->visit_first(pass2);
        def->cptr = *to;
        return to;
    }
    Def lookup(Def def) {
        assert(def->is_visited(pass2));
        return Def((const DefNode*) def->cptr);
    }

    const Scope& scope;
    ArrayRef<size_t> to_drop;
    ArrayRef<Def> drop_with;
    ArrayRef<Def> to_lift;
    GenericMap generic_map;
    World& world;
    const size_t pass1;
    const size_t pass2;
    Lambda* nentry;
    Lambda* oentry;
};

Lambda* Mangler::mangle() {
    assert(scope.num_entries() == 1 && "TODO");
    oentry = scope.entries()[0];
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
            map(oparam, drop_with[i++]);
        else {
            const Param* nparam = nentry->param(np++);
            nparam->name = oparam->name;
            map(oparam, nparam);
        }
    }

    // TODO mark users of to_lift
    for (size_t i = offset, e = nelems.size(), x = 0; i != e; ++i, ++x) {
        map(to_lift[x], nentry->param(i));
        nentry->param(i)->name = to_lift[x]->name;
    }

    map(oentry, oentry);
    mangle_body(oentry, nentry);

    for (auto cur : scope.rpo().slice_from_begin(1)) {
        if (cur->is_visited(pass2))
            mangle_body(cur, lookup(cur)->as_lambda());
    }

    return nentry;
}

Lambda* Mangler::mangle_head(Lambda* olambda) {
    assert(!olambda->is_visited(pass2));
    assert(!olambda->empty());
    Lambda* nlambda = olambda->stub(generic_map, olambda->name);
    map(olambda, nlambda);

    for (size_t i = 0, e = olambda->num_params(); i != e; ++i)
        map(olambda->param(i), nlambda->param(i));

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
            ops[0] = mangle(lit->value().get_u1().get() ? select->tval() : select->fval());
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
    if (odef->cur_pass() < pass1)
        return odef;
    if (odef->is_visited(pass2))
        return lookup(odef);

    if (auto olambda = odef->isa_lambda()) {
        assert(scope.contains(olambda));
        return mangle_head(olambda);
    } else if (auto param = odef->isa<Param>()) {
        assert(scope.contains(param->lambda()));
        return map(odef, odef);
    }

    auto oprimop = odef->as<PrimOp>();
    Array<Def> nops(oprimop->size());
    Eval eval = Eval::Infer;
    for (size_t i = 0, e = oprimop->size(); i != e; ++i) {
        auto op = mangle(oprimop->op(i));

        if (auto evalop = op->isa<EvalOp>()) {
            if (evalop->isa<Run>() && eval == Eval::Infer)
                eval = Eval::Run;
            else {
                assert(evalop->isa<Halt>());
                eval = Eval::Halt;
            }
            op = evalop->def();
        }

        nops[i] = op;
    }

    auto nprimop = world.rebuild(oprimop, nops);
    if (eval == Eval::Run) 
        nprimop = world.run(nprimop);
    else if (eval == Eval::Halt)
        nprimop = world.halt(nprimop);

    return map(oprimop, nprimop);
}

//------------------------------------------------------------------------------

Lambda* mangle(const Scope& scope, ArrayRef<size_t> to_drop, ArrayRef<Def> drop_with, 
               ArrayRef<Def> to_lift, const GenericMap& generic_map) {
    return Mangler(scope, to_drop, drop_with, to_lift, generic_map).mangle();
}

Lambda* drop(const Scope& scope, ArrayRef<Def> with) {
    size_t size = with.size();
    Array<size_t> to_drop(size);
    for (size_t i = 0; i != size; ++i)
        to_drop[i] = i;

    return mangle(scope, to_drop, with, Array<Def>(), GenericMap());
}

//------------------------------------------------------------------------------

}
