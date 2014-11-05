#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

class Mangler {
public:
    Mangler(const Scope& scope, ArrayRef<Def> drop, ArrayRef<Def> lift, const Type2Type& type2type)
        : scope(scope)
        , drop(drop)
        , lift(lift)
        , type2type(type2type)
        , world(scope.world())
        , in_scope(scope.in_scope()) // copy constructor
        , oentry(scope.entry())
        , nentry(oentry->world().lambda(oentry->name))
    {
        assert(!oentry->empty());
        assert(drop.size() == oentry->num_params());
        std::queue<Def> queue;
        for (auto def : lift)
            queue.push(def);

        while (!queue.empty()) {
            for (auto use : pop(queue)->uses()) {
                if (!use->isa_lambda() && !visit(in_scope, use))
                    queue.push(use);
            }
        }
    }

    Lambda* mangle();
    void mangle_body(Lambda* olambda, Lambda* nlambda);
    Lambda* mangle_head(Lambda* olambda);
    Def mangle(Def odef);
    Def lookup(Def def) {
        assert(old2new.contains(def));
        return old2new[def];
    }

    const Scope& scope;
    Def2Def old2new;
    ArrayRef<Def> drop;
    ArrayRef<Def> lift;
    Type2Type type2type;
    World& world;
    DefSet in_scope;
    Lambda* oentry;
    Lambda* nentry;
};

Lambda* Mangler::mangle() {
    old2new[oentry] = oentry;
    for (size_t i = 0, e = oentry->num_params(); i != e; ++i) {
        auto oparam = oentry->param(i);
        if (auto def = drop[i])
            old2new[oparam] = def;
        else
            old2new[oparam] = nentry->append_param(oparam->type()->specialize(type2type), oparam->name);
    }

    for (auto def : lift)
        old2new[def] = nentry->append_param(def->type()->specialize(type2type));

    mangle_body(oentry, nentry);
    return nentry;
}

Lambda* Mangler::mangle_head(Lambda* olambda) {
    assert(!old2new.contains(olambda));
    assert(!olambda->empty());
    Lambda* nlambda = olambda->stub(type2type, olambda->name);
    old2new[olambda] = nlambda;

    for (size_t i = 0, e = olambda->num_params(); i != e; ++i)
        old2new[olambda->param(i)] = nlambda->param(i);

    return nlambda;
}

void Mangler::mangle_body(Lambda* olambda, Lambda* nlambda) {
    assert(!olambda->empty());
    
    if (olambda->to() == world.branch()) {          // fold branch if possible
        if (auto lit = mangle(olambda->arg(0))->isa<PrimLit>())
            return nlambda->jump(mangle(lit->value().get_bool() ? olambda->arg(1) : olambda->arg(2)), {});
    }

    Array<Def> nops(olambda->ops().size());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = mangle(olambda->op(i));

    ArrayRef<Def> nargs(nops.slice_from_begin(1));  // new args of nlambda
    Def ntarget = nops.front();                     // new target of nlambda

    // check whether we can optimize tail recursion
    if (ntarget == oentry) {
        std::vector<size_t> cut;
        bool substitute = true;
        for (size_t i = 0, e = drop.size(); i != e && substitute; ++i) {
            if (auto def = drop[i]) {
                substitute &= def == nargs[i];
                cut.push_back(i);
            }
        }

        if (substitute)
            return nlambda->jump(nentry, nargs.cut(cut));
    }

    nlambda->jump(ntarget, nargs);
}

Def Mangler::mangle(Def odef) {
    if (!in_scope.contains(odef) && !old2new.contains(odef))
        return odef;
    if (old2new.contains(odef))
        return lookup(odef);

    if (auto olambda = odef->isa_lambda()) {
        auto nlambda = mangle_head(olambda);
        mangle_body(olambda, nlambda);
        return nlambda;
    } else if (auto param = odef->isa<Param>()) {
        assert(scope.contains(param->lambda()));
        return old2new[odef] = odef;
    }

    auto oprimop = odef->as<PrimOp>();
    Array<Def> nops(oprimop->size());
    Def nprimop;

    for (size_t i = 0, e = oprimop->size(); i != e; ++i)
        nops[i] = mangle(oprimop->op(i));
    nprimop = oprimop->rebuild(nops);
    return old2new[oprimop] = nprimop;
}

//------------------------------------------------------------------------------

Lambda* mangle(const Scope& scope, ArrayRef<Def> drop, ArrayRef<Def> lift, const Type2Type& type2type) {
    return Mangler(scope, drop, lift, type2type).mangle();
}

//------------------------------------------------------------------------------

}
