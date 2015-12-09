#include "thorin/transform/mangle.h"

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

class Mangler {
public:
    Mangler(const Scope& scope, ArrayRef<Type> type_args, ArrayRef<Def> args, ArrayRef<Def> lift)
        : scope(scope)
        , type_args(type_args)
        , args(args)
        , lift(lift)
        , in_scope(scope.in_scope()) // copy constructor
        , oentry(scope.entry())
        , nentry(oentry->world().lambda(oentry->loc(), oentry->name))
    {
        assert(!oentry->empty());
        assert(args.size() == oentry->num_params());
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

    World& world() const { return scope.world(); }
    Lambda* mangle();
    void mangle_body(Lambda* olambda, Lambda* nlambda);
    Lambda* mangle_head(Lambda* olambda);
    Def mangle(Def odef);

    const Scope& scope;
    Def2Def old2new;
    ArrayRef<Type> type_args;
    ArrayRef<Def> args;
    ArrayRef<Def> lift;
    Type2Type type2type;
    DefSet in_scope;
    Lambda* oentry;
    Lambda* nentry;
};

Lambda* Mangler::mangle() {
    old2new[oentry] = oentry;

#if 0
    for (size_t i = 0, e = oentry->num_type_params(); i != e; ++i) {
        auto otype_param = oentry->type_param(i);
        if (auto type_param = type_params[i])
            type2type[otype_param] = type_param;
        else
            type2type[oparam] = nentry->append_param(oparam->type()->specialize(type2type), oparam->name);
    }
#endif

    for (size_t i = 0, e = oentry->num_params(); i != e; ++i) {
        auto oparam = oentry->param(i);
        if (auto def = args[i])
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

    if (olambda->to() == world().branch()) {        // fold branch if possible
        if (auto lit = mangle(olambda->arg(0))->isa<PrimLit>())
            return nlambda->jump(mangle(lit->value().get_bool() ? olambda->arg(1) : olambda->arg(2)), {}, {});
    }

    Array<Def> nops(olambda->ops().size());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = mangle(olambda->op(i));

    ArrayRef<Def> nargs(nops.skip_front());         // new args of nlambda
    Def ntarget = nops.front();                     // new target of nlambda

    // check whether we can optimize tail recursion
    if (ntarget == oentry) {
        std::vector<size_t> cut;
        bool substitute = true;
        for (size_t i = 0, e = args.size(); i != e && substitute; ++i) {
            if (auto def = args[i]) {
                substitute &= def == nargs[i];
                cut.push_back(i);
            }
        }

        if (substitute)
            return nlambda->jump(nentry, {}, nargs.cut(cut)); // TODO type_args!!!
    }

    nlambda->jump(ntarget, {}, nargs); // TODO type_args!!!
}

Def Mangler::mangle(Def odef) {
    auto i = old2new.find(odef);
    if (i != old2new.end())
        return i->second;

    if (!in_scope.contains(odef))
        return odef;

    if (auto olambda = odef->isa_lambda()) {
        auto nlambda = mangle_head(olambda);
        mangle_body(olambda, nlambda);
        return nlambda;
    } else if (auto param = odef->isa<Param>()) {
        assert(in_scope.contains(param->lambda()));
        mangle(param->lambda());
        assert(old2new.contains(param));
        return old2new[param];
    } else {
        auto oprimop = odef->as<PrimOp>();
        Array<Def> nops(oprimop->size());
        for (size_t i = 0, e = oprimop->size(); i != e; ++i)
            nops[i] = mangle(oprimop->op(i));
        return old2new[oprimop] = oprimop->rebuild(nops);
    }
}

//------------------------------------------------------------------------------

Lambda* mangle(const Scope& scope, ArrayRef<Type> type_args, ArrayRef<Def> args, ArrayRef<Def> lift) {
    return Mangler(scope, type_args, args, lift).mangle();
}

Lambda* drop(const Call& call) {
    Scope scope(call.args().front()->as_lambda());
    return drop(scope, call.type_args(), call.args().skip_front());
}

//------------------------------------------------------------------------------

}
