#include "thorin/transform/mangle.h"

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

class Mangler {
public:
    Mangler(const Scope& scope, ArrayRef<Type> type_args, ArrayRef<const Def*> args, ArrayRef<const Def*> lift)
        : scope(scope)
        , type_args(type_args)
        , args(args)
        , lift(lift)
        , in_scope(scope.in_scope()) // copy constructor
        , oentry(scope.entry())
    {
        assert(!oentry->empty());
        assert(args.size() == oentry->num_params());
        assert(type_args.size() == oentry->num_type_params());

        std::queue<const Def*> queue;
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
    const Def* mangle(const Def* odef);

    const Scope& scope;
    Def2Def def2def;
    ArrayRef<Type> type_args;
    ArrayRef<const Def*> args;
    ArrayRef<const Def*> lift;
    Type2Type type2type;
    DefSet in_scope;
    Lambda* oentry;
    Lambda* nentry;
};

Lambda* Mangler::mangle() {
    // map type params
    std::vector<TypeParam> type_params;
    for (size_t i = 0, e = oentry->num_type_params(); i != e; ++i) {
        auto otype_param = oentry->type_param(i);
        if (auto type = type_args[i])
            type2type[*otype_param] = *type;
        else {
            auto ntype_param = world().type_param();
            type_params.push_back(ntype_param);
            type2type[*otype_param] = *ntype_param;
        }
    }

    // create nentry - but first collect and specialize all param types
    std::vector<Type> param_types;
    for (size_t i = 0, e = oentry->num_params(); i != e; ++i) {
        if (args[i] == nullptr)
            param_types.push_back(oentry->param(i)->type()->specialize(type2type));
    }
    nentry = world().lambda(world().fn_type(param_types), oentry->loc(), oentry->name);

    for (auto type_param : type_params)
        nentry->type()->bind(type_param);

    // map value params
    def2def[oentry] = oentry;
    for (size_t i = 0, j = 0, e = oentry->num_params(); i != e; ++i) {
        auto oparam = oentry->param(i);
        if (auto def = args[i])
            def2def[oparam] = def;
        else {
            auto nparam = nentry->param(j++);
            def2def[oparam] = nparam;
            nparam->name = oparam->name;
        }
    }

    for (auto def : lift)
        def2def[def] = nentry->append_param(def->type()->specialize(type2type));

    mangle_body(oentry, nentry);
    return nentry;
}

Lambda* Mangler::mangle_head(Lambda* olambda) {
    assert(!def2def.contains(olambda));
    assert(!olambda->empty());
    Lambda* nlambda = olambda->stub(type2type, olambda->name);
    def2def[olambda] = nlambda;

    for (size_t i = 0, e = olambda->num_params(); i != e; ++i)
        def2def[olambda->param(i)] = nlambda->param(i);

    return nlambda;
}

void Mangler::mangle_body(Lambda* olambda, Lambda* nlambda) {
    assert(!olambda->empty());

    if (olambda->to() == world().branch()) {        // fold branch if possible
        if (auto lit = mangle(olambda->arg(0))->isa<PrimLit>())
            return nlambda->jump(mangle(lit->value().get_bool() ? olambda->arg(1) : olambda->arg(2)), {}, {}, olambda->jump_loc());
    }

    Array<const Def*> nops(olambda->size());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = mangle(olambda->op(i));

    ArrayRef<const Def*> nargs(nops.skip_front());         // new args of nlambda
    const Def* ntarget = nops.front();                     // new target of nlambda

    // specialize all type args
    Array<Type> ntype_args(olambda->type_args().size());
    for (size_t i = 0, e = ntype_args.size(); i != e; ++i)
        ntype_args[i] = olambda->type_arg(i)->specialize(type2type);

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
            return nlambda->jump(nentry, ntype_args, nargs.cut(cut), olambda->jump_loc());
    }

    nlambda->jump(ntarget, ntype_args, nargs, olambda->jump_loc());
}

const Def* Mangler::mangle(const Def* odef) {
    auto i = def2def.find(odef);
    if (i != def2def.end())
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
        assert(def2def.contains(param));
        return def2def[param];
    } else {
        auto oprimop = odef->as<PrimOp>();
        Array<const Def*> nops(oprimop->size());
        for (size_t i = 0, e = oprimop->size(); i != e; ++i)
            nops[i] = mangle(oprimop->op(i));
        return def2def[oprimop] = oprimop->rebuild(nops);
    }
}

//------------------------------------------------------------------------------

Lambda* mangle(const Scope& scope, ArrayRef<Type> type_args, ArrayRef<const Def*> args, ArrayRef<const Def*> lift) {
    return Mangler(scope, type_args, args, lift).mangle();
}

Lambda* drop(const Call& call) {
    Scope scope(call.to()->as_lambda());
    return drop(scope, call.type_args(), call.args());
}

//------------------------------------------------------------------------------

}
