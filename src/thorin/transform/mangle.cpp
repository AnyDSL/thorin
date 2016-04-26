#include "thorin/transform/mangle.h"

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

class Mangler {
public:
    Mangler(const Scope& scope, Types type_args, Defs args, Defs lift)
        : scope(scope)
        , type_args(type_args)
        , args(args)
        , lift(lift)
        , oentry(scope.entry())
    {
        assert(!oentry->empty());
        assert(args.size() == oentry->num_params());
        assert(type_args.size() == oentry->num_type_params());

        // TODO correctly deal with continuations here
        std::queue<const Def*> queue;
        auto enqueue = [&](const Def* def) {
            if (!within(def)) {
                defs_.insert(def);
                queue.push(def);
            }
        };

        for (auto def : lift)
            enqueue(def);

        while (!queue.empty()) {
            for (auto use : pop(queue)->uses())
                enqueue(use);
        }
    }

    World& world() const { return scope.world(); }
    Continuation* mangle();
    void mangle_body(Continuation* ocontinuation, Continuation* ncontinuation);
    Continuation* mangle_head(Continuation* ocontinuation);
    const Def* mangle(const Def* odef);
    bool within(const Def* def) { return scope.contains(def) || defs_.contains(def); }

    const Scope& scope;
    Def2Def def2def;
    Types type_args;
    Defs args;
    Defs lift;
    Type2Type type2type;
    Continuation* oentry;
    Continuation* nentry;
    DefSet defs_;
};

Continuation* Mangler::mangle() {
    // map type params
    std::vector<const TypeParam*> type_params;
    for (size_t i = 0, e = oentry->num_type_params(); i != e; ++i) {
        auto otype_param = oentry->type_param(i);
        if (auto type = type_args[i])
            type2type[otype_param] = type;
        else {
            auto ntype_param = world().type_param(otype_param->name());
            type_params.push_back(ntype_param);
            type2type[otype_param] = ntype_param;
        }
    }

    // create nentry - but first collect and specialize all param types
    std::vector<const Type*> param_types;
    for (size_t i = 0, e = oentry->num_params(); i != e; ++i) {
        if (args[i] == nullptr)
            param_types.push_back(oentry->param(i)->type()->specialize(type2type));
    }

    auto fn_type = world().fn_type(param_types);
    nentry = world().continuation(close(fn_type, type_params), oentry->loc(), oentry->name);

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

Continuation* Mangler::mangle_head(Continuation* ocontinuation) {
    assert(!def2def.contains(ocontinuation));
    assert(!ocontinuation->empty());
    Continuation* ncontinuation = ocontinuation->stub(type2type, ocontinuation->name);
    def2def[ocontinuation] = ncontinuation;

    for (size_t i = 0, e = ocontinuation->num_params(); i != e; ++i)
        def2def[ocontinuation->param(i)] = ncontinuation->param(i);

    return ncontinuation;
}

void Mangler::mangle_body(Continuation* ocontinuation, Continuation* ncontinuation) {
    assert(!ocontinuation->empty());

    if (ocontinuation->callee() == world().branch()) {        // fold branch if possible
        if (auto lit = mangle(ocontinuation->arg(0))->isa<PrimLit>())
            return ncontinuation->jump(mangle(lit->value().get_bool() ? ocontinuation->arg(1) : ocontinuation->arg(2)), {}, {}, ocontinuation->jump_loc());
    }

    Array<const Def*> nops(ocontinuation->size());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = mangle(ocontinuation->op(i));

    Defs nargs(nops.skip_front());         // new args of ncontinuation
    const Def* ntarget = nops.front();                     // new target of ncontinuation

    // specialize all type args
    Array<const Type*> ntype_args(ocontinuation->type_args().size());
    for (size_t i = 0, e = ntype_args.size(); i != e; ++i)
        ntype_args[i] = ocontinuation->type_arg(i)->specialize(type2type);

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
            return ncontinuation->jump(nentry, ntype_args, nargs.cut(cut), ocontinuation->jump_loc());
    }

    ncontinuation->jump(ntarget, ntype_args, nargs, ocontinuation->jump_loc());
}

const Def* Mangler::mangle(const Def* odef) {
    auto i = def2def.find(odef);
    if (i != def2def.end())
        return i->second;

    if (!within(odef))
        return odef;

    if (auto ocontinuation = odef->isa_continuation()) {
        auto ncontinuation = mangle_head(ocontinuation);
        mangle_body(ocontinuation, ncontinuation);
        return ncontinuation;
    } else if (auto param = odef->isa<Param>()) {
        assert(within(param->continuation()));
        mangle(param->continuation());
        assert(def2def.contains(param));
        return def2def[param];
    } else {
        auto oprimop = odef->as<PrimOp>();
        Array<const Def*> nops(oprimop->size());
        for (size_t i = 0, e = oprimop->size(); i != e; ++i)
            nops[i] = mangle(oprimop->op(i));

        auto type = oprimop->type()->vinstantiate(type2type);
        return def2def[oprimop] = oprimop->rebuild(nops, type);
    }
}

//------------------------------------------------------------------------------

Continuation* mangle(const Scope& scope, Types type_args, Defs args, Defs lift) {
    return Mangler(scope, type_args, args, lift).mangle();
}

Continuation* drop(const Call& call) {
    Scope scope(call.callee()->as_continuation());
    return drop(scope, call.type_args(), call.args());
}

//------------------------------------------------------------------------------

}
