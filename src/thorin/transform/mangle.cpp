#include "thorin/transform/mangle.h"

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

class Mangler {
public:
    Mangler(const Scope& scope, Defs args, Defs lift)
        : scope(scope)
        , args(args)
        , lift(lift)
        , oentry(scope.entry())
    {
        assert(!oentry->empty());
        assert(args.size() == oentry->num_params());

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
    void mangle_body(Continuation* old_continuation, Continuation* new_continuation);
    Continuation* mangle_head(Continuation* old_continuation);
    const Def* mangle(const Def* old_def);
    bool within(const Def* def) { return scope.contains(def) || defs_.contains(def); }

    const Scope& scope;
    Def2Def def2def;
    Defs args;
    Defs lift;
    Type2Type type2type;
    Continuation* oentry;
    Continuation* nentry;
    DefSet defs_;
};

Continuation* Mangler::mangle() {
    // create nentry - but first collect and specialize all param types
    std::vector<const Type*> param_types;
    for (size_t i = 0, e = oentry->num_params(); i != e; ++i) {
        if (args[i] == nullptr)
            param_types.emplace_back(oentry->param(i)->type()); // TODO reduce
    }

    auto fn_type = world().fn_type(param_types);
    nentry = world().continuation(fn_type, oentry->loc(), oentry->name);

    // map value params
    def2def[oentry] = oentry;
    for (size_t i = 0, j = 0, e = oentry->num_params(); i != e; ++i) {
        auto old_param = oentry->param(i);
        if (auto def = args[i])
            def2def[old_param] = def;
        else {
            auto new_param = nentry->param(j++);
            def2def[old_param] = new_param;
            new_param->name = old_param->name;
        }
    }

    for (auto def : lift)
        def2def[def] = nentry->append_param(def->type()); // TODO reduce

    mangle_body(oentry, nentry);
    return nentry;
}

Continuation* Mangler::mangle_head(Continuation* old_continuation) {
    assert(!def2def.contains(old_continuation));
    assert(!old_continuation->empty());
    Continuation* new_continuation = old_continuation->stub(type2type, old_continuation->name);
    def2def[old_continuation] = new_continuation;

    for (size_t i = 0, e = old_continuation->num_params(); i != e; ++i)
        def2def[old_continuation->param(i)] = new_continuation->param(i);

    return new_continuation;
}

void Mangler::mangle_body(Continuation* old_continuation, Continuation* new_continuation) {
    assert(!old_continuation->empty());

    if (old_continuation->callee() == world().branch()) {        // fold branch if possible
        if (auto lit = mangle(old_continuation->arg(0))->isa<PrimLit>())
            return new_continuation->jump(mangle(lit->value().get_bool() ? old_continuation->arg(1) : old_continuation->arg(2)), {}, old_continuation->jump_loc());
    }

    Array<const Def*> nops(old_continuation->size());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = mangle(old_continuation->op(i));

    Defs nargs(nops.skip_front()); // new args of new_continuation
    auto ntarget = nops.front();   // new target of new_continuation

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
            return new_continuation->jump(nentry, nargs.cut(cut), old_continuation->jump_loc());
    }

    new_continuation->jump(ntarget, nargs, old_continuation->jump_loc());
}

const Def* Mangler::mangle(const Def* old_def) {
    if (auto new_def = find(def2def, old_def))
        return new_def;
    else if (!within(old_def))
        return old_def;
    else if (auto old_continuation = old_def->isa_continuation()) {

        auto new_continuation = mangle_head(old_continuation);
        mangle_body(old_continuation, new_continuation);
        return new_continuation;
    } else if (auto param = old_def->isa<Param>()) {
        assert(within(param->continuation()));
        mangle(param->continuation());
        assert(def2def.contains(param));
        return def2def[param];
    } else {
        auto old_primop = old_def->as<PrimOp>();
        Array<const Def*> nops(old_primop->size());
        for (size_t i = 0, e = old_primop->size(); i != e; ++i)
            nops[i] = mangle(old_primop->op(i));

        auto type = old_primop->type(); // TODO reduce
        return def2def[old_primop] = old_primop->rebuild(nops, type);
    }
}

//------------------------------------------------------------------------------

Continuation* mangle(const Scope& scope, Defs args, Defs lift) {
    return Mangler(scope, args, lift).mangle();
}

Continuation* drop(const Call& call) {
    Scope scope(call.callee()->as_continuation());
    return drop(scope, call.args());
}

//------------------------------------------------------------------------------

}
