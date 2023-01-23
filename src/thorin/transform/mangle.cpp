#include "thorin/transform/mangle.h"

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

const Def* Rewriter::instantiate(const Def* odef) {
    if (auto ndef = old2new.lookup(odef)) return *ndef;

    if (odef->isa_structural()) {
        Array<const Def*> nops(odef->num_ops());
        for (size_t i = 0; i != odef->num_ops(); ++i)
            nops[i] = instantiate(odef->op(i));

        auto nprimop = odef->rebuild(odef->world(), odef->type(), nops);
        return old2new[odef] = nprimop;
    }

    return old2new[odef] = odef;
}

/// Mangles a continuation's scope
/// @p args has the size of the original continuation, a null entry means the parameter remains, non-null substitutes it in scope and removes it from the signature
/// @p lift lists defs that should be replaced by a fresh param, to be appended at the end of the signature
Mangler::Mangler(const Scope& scope, Defs args, Defs lift)
    : scope_(scope)
    , args_(args)
    , lift_(lift)
    , old_entry_(scope.entry())
    , defs_(scope.defs().capacity())
    , def2def_(scope.defs().capacity())
{
    assert(old_entry()->has_body());
    assert(args.size() == old_entry()->num_params());

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

Continuation* Mangler::mangle() {
    // create new_entry - but first collect and specialize all param types
    std::vector<const Type*> param_types;
    for (size_t i = 0, e = old_entry()->num_params(); i != e; ++i) {
        if (args_[i] == nullptr)
            param_types.emplace_back(old_entry()->param(i)->type()); // TODO reduce
    }

    auto fn_type = world().fn_type(param_types);
    new_entry_ = world().continuation(fn_type, old_entry()->debug_history());

    // map value params
    def2def_[old_entry()] = old_entry();
    for (size_t i = 0, j = 0, e = old_entry()->num_params(); i != e; ++i) {
        auto old_param = old_entry()->param(i);
        if (auto def = args_[i])
            def2def_[old_param] = def;
        else {
            // we recreate params that aren't specialized
            auto new_param = new_entry()->param(j++);
            def2def_[old_param] = new_param;
            new_param->set_name(old_param->name());
        }
    }

    for (auto def : lift_)
        def2def_[def] = new_entry()->append_param(def->type()); // TODO reduce

    // mangle filter
    if (!old_entry()->filter()->is_empty()) {
        Array<const Def*> new_conditions(new_entry()->num_params());
        size_t j = 0;
        for (size_t i = 0, e = old_entry()->num_params(); i != e; ++i) {
            if (args_[i] == nullptr)
                new_conditions[j++] = mangle(old_entry()->filter()->condition(i));
        }

        for (size_t e = new_entry()->num_params(); j != e; ++j)
            new_conditions[j] = world().literal_bool(false, Debug{});

        new_entry()->set_filter(world().filter(new_conditions, old_entry()->filter()->debug()));
    }

    new_entry()->set_body(mangle_body(old_entry()->body()));

    new_entry()->verify();

    return new_entry();
}

Continuation* Mangler::mangle_head(Continuation* old_continuation) {
    assert(!def2def_.contains(old_continuation));
    assert(old_continuation->has_body());
    Continuation* new_continuation = old_continuation->mangle_stub();
    def2def_[old_continuation] = new_continuation;

    for (size_t i = 0, e = old_continuation->num_params(); i != e; ++i)
        def2def_[old_continuation->param(i)] = new_continuation->param(i);

    return new_continuation;
}

const App* Mangler::mangle_body(const App* old_body) {
    Array<const Def*> nops(old_body->num_ops());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = mangle(old_body->op(i));

    Defs nargs(nops.skip_front()); // new args of body
    auto ntarget = nops.front();   // new target of body

    // check whether we can optimize tail recursion
    if (ntarget == old_entry()) {
        std::vector<size_t> cut;
        bool substitute = true;
        for (size_t i = 0, e = args_.size(); i != e && substitute; ++i) {
            if (auto def = args_[i]) {
                substitute &= def == nargs[i];
                cut.push_back(i);
            }
        }

        if (substitute) {
            // Q: why not always change to the mangled continuation ?
            // A: if you drop a parameter it is replaced by some def (likely a free param), which will be identical for all recursive calls, since they live in the same scope (that's how scopes work)
            // so if there originally was a recursive call that specified the to-be-dropped parameter to something else, we need to call the unmangled original to preserve semantics
            const auto& args = concat(nargs.cut(cut), new_entry()->params().get_back(lift_.size()));
            return world().app(new_entry(), args, old_body->debug()); // TODO debug
        }
    }

    return world().app(ntarget, nargs, old_body->debug()); // TODO debug
}

const Def* Mangler::mangle(const Def* old_def) {
    if (auto new_def = def2def_.lookup(old_def))
        return *new_def;
    else if (!within(old_def))
        return old_def;  // we leave free variables alone
    else if (auto old_continuation = old_def->isa_nom<Continuation>()) {
        auto new_continuation = mangle_head(old_continuation);
        if (old_continuation->has_body())
            new_continuation->set_body(mangle_body(old_continuation->body()));
        return new_continuation;
    } else if (auto param = old_def->isa<Param>()) {
        assert(within(param->continuation()));
        mangle(param->continuation());
        assert(def2def_.contains(param));
        return def2def_[param];
    } else {
        Array<const Def*> nops(old_def->num_ops());
        for (size_t i = 0, e = old_def->num_ops(); i != e; ++i)
            nops[i] = mangle(old_def->op(i));

        auto type = old_def->type(); // TODO reduce
        return def2def_[old_def] = old_def->rebuild(world(), type, nops);
    }
}

//------------------------------------------------------------------------------

Continuation* mangle(const Scope& scope, Defs args, Defs lift) {
    return Mangler(scope, args, lift).mangle();
}

Continuation* drop(const Def* callee, const Defs specialized_args) {
    Scope scope(callee->as_nom<Continuation>());
    return drop(scope, specialized_args);
}

//------------------------------------------------------------------------------

}
