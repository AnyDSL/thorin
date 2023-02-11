#include "thorin/transform/mangle.h"

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

/// Mangles a continuation's scope
/// @p args has the size of the original continuation, a null entry means the parameter remains, non-null substitutes it in scope and removes it from the signature
/// @p lift lists defs that should be replaced by a fresh param, to be appended at the end of the signature
Mangler::Mangler(const Scope& scope, Defs args, Defs lift)
    : Rewriter(scope.world())
    , scope_(scope)
    , args_(args)
    , lift_(lift)
    , old_entry_(scope.entry())
    , defs_(scope.defs().capacity())
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

    auto fn_type = dst().fn_type(param_types);
    new_entry_ = dst().continuation(fn_type, old_entry()->debug_history());

    // map value params
    insert(old_entry(), old_entry());
    for (size_t i = 0, j = 0, e = old_entry()->num_params(); i != e; ++i) {
        auto old_param = old_entry()->param(i);
        if (auto def = args_[i])
            insert(old_param, def);
        else {
            // we recreate params that aren't specialized
            auto new_param = new_entry()->param(j++);
            insert(old_param, new_param);
            new_param->set_name(old_param->name());
        }
    }

    for (auto def : lift_)
        insert(def, new_entry()->append_param(def->type()));

    // cut/widen filter
    if (!old_entry()->filter()->is_empty()) {
        Array<const Def*> new_conditions(new_entry()->num_params());
        size_t j = 0;
        for (size_t i = 0, e = old_entry()->num_params(); i != e; ++i) {
            if (args_[i] == nullptr)
                new_conditions[j++] = instantiate(old_entry()->filter()->condition(i));
        }

        for (size_t e = new_entry()->num_params(); j != e; ++j)
            new_conditions[j] = dst().literal_bool(false, Debug{});

        new_entry()->set_filter(dst().filter(new_conditions, old_entry()->filter()->debug()));
    }

    new_entry()->set_body(instantiate(old_entry()->body())->as<App>());
    new_entry()->verify();

    return new_entry();
}

const Def* Mangler::rewrite(const Def* old_def) {
    if (!within(old_def))
        return old_def;  // we leave free variables alone
    if (auto param = old_def->isa<Param>())
        assert(within(param->continuation()) && "if the param is not free, the continuation should not be either!");
    auto ndef = Rewriter::rewrite(old_def);
    if (auto app = ndef->isa<App>()) {
        auto oargs = app->args();
        auto nargs = Array<const Def*>(oargs.size(), [&](size_t i) { return rewrite(oargs[i]); });

        // check whether we can optimize tail recursion
        if (app->callee() == old_entry()) {
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
                auto rebuilt_filter = rewrite(app->filter())->as<Filter>();
                const Filter* nfilter = rebuilt_filter->is_empty() ? rebuilt_filter : dst().filter(concat(rebuilt_filter->cut(cut)->ops(), Array<const Def*>(lift_.size(), [&](size_t) { return dst().literal_bool(false, {}); })));
                return dst().app(nfilter, new_entry(), args, old_def->debug()); // TODO debug
            }
        }
    }
    return ndef;
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
