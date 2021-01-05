#include "thorin/transform/mangle.h"

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

Mangler::Mangler(const Scope& scope, Defs args, Defs lift)
    : scope_(scope)
    , args_(args)
    , old_entry_(scope.entry()->as<Lam>())
    , defs_(scope.bound().capacity())
    , old2new_(scope.bound().capacity())
{
    assert(old_entry()->is_set());
    assert(args.size() == old_entry()->num_vars());

    // TODO correctly deal with lams here
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

Lam* Mangler::mangle() {
    // create new_entry - but first collect and specialize all var types
    std::vector<const Def*> var_types;
    for (size_t i = 0, e = old_entry()->num_vars(); i != e; ++i) {
        if (args_[i]->isa<Top>())
            var_types.emplace_back(old_entry()->var(i)->type());
    }

    auto cn = world().cn(var_types);
    new_entry_ = world().nom_lam(cn, old_entry()->debug_history());

    // HACK we wil remove this code anyway
    bool all = true;
    // map vars
    //old2new_[old_entry()] = old_entry();
    for (size_t i = 0, j = 0, e = old_entry()->num_vars(); i != e; ++i) {
        auto old_var = old_entry()->var(i);
        if (!args_[i]->isa<Top>())
            old2new_[old_var] = args_[i];
        else {
            all = false;
            auto new_var = new_entry()->var(j++, old_var->debug_history());
            old2new_[old_var] = new_var;
        }
    }

    if (all)
        old2new_[old_entry()->var()] = world().tuple(args_);

    auto new_filter = mangle(old_entry()->filter());
    auto new_body   = mangle(old_entry()->body());
    new_entry()->set(new_filter, new_body);

    return new_entry();
}

const Def* Mangler::mangle(const Def* old_def) {
    // TODO merge with importer
    if (auto new_def = old2new_.lookup(old_def)) return *new_def;
    if (!within(old_def)) return old_def;

    auto new_type = mangle(old_def->type());
    auto new_dbg  = old_def->dbg() ? mangle(old_def->dbg()) : nullptr;

    const Def* new_def = nullptr;
    if (auto old_nom = old_def->isa_nom()) {
        new_def = old_nom->stub(world(), new_type, new_dbg );
        old2new_[old_def] = new_def;
    }

    size_t size = old_def->num_ops();
    Array<const Def*> new_ops(size);
    for (size_t i = 0; i != size; ++i) {
        new_ops[i] = mangle(old_def->op(i));
        assert(&new_ops[i]->world() == &world());
    }

    if (new_def) {
        for (size_t i = 0; i != size; ++i)
            const_cast<Def*>(new_def)->set(i, new_ops[i]);
    } else {
        // check whether we can optimize tail recursion
        if (auto app = old_def->isa<App>()) {
            assert(new_ops.size() == 2);

            if (app->callee() == old_entry()) {
                if (args_.size() == 1 && args_[0] == new_ops[1])
                    return world().app(new_entry(), thorin::Defs {}, app->dbg ());

                if (auto tuple = new_ops[1]->isa<Tuple>()) {
                    assert(tuple->num_ops() == args_.size());

                    std::vector<size_t> cut;
                    bool substitute = true;
                    for (size_t i = 0, e = args_.size(); i != e && substitute; ++i) {
                        if (!args_[i]->isa<Top>()) {
                            substitute &= args_[i] == tuple->op(i);
                            cut.push_back(i);
                        }
                    }

                    if (substitute) {
                        // TODO lifting
                        //const auto& args = concat(new_args.cut(cut), new_entry()->vars().get_back(lift_.size()));
                        auto args = tuple->ops().cut(cut);
                        return world().app(new_entry(), args, app->dbg ());
                    }
                }
            }
        }

        new_def = old_def->rebuild(world(), new_type, new_ops, new_dbg );
        old2new_[old_def] = new_def;
    }

    return new_def;
}

//------------------------------------------------------------------------------

Lam* mangle(const Scope& scope, Defs args, Defs lift) {
    return Mangler(scope, args, lift).mangle();
}

Lam* drop(const App* app) {
    Scope scope(app->callee()->as_nom<Lam>());
    return drop(scope, app->args());
}

//------------------------------------------------------------------------------

}
