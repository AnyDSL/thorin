#include "thorin/transform/mangle.h"

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

const Def* Rewriter::instantiate(const Def* odef) {
    if (auto ndef = find(old2new, odef))
        return ndef;

    if (auto oprimop = odef->isa<PrimOp>()) {
        Array<const Def*> nops(oprimop->num_ops());
        for (size_t i = 0; i != oprimop->num_ops(); ++i)
            nops[i] = instantiate(odef->op(i));

        auto nprimop = oprimop->rebuild(nops);
        return old2new[oprimop] = nprimop;
    }

    return old2new[odef] = odef;
}

Mangler::Mangler(const Scope& scope, Defs args, Defs lift)
    : scope_(scope)
    , args_(args)
    , lift_(lift)
    , old_entry_(scope.entry())
    , defs_(scope.defs().capacity())
    , def2def_(scope.defs().capacity())
{
    assert(!old_entry()->is_empty());
    assert(args.size() == old_entry()->num_params());

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
    // create new_entry - but first collect and specialize all param types
    std::vector<const Type*> param_types;
    for (size_t i = 0, e = old_entry()->num_params(); i != e; ++i) {
        if (args_[i]->isa<Top>())
            param_types.emplace_back(old_entry()->param(i)->type()); // TODO reduce
    }

    auto pi = world().cn(param_types);
    new_entry_ = world().lam(pi, old_entry()->debug_history());

    // map value params
    def2def_[old_entry()] = old_entry();
    for (size_t i = 0, j = 0, e = old_entry()->num_params(); i != e; ++i) {
        auto old_param = old_entry()->param(i);
        if (auto def = args_[i])
            def2def_[old_param] = def;
        else {
            auto new_param = new_entry()->param(j++);
            def2def_[old_param] = new_param;
            new_param->debug().set(old_param->name());
        }
    }

    // TODO lifting
    //for (auto def : lift_)
        //def2def_[def] = new_entry()->append_param(def->type()); // TODO reduce

    // mangle filter
    if (old_entry()->filter() != nullptr) {
        Array<const Def*> new_filter(new_entry()->num_params());
        size_t j = 0;
        for (size_t i = 0, e = old_entry()->num_params(); i != e; ++i) {
            if (args_[i]->isa<Top>())
                new_filter[j++] = mangle(old_entry()->filter(i));
        }

        for (size_t e = new_entry()->num_params(); j != e; ++j)
            new_filter[j] = world().literal_bool(false, Debug{});

        new_entry()->set_filter(new_filter);
    }

    mangle_body(old_entry(), new_entry());

    return new_entry();
}

Lam* Mangler::mangle_head(Lam* old_lam) {
    assert(!def2def_.contains(old_lam));
    assert(!old_lam->is_empty());
    Lam* new_lam = old_lam->stub()->as_lam();
    def2def_[old_lam] = new_lam;

    for (size_t i = 0, e = old_lam->num_params(); i != e; ++i)
        def2def_[old_lam->param(i)] = new_lam->param(i);

    return new_lam;
}

void Mangler::mangle_body(Lam* old_lam, Lam* new_lam) {
    // check whether we can optimize tail recursion
    //if (ntarget == old_entry()) {
        //std::vector<size_t> cut;
        //bool substitute = true;
        //for (size_t i = 0, e = args_.size(); i != e && substitute; ++i) {
            //if (auto def = args_[i]) {
                //substitute &= def == nargs[i];
                //cut.push_back(i);
            //}
        //}

        //if (substitute) {
            //const auto& args = concat(nargs.cut(cut), new_entry()->params().get_back(lift_.size()));
            //return new_lam->app(new_entry(), args, old_lam->app()->debug());
        //}
    //}

    auto new_filter = mangle(old_lam->filter());
    auto new_body   = mangle(old_lam->body());
    new_lam->set_filter(new_filter);
    new_lam->set_body  (new_body);
}

const Def* Mangler::mangle(const Def* old_def) {
    if (auto new_def = find(def2def_, old_def))
        return new_def;
    else if (!within(old_def))
        return old_def;
    else if (auto old_lam = old_def->isa_lam()) {
        auto new_lam = mangle_head(old_lam);
        mangle_body(old_lam, new_lam);
        return new_lam;
    } else if (auto param = old_def->isa<Param>()) {
        assert(within(param->lam()));
        mangle(param->lam());
        assert(def2def_.contains(param));
        return def2def_[param];
    } else {
        auto old_primop = old_def->as<PrimOp>();
        Array<const Def*> nops(old_primop->num_ops());
        for (size_t i = 0, e = old_primop->num_ops(); i != e; ++i)
            nops[i] = mangle(old_primop->op(i));

        auto type = old_primop->type();
        return def2def_[old_primop] = old_primop->rebuild(type, nops);
    }
}

//------------------------------------------------------------------------------

Lam* mangle(const Scope& scope, Defs args, Defs lift) {
    return Mangler(scope, args, lift).mangle();
}

Lam* drop(const App* app) {
    Scope scope(app->callee()->as_lam());
    return drop(scope, app->args());
}

//------------------------------------------------------------------------------

}
