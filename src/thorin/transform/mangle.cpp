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
    assert(!old_entry()->empty());
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
            auto new_param = new_entry()->param(j++);
            def2def_[old_param] = new_param;
            new_param->debug().set(old_param->name());
        }
    }

    for (auto def : lift_)
        def2def_[def] = new_entry()->append_param(def->type()); // TODO reduce

    // mangle pe_profile
    if (!old_entry()->pe_profile().empty()) {
        Array<const Def*> new_pe_profile(new_entry()->num_params());
        size_t j = 0;
        for (size_t i = 0, e = old_entry()->num_params(); i != e; ++i) {
            if (args_[i] == nullptr)
                new_pe_profile[j++] = mangle(old_entry()->pe_profile(i));
        }

        for (size_t e = new_entry()->num_params(); j != e; ++j)
            new_pe_profile[j] = world().literal_bool(false, Debug{});

        new_entry()->set_pe_profile(new_pe_profile);
    }

    mangle_body(old_entry(), new_entry());

    if (!lift_.empty())
        return new_entry();

    // this is the eat-up-return trick
    // if the new continuations only contains one call to one of its arguments - eat up this call, too
    ContinuationSet rets;
    for (auto arg : args_) {
        if (arg && arg->isa_continuation())
            rets.emplace(arg->as_continuation());
    }

    if (rets.empty())
        return new_entry();

    Continuation* fold = nullptr;
    for (auto new_continuation : new_continuations_) {
        if (auto callee = new_continuation->callee()->isa_continuation()) {
            if (rets.contains(callee)) {
                if (fold == nullptr)
                    fold = new_continuation;
                else
                    return new_entry(); // more then one "returns"
            }
        }
    }

    if (fold != nullptr && !fold->callee()->empty()) {
        Scope s(fold->callee()->as_continuation());
        auto dropped = drop(s, fold->args());
        fold->jump(dropped, {}, fold->jump_debug());
    }

    return new_entry();
}

Continuation* Mangler::mangle_head(Continuation* old_continuation) {
    assert(!def2def_.contains(old_continuation));
    assert(!old_continuation->empty());
    Continuation* new_continuation = old_continuation->stub();
    def2def_[old_continuation] = new_continuation;

    for (size_t i = 0, e = old_continuation->num_params(); i != e; ++i)
        def2def_[old_continuation->param(i)] = new_continuation->param(i);

    return new_continuation;
}

void Mangler::mangle_body(Continuation* old_continuation, Continuation* new_continuation) {
    assert(!old_continuation->empty());
    new_continuations_.emplace_back(new_continuation);

    // fold branch and match
    // TODO find a way to factor this out in continuation.cpp
    if (auto callee = old_continuation->callee()->isa_continuation()) {
        switch (callee->intrinsic()) {
            case Intrinsic::Branch: {
                if (auto lit = mangle(old_continuation->arg(0))->isa<PrimLit>()) {
                    auto cont = lit->value().get_bool() ? old_continuation->arg(1) : old_continuation->arg(2);
                    return new_continuation->jump(mangle(cont), {}, old_continuation->jump_debug());
                }
                break;
            }
            case Intrinsic::Match:
                if (old_continuation->num_args() == 2)
                    return new_continuation->jump(mangle(old_continuation->arg(1)), {}, old_continuation->jump_debug());

                if (auto lit = mangle(old_continuation->arg(0))->isa<PrimLit>()) {
                    for (size_t i = 2; i < old_continuation->num_args(); i++) {
                        auto new_arg = mangle(old_continuation->arg(i));
                        if (world().extract(new_arg, 0_s)->as<PrimLit>() == lit)
                            return new_continuation->jump(world().extract(new_arg, 1), {}, old_continuation->jump_debug());
                    }
                }
                break;
            default:
                break;
        }
    }

    Array<const Def*> nops(old_continuation->num_ops());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = mangle(old_continuation->op(i));

    Defs nargs(nops.skip_front()); // new args of new_continuation
    auto ntarget = nops.front();   // new target of new_continuation

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
            const auto& args = concat(nargs.cut(cut), new_entry()->params().get_back(lift_.size()));
            return new_continuation->jump(new_entry(), args, old_continuation->jump_debug());
        }
    }

    new_continuation->jump(ntarget, nargs, old_continuation->jump_debug());
}

const Def* Mangler::mangle(const Def* old_def) {
    if (auto new_def = find(def2def_, old_def))
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
        assert(def2def_.contains(param));
        return def2def_[param];
    } else {
        auto old_primop = old_def->as<PrimOp>();
        Array<const Def*> nops(old_primop->num_ops());
        for (size_t i = 0, e = old_primop->num_ops(); i != e; ++i)
            nops[i] = mangle(old_primop->op(i));

        auto type = old_primop->type(); // TODO reduce
        return def2def_[old_primop] = old_primop->rebuild(nops, type);
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
