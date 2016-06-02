#include "thorin/transform/mangle.h"

#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/queue.h"

namespace thorin {

Mangler::Mangler(const Scope& scope, Types type_args, Defs args, Defs lift)
    : scope_(scope)
    , type_args_(type_args)
    , args_(args)
    , lift_(lift)
    , old_entry_(scope.entry())
{
    assert(!old_entry()->empty());
    assert(args.size() == old_entry()->num_params());
    assert(type_args.size() == old_entry()->num_type_params());

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
    // map type params
    std::vector<const TypeParam*> type_params;
    for (size_t i = 0, e = old_entry()->num_type_params(); i != e; ++i) {
        auto otype_param = old_entry()->type_param(i);
        if (auto type = type_args_[i])
            type2type_[otype_param] = type;
        else {
            auto ntype_param = world().type_param(otype_param->name());
            type_params.push_back(ntype_param);
            type2type_[otype_param] = ntype_param;
        }
    }

    // create new_entry - but first collect and specialize all param types
    std::vector<const Type*> param_types;
    for (size_t i = 0, e = old_entry()->num_params(); i != e; ++i) {
        if (args_[i] == nullptr)
            param_types.push_back(old_entry()->param(i)->type()->specialize(type2type_));
    }

    auto fn_type = world().fn_type(param_types);
    new_entry_ = world().continuation(close(fn_type, type_params), old_entry()->loc(), old_entry()->name);

    // map value params
    def2def_[old_entry()] = old_entry();
    for (size_t i = 0, j = 0, e = old_entry()->num_params(); i != e; ++i) {
        auto old_param = old_entry()->param(i);
        if (auto def = args_[i])
            def2def_[old_param] = def;
        else {
            auto new_param = new_entry()->param(j++);
            def2def_[old_param] = new_param;
            new_param->name = old_param->name;
        }
    }

    for (auto def : lift_) {
        auto param = new_entry()->append_param(def->type()->specialize(type2type_));
        def2def_[def] = param;
    }

    mangle_body(old_entry(), new_entry());
    return new_entry();
}

Continuation* Mangler::mangle_head(Continuation* old_continuation) {
    assert(!def2def_.contains(old_continuation));
    assert(!old_continuation->empty());
    Continuation* new_continuation = old_continuation->stub(type2type_, old_continuation->name);
    def2def_[old_continuation] = new_continuation;

    for (size_t i = 0, e = old_continuation->num_params(); i != e; ++i)
        def2def_[old_continuation->param(i)] = new_continuation->param(i);

    return new_continuation;
}

void Mangler::mangle_body(Continuation* old_continuation, Continuation* new_continuation) {
    assert(!old_continuation->empty());

    if (old_continuation->callee() == world().branch()) {        // fold branch if possible
        if (auto lit = mangle(old_continuation->arg(0))->isa<PrimLit>())
            return new_continuation->jump(mangle(lit->value().get_bool() ? old_continuation->arg(1) : old_continuation->arg(2)), {}, {}, old_continuation->jump_loc());
    }

    Array<const Def*> nops(old_continuation->size());
    for (size_t i = 0, e = nops.size(); i != e; ++i)
        nops[i] = mangle(old_continuation->op(i));

    Defs nargs(nops.skip_front());      // new args of new_continuation
    const Def* ntarget = nops.front();  // new target of new_continuation

    // specialize all type args
    Array<const Type*> ntype_args(old_continuation->type_args().size());
    for (size_t i = 0, e = ntype_args.size(); i != e; ++i)
        ntype_args[i] = old_continuation->type_arg(i)->specialize(type2type_);

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
            const auto& args = concat(nargs.cut(cut), new_entry()->params().get_last(lift_.size()));
            return new_continuation->jump(new_entry(), ntype_args, args, old_continuation->jump_loc());
        }
    }

    new_continuation->jump(ntarget, ntype_args, nargs, old_continuation->jump_loc());
}

const Def* Mangler::mangle(const Def* old_def) {
    auto i = def2def_.find(old_def);
    if (i != def2def_.end())
        return i->second;

    if (!within(old_def))
        return old_def;

    if (auto old_continuation = old_def->isa_continuation()) {
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
        Array<const Def*> nops(old_primop->size());
        for (size_t i = 0, e = old_primop->size(); i != e; ++i)
            nops[i] = mangle(old_primop->op(i));

        auto type = old_primop->type()->vinstantiate(type2type_);
        return def2def_[old_primop] = old_primop->rebuild(nops, type);
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
