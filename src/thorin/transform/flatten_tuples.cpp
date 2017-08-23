#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"
#include "thorin/util/log.h"

namespace thorin {

static Continuation* flatten_continuation(Continuation* cont, const Param* ret, const Type* new_ret_type) {
    // The following code transforms:
    //
    // cont(mem, a : T, ret: fn (mem, (U, V))
    //     ...
    //     ret(mem, (a, b))
    //
    // into:
    //
    // new_cont(mem, a : T, new_ret: fn(mem, U, V))
    //     cont(mem, a, fixup_cont)
    //
    //     fixup_cont(mem, tuple: (U, V)):
    //         new_ret(mem, U, V)

    auto& world = cont->world();

    // create the new continuation
    Array<const Type*> param_types(cont->num_params());
    for (size_t i = 0, e = cont->num_params(); i != e; ++i)
        param_types[i] = i == ret->index() ? new_ret_type : cont->param(i)->type();

    auto new_cont_type = world.fn_type(param_types);
    auto new_cont = world.continuation(new_cont_type, cont->debug());
    auto new_ret = new_cont->param(ret->index());

    // create the fixup continuation, that extracts the elements from tuples
    auto fixup_cont = world.continuation(ret->type()->as<FnType>(), ret->debug());
    std::vector<const Def*> fixup_args;
    for (size_t i = 0, e = fixup_cont->num_params(); i != e; ++i) {
        auto param = fixup_cont->param(i);
        if (auto tuple_type = param->type()->isa<TupleType>()) {
            for (size_t j = 0, e = tuple_type->num_ops(); j != e; ++j)
                fixup_args.push_back(world.extract(param, j, ret->debug()));
        } else {
            fixup_args.push_back(param);
        }
    }
    fixup_cont->jump(new_ret, fixup_args, ret->debug());

    // wire the new continuation to the old one through the fixup continuation
    Array<const Def*> call_args(new_cont->num_params() + 1);
    for (size_t i = 0, e = new_cont->num_params(); i != e; ++i) {
        call_args[i + 1] = i == ret->index() ? fixup_cont->as<Def>() : new_cont->param(i);
    }
    call_args[0] = cont;
    auto dropped = drop(Call(call_args));

    new_cont->jump(dropped->callee(), dropped->args(), cont->debug());
    return new_cont;
}

static bool flatten_uses(Continuation* cont, const Param* ret, Continuation* new_cont) {
    // The following code transforms uses of cont:
    // 
    // foo(...)
    //     cont(..., res_cont)
    //
    //     res_cont(mem, tuple: (U, V)):
    //         ...
    //
    // into:
    // 
    // foo(...)
    //     new_cont(..., fixup_cont)
    //
    //     fixup_cont(mem, a: U, b: V):
    //         res_cont(mem, tuple(a, b))
    //
    //     res_cont(mem, tuple: (U, V)):
    //         ...

    auto& world = cont->world();

    bool transformed = false;
    for (auto use : cont->copy_uses()) {
        auto ucontinuation = use->isa_continuation();
        if (!ucontinuation || use.index() != 0 || use.def() == new_cont) continue;

        auto res_cont = ucontinuation->arg(ret->index());
        auto res_type = res_cont->type()->as<FnType>();

        // create the fixup continuation
        std::vector<const Def*> fixup_args;
        auto fixup_type = new_cont->param(ret->index())->type()->as<FnType>();
        auto fixup_cont = world.continuation(fixup_type, res_cont->debug());

        for (size_t i = 0, j = 0, e = res_type->num_ops(); i != e; ++i) {
            auto param_type = res_type->op(i);
            if (auto tuple_type = param_type->isa<TupleType>()) {
                Array<const Def*> tuple_args(tuple_type->num_ops());
                for (size_t k = 0, e = tuple_type->num_ops(); k != e; ++k)
                    tuple_args[k] = fixup_cont->param(j++);

                fixup_args.push_back(world.tuple(tuple_args));
            } else {
                fixup_args.push_back(fixup_cont->param(j++));
            }
        }
        fixup_cont->jump(res_cont, fixup_args, res_cont->debug());

        // call the new continuation through using the fixup continuation as return argument
        Array<const Def*> new_args(ucontinuation->num_args());
        for (size_t i = 0, e = ucontinuation->num_args(); i != e; ++i)
            new_args[i] = i == ret->index() ? fixup_cont : ucontinuation->arg(i);
        ucontinuation->jump(new_cont, new_args, ucontinuation->jump_debug());

        transformed = true;
    }
    return transformed;
}

void flatten_tuples(World& world) {
    // flatten tuples returned by functions
    bool todo = true;
    while (todo) {
        todo = false;
        for (auto cont : world.copy_continuations()) {
            if (!cont->is_returning() || cont->empty()) continue;

            auto ret = cont->ret_param();
            if (!ret) continue;

            bool needs_flattening = false;
            auto ret_type = ret->type()->as<FnType>();
            std::vector<const Type*> new_ret_params;
            for (size_t i = 0, e = ret_type->num_ops(); i != e; ++i) {
                auto param_type = ret_type->op(i);
                if (auto tuple_type = param_type->isa<TupleType>()) {
                    // flatten the arguments
                    needs_flattening = true;
                    for (auto op : tuple_type->ops())
                        new_ret_params.push_back(op);
                } else {
                    new_ret_params.push_back(param_type);
                }
            }

            if (!needs_flattening) continue;

            auto new_cont = flatten_continuation(cont, ret, world.fn_type(new_ret_params));          
            todo |= flatten_uses(cont, ret, new_cont);

            DLOG("flattened: {}", cont);
        }
        world.cleanup();
    }
}

}
