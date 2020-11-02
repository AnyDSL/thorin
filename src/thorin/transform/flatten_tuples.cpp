#include "thorin/lam.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"
#include "thorin/analyses/verify.h"
#include "thorin/util/log.h"

#include <limits>

namespace thorin {

static Lam*   wrap_def(Def2Def&, Def2Def&, const Def*, const Pi*);
static Lam* unwrap_def(Def2Def&, Def2Def&, const Def*, const Pi*);

// Computes the type of the wrapped function
static const Type* flatten_type(const Type* type) {
    if (type->isa<TupleType>()) {
        std::vector<const Type*> flat_ops;
        for (auto op : type->ops()) {
            auto flat_op = flatten_type(op);
            if (flat_op->isa<TupleType>())
                flat_ops.insert(flat_ops.end(), flat_op->ops().begin(), flat_op->ops().end());
            else
                flat_ops.push_back(flat_op);
        }
        return type->table().tuple_type(flat_ops);
    } else if (auto pi = type->isa<Pi>()) {
        return type->table().pi(flatten_type(pi->domain()), flatten_type(pi->codomain()));
    } else
        return type;
}

static Lam* app(Lam* lam, Array<const Def*>& args) {
    lam->app(args[0], args.skip_front(), args[0]->debug());
    return lam;
}

static Lam* try_inline(Lam* lam, Array<const Def*>& args) {
    if (args[0]->isa_lam()) {
        auto app = lam->world().app(args.front(), lam->world().tuple(args.skip_front()))->as<App>();
        auto dropped = drop(app);
        lam->app(dropped->app()->callee(), dropped->app()->args(), args[0]->debug());
    } else {
        app(lam, args);
    }
    return lam;
}

static void inline_calls(Lam* lam) {
    for (auto use : lam->copy_uses()) {
        auto ulam = use->isa_lam();
        if (!ulam || use.index() != 0) continue;

        Array<const Def*> args(ulam->app()->num_args() + 1);
        for (size_t i = 0, e = ulam->app()->num_args(); i != e; ++i) args[i + 1] = ulam->app()->arg(i);
        args[0] = ulam->app()->callee();
        try_inline(ulam, args);
    }
}

// Wraps around a def, flattening tuples passed as parameters (dual of unwrap)
static Lam* wrap_def(Def2Def& wrapped, Def2Def& unwrapped, const Def* old_def, const Pi* new_type) {
    // Transform:
    //
    // old_def(a: T, b: (U, V), c: fn (W, (X, Y))):
    //     ...
    //
    // into:
    //
    // new_lam(a: T, b: U, c: V, d: fn (W, X, Y)):
    //     old_def(a, (b, c), unwrap_d)
    //
    //     unwrap_d(a: W, b: (X, Y)):
    //         e = extract(b, 0)
    //         f = extract(b, 1)
    //         d(a, (e, f))

    if (wrapped.contains(old_def)) return wrapped[old_def]->as_lam();

    auto& world = old_def->world();
    auto old_type = old_def->type()->as<Pi>();
    auto new_lam = world.lam(new_type, old_def->debug());
    Array<const Def*> call_args(old_type->num_ops() + 1);

    wrapped.emplace(old_def, new_lam);

    for (size_t i = 0, j = 0, e = old_type->num_ops(); i != e; ++i) {
        auto op = old_type->op(i);
        if (auto tuple_type = op->isa<TupleType>()) {
            Array<const Def*> tuple_args(tuple_type->num_ops());
            for (size_t k = 0, e = tuple_type->num_ops(); k != e; ++k)
                tuple_args[k] = new_lam->param(j++);
            call_args[i + 1] = world.tuple(tuple_args);
        } else if (auto cn = op->isa<Pi>()) {
            auto fn_param = new_lam->param(j++);
            // no need to unwrap if the types are identical
            if (fn_param->type() != op)
                call_args[i + 1] = unwrap_def(wrapped, unwrapped, fn_param, cn);
            else
                call_args[i + 1] = fn_param;
        } else {
            call_args[i + 1] = new_lam->param(j++);
        }
    }

    call_args[0] = old_def;
    // inline the call, so that the old lam is eliminated
    return try_inline(new_lam, call_args);
}

// Unwrap a def, flattening tuples passed as arguments (dual of wrap)
static Lam* unwrap_def(Def2Def& wrapped, Def2Def& unwrapped, const Def* new_def, const Pi* old_type) {
    // Transform:
    //
    // new_def(a: T, b: U, c: V, d: fn (W, X, Y)):
    //     ...
    //
    // into:
    //
    // old_lam(a: T, b: (U, V), d: fn (W, (X, Y))):
    //     e = extract(b, 0)
    //     f = extract(b, 1)
    //     new_def(a, e, f, wrap_d)
    //
    //     wrap_d(a: W, b: X, c: Y):
    //         d(a, (b, c))

    if (unwrapped.contains(new_def)) return unwrapped[new_def]->as_lam();

    auto& world = new_def->world();
    auto new_type = new_def->type()->as<Pi>();
    auto old_lam = world.lam(old_type, new_def->debug());
    Array<const Def*> call_args(new_type->num_ops() + 1);

    unwrapped.emplace(new_def, old_lam);

    for (size_t i = 0, j = 1, e = old_lam->num_params(); i != e; ++i) {
        auto param = old_lam->param(i);
        if (auto tuple_type = param->type()->isa<TupleType>()) {
            for (size_t k = 0, e = tuple_type->num_ops(); k != e; ++k)
                call_args[j++] = world.extract(param, k);
        } else if (auto cn = param->type()->isa<Pi>()) {
            auto new_cn = new_type->op(j - 1)->as<Pi>();
            // no need to wrap if the types are identical
            if (cn != new_cn)
                call_args[j++] = wrap_def(wrapped, unwrapped, param, new_cn);
            else
                call_args[j++] = param;
        } else {
            call_args[j++] = param;
        }
    }

    call_args[0] = new_def;
    // we do not inline the call, so that we keep the flattened version around
    return app(old_lam, call_args);
}

void flatten_tuples(World& world) {
    // flatten tuples passed as arguments to functions
    bool todo = true;
    Def2Def wrapped, unwrapped;
    DefSet unwrapped_codom;

    while (todo) {
        todo = false;

        for (auto pair : unwrapped) unwrapped_codom.emplace(pair.second);

        for (auto lam : world.copy_lams()) {
            // do not change the signature of intrinsic/external functions
            if (lam->is_empty() ||
                lam->is_intrinsic() ||
                lam->is_exported() ||
                is_passed_to_accelerator(lam))
                continue;

            auto new_type = flatten_type(lam->type())->as<Pi>();
            if (new_type == lam->type()) continue;

            // do not transform lams multiple times
            if (wrapped.contains(lam) || unwrapped_codom.contains(lam)) continue;

            // generate a version of that lam that operates without tuples
            wrap_def(wrapped, unwrapped, lam, new_type);

            todo = true;

            DLOG("flattened {}", lam);
        }

        // remove original versions of wrapped functions
        auto wrapped_copy = wrapped;
        for (auto wrap_pair : wrapped_copy) {
            auto def = wrap_pair.first;
            if (def->num_ops() == 0) continue;

            auto new_lam = wrap_pair.second->as_lam();
            auto old_lam = unwrap_def(wrapped, unwrapped, new_lam, def->type()->as<Pi>());

            def->replace(old_lam);
            if (auto lam = def->isa_lam())
                lam->destroy_body();
        }
    }

    for (auto unwrap_pair : unwrapped)
        inline_calls(unwrap_pair.second->as_lam());

    world.cleanup();
    debug_verify(world);
}

}
