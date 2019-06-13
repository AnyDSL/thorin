#include "thorin/world.h"
#include "thorin/util.h"
#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/mangle.h"
#include "thorin/analyses/verify.h"
#include "thorin/util/log.h"

#include <limits>

namespace thorin {

static Lam*   wrap_def(Def2Def&, Def2Def&, const Def*, const Pi*, size_t);
static Lam* unwrap_def(Def2Def&, Def2Def&, const Def*, const Pi*, size_t);

// Computes the type of the wrapped function
static const Def* wrapped_type(const Pi* cn, size_t max_tuple_size) {
    std::vector<const Def*> nops;
    for (auto op : cn->domains()) {
        if (auto sigma = op->isa<Sigma>()) {
            if (sigma->num_ops() <= max_tuple_size) {
                for (auto arg : sigma->ops())
                    nops.push_back(arg);
            } else
                nops.push_back(op);
        } else if (auto op_cn = op->isa<Pi>()) {
            nops.push_back(wrapped_type(op_cn, max_tuple_size));
        } else {
            nops.push_back(op);
        }
    }
    return cn->world().pi(nops, cn->codomain());
}

static Lam* app(Lam* lam, Array<const Def*>& args) {
    lam->app(args[0], args.skip_front(), args[0]->debug());
    return lam;
}

static Lam* try_inline(Lam* lam, Array<const Def*>& args) {
    if (args[0]->isa_nominal<Lam>()) {
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
        auto ulam = use->isa_nominal<Lam>();
        if (!ulam || use.index() != 0) continue;

        Array<const Def*> args(ulam->app()->num_args() + 1);
        for (size_t i = 0, e = ulam->app()->num_args(); i != e; ++i) args[i + 1] = ulam->app()->arg(i);
        args[0] = ulam->app()->callee();
        try_inline(ulam, args);
    }
}

// Wraps around a def, flattening tuples passed as parameters (dual of unwrap)
static Lam* wrap_def(Def2Def& wrapped, Def2Def& unwrapped, const Def* old_def, const Pi* new_type, size_t max_tuple_size) {
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

    if (wrapped.contains(old_def)) return wrapped[old_def]->as_nominal<Lam>();

    auto& world = old_def->world();
    auto old_type = old_def->type()->as<Pi>();
    auto new_lam = world.lam({}, new_type, old_def->debug());
    Array<const Def*> call_args(old_type->num_domains() + 1);

    wrapped.emplace(old_def, new_lam);

    for (size_t i = 0, j = 0, e = old_type->num_domains(); i != e; ++i) {
        auto op = old_type->domain(i);
        if (auto sigma = op->isa<Sigma>()) {
            if (sigma->num_ops() <= max_tuple_size) {
                Array<const Def*> tuple_args(sigma->num_ops());
                for (size_t k = 0, e = sigma->num_ops(); k != e; ++k)
                    tuple_args[k] = new_lam->param(j++);
                call_args[i + 1] = world.tuple(sigma, tuple_args);
            } else
                call_args[i + 1] = new_lam->param(j++);
        } else if (auto cn = op->isa<Pi>()) {
            auto fn_param = new_lam->param(j++);
            // no need to unwrap if the types are identical
            if (fn_param->type() != op)
                call_args[i + 1] = unwrap_def(wrapped, unwrapped, fn_param, cn, max_tuple_size);
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
static Lam* unwrap_def(Def2Def& wrapped, Def2Def& unwrapped, const Def* new_def, const Pi* old_type, size_t max_tuple_size) {
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

    if (unwrapped.contains(new_def)) return unwrapped[new_def]->as_nominal<Lam>();

    auto& world = new_def->world();
    auto new_type = new_def->type()->as<Pi>();
    auto old_lam = world.lam({}, old_type, new_def->debug());
    Array<const Def*> call_args(new_type->num_domains() + 1);

    unwrapped.emplace(new_def, old_lam);

    for (size_t i = 0, j = 1, e = old_lam->num_params(); i != e; ++i) {
        auto param = old_lam->param(i);
        if (auto sigma = param->type()->isa<Sigma>()) {
            if (sigma->num_ops() <= max_tuple_size) {
                for (size_t k = 0, e = sigma->num_ops(); k != e; ++k)
                    call_args[j++] = world.extract(param, k);
            } else
                call_args[j++] = param;
        } else if (auto cn = param->type()->isa<Pi>()) {
            auto new_cn = new_type->domain(j - 1)->as<Pi>();
            // no need to wrap if the types are identical
            if (cn != new_cn)
                call_args[j++] = wrap_def(wrapped, unwrapped, param, new_cn, max_tuple_size);
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

static void flatten_tuples(World& world, size_t max_tuple_size) {
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
                lam->is_external() ||
                is_passed_to_accelerator(lam))
                continue;

            auto new_type = wrapped_type(lam->type(), max_tuple_size)->as<Pi>();
            if (new_type == lam->type()) continue;

            // do not transform lams multiple times
            if (wrapped.contains(lam) || unwrapped_codom.contains(lam)) continue;

            // generate a version of that lam that operates without tuples
            wrap_def(wrapped, unwrapped, lam, new_type, max_tuple_size);

            todo = true;

            DLOG("flattened {}", lam);
        }

        // remove original versions of wrapped functions
        auto wrapped_copy = wrapped;
        for (auto wrap_pair : wrapped_copy) {
            auto def = wrap_pair.first;
            if (def->is_replaced()) {
                // Already replaced in previous pass
                continue;
            }

            auto new_lam = wrap_pair.second->as_nominal<Lam>();
            auto old_lam = unwrap_def(wrapped, unwrapped, new_lam, def->type()->as<Pi>(), max_tuple_size);

            def->replace(old_lam);
            if (auto lam = def->isa_nominal<Lam>())
                lam->destroy();
        }
    }

    for (auto unwrap_pair : unwrapped)
        inline_calls(unwrap_pair.second->as_nominal<Lam>());

    cleanup_world(world);
    debug_verify(world);
}

void flatten_tuples(World& world) {
    flatten_tuples(world, std::numeric_limits<size_t>::max());
}

}
