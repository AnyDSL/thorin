#include "thorin/transform/mangle.h"
#include "thorin/world.h"

namespace thorin {

static bool is_task_type(const Type* type) {
    if (auto struct_type = type->isa<StructType>())
        return struct_type->name().str() == "FlowTask";
    return false;
}

static bool is_graph_type(const Type* type) {
    if (auto struct_type = type->isa<StructType>())
        return struct_type->name().str() == "FlowGraph";
    return false;
}

static bool has_task_or_graph_type(TypeMap<bool>& cache, const Type* type) {
    if (cache.emplace(type, false).second) {
        if (is_task_type(type) || is_graph_type(type))
            return cache[type] = true;
        bool contains = false;
        for (auto op : type->ops()) {
            if (has_task_or_graph_type(cache, op)) {
                contains = true;
                break;
            }
        }
        return cache[type] = contains;
    }

    return cache[type];
}

static const Type* task_type(World& world) {
    return world.type_qs32();
}

static const Type* graph_type(World& world) {
    return world.type_qs32();
}

static const Type* rewrite_type(World& world, const Type* type) {
    Array<const Type*> new_ops(type->num_ops());
    for (size_t i = 0; i < type->num_ops(); ++i) {
        if (is_graph_type(type->op(i)))
            new_ops[i] = graph_type(world);
        else if (is_task_type(type->op(i)))
            new_ops[i] = task_type(world);
        else
            new_ops[i] = rewrite_type(world, type->op(i));
    }

    return type->rebuild(world, new_ops);
}

static void rewrite_jump(Lam* old_lam, Lam* new_lam, Rewriter& rewriter) {
    Array<const Def*> args(old_lam->app()->num_args());
    for (size_t i = 0; i < old_lam->app()->num_args(); ++i)
        args[i] = rewriter.instantiate(old_lam->app()->arg(i));

    auto callee = rewriter.instantiate(old_lam->app()->callee());
    new_lam->app(callee, args, old_lam->app()->debug());
}

static void rewrite_def(const Def* def, Rewriter& rewriter) {
    if (rewriter.old2new.count(def) || def->isa_lam())
        return;

    for (auto op : def->ops())
        rewrite_def(op, rewriter);

    auto new_type = rewrite_type(def->world(), def->type());
    if (new_type != def->type()) {
        auto primop = def->as<PrimOp>();
        Array<const Def*> ops(def->num_ops());
        for (size_t i = 0; i < def->num_ops(); ++i)
            ops[i] = rewriter.instantiate(def->op(i));
        rewriter.old2new[primop] = primop->rebuild(new_type, ops);
        for (auto use : primop->uses())
            rewrite_def(use.def(), rewriter);
    } else {
        rewriter.instantiate(def);
    }
}

void rewrite_flow_graphs(World& world) {
    Rewriter rewriter;
    std::vector<std::pair<Lam*, Lam*>> transformed;
    TypeMap<bool> cache;

    for (auto lam : world.copy_lams()) {
        bool transform = false;
        for (auto param : lam->params()) {
            if (has_task_or_graph_type(cache, param->type())) {
                transform = true;
                break;
            }
        }
        if (!transform)
            continue;

        auto new_lam = world.lam(rewrite_type(world, lam->type())->as<Pi>(), lam->debug());
        if (lam->is_external())
            new_lam->make_external();
        rewriter.old2new[lam] = new_lam;

        if (!lam->is_intrinsic()) {
            for (size_t i = 0; i < lam->num_params(); ++i)
                rewriter.old2new[lam->param(i)] = new_lam->param(i);
            transformed.emplace_back(new_lam, lam);
        }
    }

    for (auto p : transformed) {
        for (auto use : p.second->param()->uses())
            rewrite_def(use.def(), rewriter);
    }

    for (auto pair : transformed)
        rewrite_jump(pair.second, pair.first, rewriter);

    for (auto lam : world.copy_lams())
        rewrite_jump(lam, lam, rewriter);

    world.cleanup();
}

}
