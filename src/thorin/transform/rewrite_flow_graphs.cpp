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

static const Type* rewrite_type(World& world, Type2Type& types, const Type* type) {
    auto it = types.find(type);
    if (it != types.end())
        return it->second;

    auto nominal_type = type->isa<NominalType>();
    if (nominal_type) {
        auto ntype = nominal_type->stub(world);
        types[type] = ntype;
        for (size_t i = 0; i != type->num_ops(); ++i)
            ntype->set(i, rewrite_type(world, types, type->op(i)));
        return ntype;
    }

    Array<const Type*> new_ops(type->num_ops());
    for (size_t i = 0; i < type->num_ops(); ++i) {
        if (is_graph_type(type->op(i)))
            new_ops[i] = graph_type(world);
        else if (is_task_type(type->op(i)))
            new_ops[i] = task_type(world);
        else
            new_ops[i] = rewrite_type(world, types, type->op(i));
    }

    return types[type] = type->rebuild(world, new_ops);
}

static void rewrite_jump(Continuation* old_cont, Continuation* new_cont, Rewriter& rewriter) {
    Array<const Def*> args(old_cont->num_args());
    for (size_t i = 0; i< old_cont->num_args(); ++i)
        args[i] = rewriter.instantiate(old_cont->arg(i));

    auto callee = rewriter.instantiate(old_cont->callee());
    new_cont->jump(callee, args, old_cont->jump_debug());
}

static void rewrite_def(const Def* def, Type2Type& types, Rewriter& rewriter) {
    if (rewriter.old2new.count(def) || def->isa_continuation())
        return;

    for (auto op : def->ops())
        rewrite_def(op, types, rewriter);

    auto new_type = rewrite_type(def->world(), types, def->type());
    if (new_type != def->type()) {
        auto primop = def->as<PrimOp>();
        Array<const Def*> ops(def->num_ops());
        for (size_t i = 0; i < def->num_ops(); ++i)
            ops[i] = rewriter.instantiate(def->op(i));
        rewriter.old2new[primop] = primop->rebuild(ops, new_type);
        for (auto use : primop->uses())
            rewrite_def(use.def(), types, rewriter);
    } else {
        rewriter.instantiate(def);
    }
}

void rewrite_flow_graphs(World& world) {
    Rewriter rewriter;
    std::vector<std::pair<Continuation*, Continuation*>> transformed;
    Type2Type rewritten_types;
    TypeMap<bool> cache;

    for (auto cont : world.copy_continuations()) {
        bool transform = false;
        for (auto param : cont->params()) {
            if (has_task_or_graph_type(cache, param->type())) {
                transform = true;
                break;
            }
        }
        if (!transform)
            continue;

        auto new_cont = world.continuation(rewrite_type(world, rewritten_types, cont->type())->as<FnType>(), cont->attributes(), cont->debug());
        rewriter.old2new[cont] = new_cont;

        if (!cont->is_intrinsic()) {
            for (size_t i = 0; i < cont->num_params(); ++i)
                rewriter.old2new[cont->param(i)] = new_cont->param(i);
            transformed.emplace_back(new_cont, cont);
        }
    }

    for (auto pair : transformed) {
        for (auto param : pair.second->params()) {
            for (auto use : param->uses())
                rewrite_def(use.def(), rewritten_types, rewriter);
        }
    }

    for (auto pair : transformed)
        rewrite_jump(pair.second, pair.first, rewriter);

    for (auto cont : world.continuations()) {
        if (cont->empty())
            continue;
        rewrite_jump(cont, cont, rewriter);
    }

    world.cleanup();
}

}
