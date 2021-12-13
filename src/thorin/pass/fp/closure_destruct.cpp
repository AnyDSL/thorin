
#include "thorin/transform/closure_conv.h"
#include "thorin/pass/fp/closure_destruct.h"

namespace thorin {

const Def* ClosureDestruct::rewrite(const Def* def) {
    if (auto c = isa_closure(def)) {
        if (!escape_.contains(c.lam()) && c.env()->type() != world().sigma()) {
            auto [old_env, new_lam] = clos2dropped_[c.lam()];
            if (new_lam && c.env() == old_env)
                return new_lam;
            // TODO: Mark non-escaping, only lambda-drop bb's
            auto doms = world().sigma(Array<const Def*>(c.lam()->num_doms(), [&](auto i) {
                return (i == 0) ? world().sigma() : c.lam()->dom(i);
            }));
            new_lam = c.lam()->stub(world(), world().cn(doms), c.lam()->dbg());
            world().DLOG("drop ({}, {}) => {}", c.env(), c.lam(), new_lam);
            auto new_vars = Array<const Def*>(new_lam->num_doms(), [&](auto i) {
                return (i == 0) ? c.env() : new_lam->var(i); 
            });
            new_lam->set(c.lam()->apply(world().tuple(new_vars)));
            return world().tuple(c.type(), {world().tuple(), new_lam}, def->dbg());
            // }
        }
    }
    return def;
}

static bool interesting_type_b(const Def* type) {
    return isa_pct(type) != nullptr;
}

static bool interesting_type(const Def* type) {
    if (interesting_type_b(type))
        return true;
    if (auto sigma = type->isa<Sigma>())
        return std::any_of(sigma->ops().begin(), sigma->ops().end(), interesting_type);
    if (auto arr = type->isa<Arr>())
        return interesting_type(arr->body());
    return false;
}

static std::pair<const Def*, Def*> isa_var(const Def* a) {
    if (auto proj = a->isa<Extract>()) {
        if (auto var = proj->tuple()->isa<Var>(); var && var->nom()->isa<Lam>())
            return {a, var->nom()};
    }
    if (auto var = a->isa<Var>()) {
        if (auto lam = var->nom()->isa<Lam>()) {
            assert(lam->num_doms() == 1 && "Analyzed whole arg tuple");
            return {a, var->nom()};
        }
    }
    return {nullptr, nullptr};
}

static void split(DefSet& out, const Def* def) {
    if (auto lam = def->isa<Lam>())
        out.insert(def);
    else if (auto c = isa_closure(def))
        out.insert(c.lam());
    else if (auto [var, lam] = isa_var(def); var && lam)
        out.insert(var);
    else if (auto proj = def->isa<Extract>())
        split(out, proj->tuple());
    else if (auto pack = def->isa<Pack>())
        split(out, pack->body());
    else if (auto tuple = def->isa<Tuple>())
        for (auto op: tuple->ops())
            split(out, op);
}

static DefSet&& split(DefSet&& out, const Def* def) {
    split(out, def);
    return std::move(out);
}

undo_t ClosureDestruct::join(DefSet& out, DefSet& defs, bool cond) {
    if (!cond)
        return No_Undo;
    auto undo = No_Undo;
    for (auto def: defs) {
        if (out.contains(def))
            continue;
        out.insert(def);
        if (auto [_, lam] = isa_var(def); lam) {
            undo = std::min(undo, undo_visit(lam));
        } else {
            lam = def->isa_nom<Lam>();
            assert(lam);
            undo = std::min(undo, undo_visit(lam));
        }
    }
    return undo;
}

undo_t ClosureDestruct::join(DefSet& out, const Def* def, bool cond) {
    if (!cond)
        return No_Undo;
    auto defs = DefSet();
    split(defs, def);
    return join(out, defs, cond);
}

undo_t ClosureDestruct::analyze(const Def* def) {
    if (auto c = isa_closure(def)) {
        return join(escape_, c.env(), is_esc(c.lam()) && is_esc(c.env_var()));
    } else if (auto app = def->isa<App>()) {
        auto callees = split(DefSet(), def);
        auto undo = No_Undo;
        for (size_t i = 0; i < app->num_args(); i++) {
            auto a = app->arg(i);
            if (!interesting_type(a->type()))
                continue;
            auto cond = std::any_of(callees.begin(), callees.end(), [&](const Def* c) {
                if (auto [var, lam] = isa_var(c); var && lam)
                    return true;
                auto lam = c->isa_nom<Lam>();
                assert(lam && "callee should be lam or var");
                return is_esc(lam->var(i));
            });
            if (!cond)
                continue;
            auto args = split(DefSet(), app->arg(i));
            undo = std::min(undo, join(escape_, args, true));
        }
        return undo;
    }
    return No_Undo;
}

} // namespace thorin
