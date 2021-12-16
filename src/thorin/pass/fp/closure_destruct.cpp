
#include "thorin/transform/closure_conv.h"
#include "thorin/pass/fp/closure_destruct.h"

namespace thorin {


const Def* ClosureDestruct::rewrite(const Def* def) {
    if (auto c = isa_closure(def)) {
        if (is_esc(c.lam())) {
            world().DLOG("mark escaping: {}", c.lam());
            auto dbg = def->debug();
            dbg.meta = nullptr;
            def->set_dbg(world().dbg(dbg));
            return def;
        }
        if (c.marked_no_esc())
            return def;
        auto new_dbg = ClosureWrapper::get_esc_annot(def);
        if (!c.is_basicblock()) {
            world().DLOG("mark no esc ({}, {})", c.env(), c.env_var());
            def->set_dbg(new_dbg);
            return def;
        }
        auto& [old_env, new_lam] = clos2dropped_[c.lam()];
        if (new_lam && c.env() == old_env)
            return new_lam;
        old_env = c.env();
        auto doms = world().sigma(Array<const Def*>(c.lam()->num_doms(), [&](auto i) {
            return (i == 0) ? world().sigma() : c.lam()->dom(i);
        }));
        new_lam = c.lam()->stub(world(), world().cn(doms), c.lam()->dbg());
        world().DLOG("drop ({}, {}) => {}", c.env(), c.lam(), new_lam);
        auto new_vars = Array<const Def*>(new_lam->num_doms(), [&](auto i) {
            return (i == 0) ? c.env() : new_lam->var(i); 
        });
        new_lam->set(c.lam()->apply(world().tuple(new_vars)));
        return world().tuple(c.type(), {world().tuple(), new_lam}, new_dbg);
    }
    return def;
}

static bool interesting_type_b(const Def* type) {
    return isa_ctype(type) != nullptr;
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

static void split(DefSet& out, const Def* def, bool keep_others) {
    if (auto lam = def->isa<Lam>())
        out.insert(def);
    else if (auto c = isa_closure(def))
        out.insert(c.lam());
    else if (auto [var, lam] = isa_var(def); var && lam)
        out.insert(var);
    else if (auto proj = def->isa<Extract>())
        split(out, proj->tuple(), keep_others);
    else if (auto pack = def->isa<Pack>())
        split(out, pack->body(), keep_others);
    else if (auto tuple = def->isa<Tuple>())
        for (auto op: tuple->ops())
            split(out, op, keep_others);
    else if (keep_others)
        out.insert(def);
}

static DefSet&& split(DefSet&& out, const Def* def, bool keep_others) {
    split(out, def, keep_others);
    return std::move(out);
}

bool ClosureDestruct::is_esc(const Def* def) {
    if (escape_.contains(def))
        return true;
    if (auto lam = def->isa_nom<Lam>()) {
        if (auto [_, rw] = clos2dropped_[lam]; rw && escape_.contains(rw)) {
            escape_.emplace(lam);
            return true;
        }
    }
    if (auto [var, lam] = isa_var(def); var && lam)
        return !lam->is_set();
    return false;
}

undo_t ClosureDestruct::join(DefSet& defs, bool cond) {
    if (!cond)
        return No_Undo;
    auto undo = No_Undo;
    for (auto def: defs) {
        if (is_esc(def))
            continue;
        world().DLOG("escape {}", def);
        escape_.insert(def);
        if (auto [_, lam] = isa_var(def); lam) {
            undo = std::min(undo, undo_visit(lam));
        } else {
            lam = def->isa_nom<Lam>();
            assert(lam && "should be lam or var");
            undo = std::min(undo, undo_visit(lam));
        }
    }
    return undo;
}

undo_t ClosureDestruct::join(const Def* def, bool escapes) {
    if (!escapes)
        return No_Undo;
    auto defs = split(DefSet(), def, false);
    return join(defs, escapes);
}

// store [type, space] (:mem, ptr, x)
const Def* try_get_stored(const Def* def) {
    if (auto top_app = def->isa<App>()) 
        if (auto head = top_app->callee()->isa<App>(); head && head->axiom())
            if (head->axiom()->tag() == Tag::Store)
                return top_app->arg(2_u64); 
    return nullptr;
}

undo_t ClosureDestruct::analyze(const Def* def) {
    if (auto c = isa_closure(def)) {
        world().DLOG("closure ({}, {})", c.env(), c.lam());
        return join(c.env(), is_esc(c.lam()) && is_esc(c.env_var()));
    } else if (auto stored = try_get_stored(def)) {
        return join(stored, true);
    } else if (auto app = def->isa<App>(); app && app->callee_type()->is_cn()) {
        auto callees = split(DefSet(), app->callee(), true);
        auto undo = No_Undo;
        world().DLOG("call {}", app);
        assert(callees.size() > 0);
        for (auto callee: callees)
            world().DLOG("callee: {}", callee);
        for (size_t i = 0; i < app->num_args(); i++) {
            auto a = app->arg(i);
            if (!interesting_type(a->type()))
                continue;
            auto escapes = std::any_of(callees.begin(), callees.end(), [&](const Def* c) {
                if (auto lam = c->isa_nom<Lam>())
                    return is_esc(lam->var(i));
                return true;
            });
            if (!escapes)
                continue;
            undo = std::min(undo, join(a, true));
        }
        return undo;
    }
    return No_Undo;
}

} // namespace thorin
