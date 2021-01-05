#include "thorin/lam.h"

#include "thorin/world.h"

namespace thorin {

Lam* Lam::set_filter(bool filter) { return set_filter(world().lit_bool(filter)); }

const Def* Lam::mem_var(const Def* dbg) {
    return thorin::isa<Tag::Mem>(var(0_s)->type()) ? var(0, dbg) : nullptr;
}

const Def* Lam::ret_var(const Def* dbg) {
    if (num_vars() > 0) {
        auto p = var(num_vars() - 1, dbg);
        if (auto pi = p->type()->isa<thorin::Pi>(); pi != nullptr && pi->is_cn()) return p;
    }
    return nullptr;
}

bool Lam::is_basicblock() const { return type()->is_basicblock(); }
bool Lam::is_returning() const { return type()->is_returning(); }

void Lam::app(const Def* callee, const Def* arg, const Def* dbg) {
    assert(isa_nom());
    auto filter = world().lit_false();
    set(filter, world().app(callee, arg, dbg));
}

void Lam::app(const Def* callee, Defs args, const Def* dbg) { app(callee, world().tuple(args), dbg); }

void Lam::branch(const Def* cond, const Def* t, const Def* f, const Def* mem, const Def* dbg) {
    return app(world().select(t, f, cond), mem, dbg);
}

void Lam::test(const Def* value, const Def* index, const Def* match, const Def* clash, const Def* mem, const Def* dbg) {
    return app(world().test(value, index, match, clash), {mem}, dbg);
}

/*
 * Pi
 */

Pi* Pi::set_dom(Defs doms) { return Def::set(0, world().sigma(doms))->as<Pi>(); }

bool Pi::is_cn() const { return codom()->isa<Bot>(); }

bool Pi::is_returning() const {
    bool ret = false;
    for (auto op : ops()) {
        switch (op->order()) {
            case 1:
                if (!ret) {
                    ret = true;
                    continue;
                }
                return false;
            default: continue;
        }
    }
    return ret;
}

// TODO remove
Lam* get_var_lam(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return extract->tuple()->as<Var>()->nom()->as<Lam>();
    return def->as<Var>()->nom()->as<Lam>();
}

// TODO remove
size_t get_var_index(const Def* def) {
    if (auto extract = def->isa<Extract>())
        return as_lit<size_t>(extract->index());
    assert(def->isa<Var>());
    return 0;
}

std::vector<Peek> peek(const Def* var) {
    std::vector<Peek> peeks;
    size_t index = get_var_index(var);
    for (auto use : get_var_lam(var)->uses()) {
        if (auto app = use->isa<App>()) {
            for (auto use : app->uses()) {
                if (auto pred = use->isa_nom<Lam>()) {
                    if (pred->body() == app)
                        peeks.emplace_back(app->arg(index), pred);
                }
            }
        }
    }

    return peeks;
}

}
