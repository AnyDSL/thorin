#include "thorin/pass/copy_prop.h"

#include "thorin/util.h"

namespace thorin {

void CopyProp::visit(Def*, Def* nom) {
    auto lam = nom->isa<Lam>();
    if (lam == nullptr || keep_.contains(lam)) return;
}

void CopyProp::enter(Def*) {
}

const Def* CopyProp::rewrite(Def*, const Def* def) {
    if (auto app = def->isa<App>()) {
        if (auto lam = app->callee()->isa_nominal<Lam>()) {
            return lam;
        }
    }

    return def;
}

undo_t CopyProp::analyze(Def* cur_nom, const Def* def) {
    if (def->isa<Param>()) return No_Undo;

    auto cur_lam = cur_nom->isa<Lam>();
    if (cur_lam == nullptr || def->isa<Param>()) return No_Undo;

    auto undo = No_Undo;
    for (size_t i = 0, e = def->num_ops(); i != e; ++i) {
        auto op = def->op(i);
        if (auto lam = op->isa<Lam>()) {
            return lam->gid();
        }
    }

    return undo;
}

}
