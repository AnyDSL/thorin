#include "thorin/transform/importer.h"

namespace thorin {

const Def* Importer::import(const Def* odef) {
    if (auto ndef = def_old2new_.lookup(odef)) {
        assert(&(*ndef)->world() == &world());
        return *ndef;
    }

    assert(required_defs.empty());
    required_defs.push(std::pair(odef, false));

    const Def* return_def = nullptr;
    while (!required_defs.empty()) {
        return_def = import_nonrecursive();
    }

    assert(return_def);
    assert(return_def == def_old2new_.lookup(odef));

    return return_def;
}

const Def* Importer::import_nonrecursive() {
    const Def* odef = required_defs.top().first;
    bool jump_to_analyze = required_defs.top().second;

    std::optional<const Def*> ndef = std::nullopt;
    if (ndef = def_old2new_.lookup(odef)) {
        assert(&(*ndef)->world() == &world());
        if (!jump_to_analyze) {
            required_defs.pop();
            return *ndef;
        }
    }

    if (!jump_to_analyze) {
        required_defs.pop();
        required_defs.push(std::pair(odef, true));
    }

    if (odef == odef->world().star()) {
        def_old2new_[odef] = world().star();
        required_defs.pop();
        return world().star();
    }

    if (!def_old2new_.lookup(odef->type())) {
        required_defs.push(std::pair(odef->type(), false));
        return nullptr;
    }
    auto ntype = import(odef->type())->as<Type>();

    Def* stub = nullptr;
    if (odef->isa_nom()) {
        if (ndef) {
            stub = (*ndef)->as_nom();
        } else {
            stub = odef->stub(world(), ntype);
            def_old2new_[odef] = stub;
        }
    }

    size_t size = odef->num_ops();
    bool unfinished = false;
    for (size_t i = 0; i != size; ++i)
        if (!def_old2new_.lookup(odef->op(i))) {
            required_defs.push(std::pair(odef->op(i), false));
            unfinished = true;
        }
    if (unfinished)
        return nullptr;

    Array<const Def*> nops(size);
    for (size_t i = 0; i != size; ++i) {
        assert(odef->op(i) != odef);
        nops[i] = import(odef->op(i));
        assert(&nops[i]->world() == &world());
    }

    if (odef->isa_structural()) {
        if (!ndef)
            ndef = odef->rebuild(world(), ntype, nops);

        if (auto oglobal = odef->isa<Global>()) {
            if (oglobal->is_external())
                world().make_external(const_cast<Def*>(*ndef));
        }

        todo_ |= odef->tag() != (*ndef)->tag();
        required_defs.pop();
        return def_old2new_[odef] = *ndef;
    } else {
        assert(odef->isa_nom() && stub);
        stub->rebuild_from(odef, nops);
        required_defs.pop();
        return stub;
    }
}

}
