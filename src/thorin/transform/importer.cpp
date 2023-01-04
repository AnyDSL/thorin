#include "thorin/transform/importer.h"

namespace thorin {

const Type* Importer::import_type(const Type* otype) {
    if (auto ntype = type_old2new_.lookup(otype)) {
        assert(&(*ntype)->table() == &world_);
        return *ntype;
    }
    size_t size = otype->num_ops();

    if (auto nominal_type = otype->isa<NominalType>()) {
        auto ntype = nominal_type->stub(world_);
        type_old2new_[otype] = ntype;
        for (size_t i = 0; i != size; ++i)
            ntype->set(i, import_type(otype->op(i)));
        return ntype;
    }

    Array<const Type*> nops(size);
    for (size_t i = 0; i != size; ++i)
        nops[i] = import_type(otype->op(i));

    auto ntype = otype->rebuild(world_, nops);
    type_old2new_[otype] = ntype;
    assert(&ntype->table() == &world_);

    return ntype;
}

void Importer::enqueue(const Def* elem) {
    if (elem->isa_nom<Continuation>()) {
        if (analyzed_conts.find(elem) != analyzed_conts.end()) {
            required_defs.push(std::pair(elem, false));
        } else {
            analyzed_conts.insert(elem);
            required_defs.push(std::pair(elem, true));
        }
    } else {
        required_defs.push(std::pair(elem, false));
    }
}

const Def* Importer::import(const Def* odef) {
    if (auto ndef = def_old2new_.lookup(odef)) {
        assert(&(*ndef)->world() == &world_);
        return *ndef;
    }

    assert(required_defs.empty());
    enqueue(odef);

    const Def* return_def = nullptr;
    while (!required_defs.empty()) {
        return_def = import_nonrecursive();
    }

    assert(return_def);
    assert (return_def == def_old2new_.lookup(odef));

    analyzed_conts.clear();

    return return_def;
}

const Def* Importer::import_nonrecursive() {
    const Def* odef = required_defs.top().first;
    bool jump_to_analyze = required_defs.top().second;

    Continuation* ncontinuation = nullptr;

    if (auto ndef = def_old2new_.lookup(odef)) {
        assert(&(*ndef)->world() == &world_);
        if (odef->isa_nom<Continuation>()) {
            if (!jump_to_analyze) {
                required_defs.pop();
                return *ndef;
            }
            ncontinuation = (*ndef)->as_nom<Continuation>();
        } else {
            required_defs.pop();
            return *ndef;
        }
    }

    auto ntype = import_type(odef->type());

    if (auto oparam = odef->isa<Param>()) {
        if (!def_old2new_.lookup(oparam->continuation())) {
            enqueue(oparam->continuation());
            return nullptr;
        }
        import(oparam->continuation())->as_nom<Continuation>();
        auto nparam = def_old2new_[oparam];
        assert(nparam && &nparam->world() == &world_);
        required_defs.pop();
        return def_old2new_[oparam] = nparam;
    }

    if (auto ofilter = odef->isa<Filter>()) {
        Array<const Def*> new_conditions(ofilter->num_ops());

        bool unfinished_business = false;
        for (size_t i = 0, e = ofilter->size(); i != e; ++i)
            if (!def_old2new_.lookup(ofilter->condition(i))) {
                enqueue(ofilter->condition(i));
                unfinished_business = true;
            }
        if (unfinished_business)
            return nullptr;

        for (size_t i = 0, e = ofilter->size(); i != e; ++i)
            new_conditions[i] = import(ofilter->condition(i));
        auto nfilter = world().filter(new_conditions, ofilter->debug());
        required_defs.pop();
        return def_old2new_[ofilter] = nfilter;
    }

    if (auto ocontinuation = odef->isa_nom<Continuation>(); ocontinuation && !ncontinuation) { // create stub in new world
        assert(!ocontinuation->dead_);
        // TODO maybe we want to deal with intrinsics in a more streamlined way
        if (ocontinuation == ocontinuation->world().branch()) {
            required_defs.pop();
            return def_old2new_[ocontinuation] = world().branch();
        } else if (ocontinuation == ocontinuation->world().end_scope()) {
            required_defs.pop();
            return def_old2new_[ocontinuation] = world().end_scope();
        }
        auto npi = import_type(ocontinuation->type())->as<FnType>();
        ncontinuation = world().continuation(npi, ocontinuation->attributes(), ocontinuation->debug_history());
        assert(&ncontinuation->world() == &world());
        assert(&npi->table() == &world());
        for (size_t i = 0, e = ocontinuation->num_params(); i != e; ++i) {
            ncontinuation->param(i)->set_name(ocontinuation->param(i)->debug_history().name);
            def_old2new_[ocontinuation->param(i)] = ncontinuation->param(i);
        }

        def_old2new_[ocontinuation] = ncontinuation;

        if (ocontinuation->is_external())
            world().make_external(ncontinuation);
    }

    size_t size = odef->num_ops();
    Array<const Def*> nops(size);

    bool unfinished = false;
    for (size_t i = 0; i != size; ++i)
        if (!def_old2new_.lookup(odef->op(i))) {
            enqueue(odef->op(i));
            unfinished = true;
        }
    if (unfinished)
        return nullptr;

    for (size_t i = 0; i != size; ++i) {
        assert(odef->op(i) != odef);
        nops[i] = import(odef->op(i));
        assert(&nops[i]->world() == &world());
    }

    if (odef->isa_structural()) {
        auto ndef = odef->rebuild(world(), ntype, nops);

        if (auto oglobal = odef->isa<Global>()) {
            if (oglobal->is_external())
                world().make_external(const_cast<Def*>(ndef));
        }


        todo_ |= odef->tag() != ndef->tag();
        required_defs.pop();
        return def_old2new_[odef] = ndef;
    }

    assert(ncontinuation && &ncontinuation->world() == &world());
    auto napp = nops[0]->isa<App>();
    if (napp)
        ncontinuation->set_body(napp);
    ncontinuation->set_filter(nops[1]->as<Filter>());
    ncontinuation->verify();
    required_defs.pop();
    return ncontinuation;
}

}
