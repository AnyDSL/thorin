#include "thorin/world.h"

namespace thorin {

const Type* import(Type2Type& old2new, World& to, const Type* otype) {
    if (auto ntype = old2new.find(otype))
        return ntype;

    size_t size = otype->size();
    Array<const Type*> nelems(size);
    for (size_t i = 0; i != size; ++i)
        nelems[i] = import(old2new, to, otype->elem(i));
    
    return old2new[otype] = World::rebuild(to, otype, nelems);
}

Def import(Type2Type& type_old2new, Def2Def& def_old2new, World& to, Def odef) {
    if (auto ndef = def_old2new.find(odef))
            return ndef;

    auto ntype = import(type_old2new, to, odef->type());

    if (auto oparam = odef->isa<Param>()) {
        auto nlambda = import(type_old2new, def_old2new, to, oparam->lambda())->as_lambda();
        auto nparam = def_old2new.find(oparam);
        assert(nparam);
        return nparam;
    }

    if (auto olambda = odef->isa_lambda()) {
        // create stub in new world
        auto npi = import(type_old2new, to, olambda->pi())->as<Pi>();
        auto nlambda = to.lambda(npi, olambda->attribute(), olambda->name);
        for (size_t i = 0, e = olambda->num_params(); i != e; ++i) {
            nlambda->param(i)->name = olambda->param(i)->name;
            def_old2new[olambda->param(i)] = nlambda->param(i);
        }

        def_old2new[nlambda] = olambda;
    }

    size_t size = odef->size();
    Array<Def> nops(size);
    for (size_t i = 0; i != size; ++i)
        nops[i] = import(type_old2new, def_old2new, to, odef->op(i));
    
    if (auto oprimop = odef->isa<PrimOp>())
        return def_old2new[oprimop] = World::rebuild(to, oprimop, nops, ntype);
}

const Type* import(World& to, const Type* otype) {
    Type2Type old2new;
    return import(old2new, to, otype);
}

Def import(World& to, Def odef) {
    Def2Def def_old2new;
    Type2Type type_old2new;
    return import(type_old2new, def_old2new, to, odef);
}

}
