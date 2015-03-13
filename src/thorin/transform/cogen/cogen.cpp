#include "thorin/transform/cogen/cogen.h"

#include <string>
#include "thorin/analyses/scope.h"

namespace {

std::string toCType(thorin::Type t) {
    // TODO: do something meaningful
    return "int";
}

}

namespace thorin {

void CoGen::run(World &world) {
    bta.run(world);

    emit_head(std::cout);

    Scope::for_each(world, [&](Scope const & s){
            emit_generator(std::cout, s.entry());
    });
}

void CoGen::emit_generator(std::ostream &out, Lambda *lambda) {
    auto name = lambda->unique_name() + "_gen";
    std::vector<Param const *> static_params;
    std::vector<Param const *> dynamic_params;

    /* Distribute params into static and dynamic. */
    for (auto param : lambda->params()) {
        if (bta.get(param).isTop())
            dynamic_params.push_back(param);
        else
            static_params.push_back(param);
    }

    out << "Lambda * " << name << "(";
    out << "World &world";
    for (auto param : static_params) {
        out << ", " << toCType(param->type()) << " " << param->name;
    }
    out << ") {\n";

    // TODO: emit body

    out << "}\n";
}

void CoGen::emit_head(std::ostream &out) {
    out
        << "#include \"thorin/def.h\"\n"
        << "#include \"thorin/world.h\"\n"
        << "#include \"thorin/be/thorin.h\"\n"
        << "\n"
        << "using namespace thorin;\n"
        << "\n"
        ;
}

}
