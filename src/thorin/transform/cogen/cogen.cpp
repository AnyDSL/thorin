#include "thorin/transform/cogen/cogen.h"

#include "thorin/analyses/scope.h"

namespace thorin {

void CoGen::run(World &world) {
    bta.run(world);

    emit_preamble();

    /* Emit a generator for every top-level lambda. */
    Scope::for_each(world, [&](Scope const &s){
            varCount = 0;
            labelCount = 0;
            def_map.clear();
            type_map.clear();
            decls.clear();
            defs.clear();

            emit_generator(s.entry());
    });

    emit_epilogue();

    std::cout << header.str();
    std::cout << "\n\n";
    std::cout << source.str();
    std::cout << "\n\n";
}

//------------------------------------------------------------------------------

void CoGen::emit_preamble() {
    header
        << "#ifndef GENERATOR_H\n"
        << "#define GENERATOR_H\n"
        << "\n"
        ;

    source
        << "#include \"thorin/def.h\"\n"
        << "#include \"thorin/world.h\"\n"
        << "#include \"thorin/be/thorin.h\"\n"
        << "\n"
        << "using namespace thorin;\n"
        << "\n"
        ;
}

void CoGen::emit_epilogue() {
    header
        << "\n"
        << "#endif"
        ;
}

void CoGen::emit_generator(Lambda *lambda) {
    auto name = lambda->unique_name() + "_gen";

    /* Construct the function prototype. */
    std::ostringstream fn_proto;
    fn_proto << "Lambda * " << name << "(";
    fn_proto << "World &world";
    {   /* Select static params. */
        auto params = lambda->params();
        std::string sep = "";
        for (auto it = params.begin(); params.end() != it; ++it) {
            if (not bta.get(*it).isTop())
                continue;
            auto pname = (*it)->name;
            def_map.insert(std::make_pair(*it, pname));
            fn_proto << ", " << toCType((*it)->type()) << " " << pname;
        }
    }
    fn_proto << ")";

    header << fn_proto.str() << ";\n";
    source << fn_proto.str() << " {\n";

    build(lambda, lambda->unique_name() + "_spec");
    for (auto decl : decls)
        source << "    " << decl << ";\n";
    if (not decls.empty() && not defs.empty())
        source << "\n";
    for (auto def : defs)
        source << "    " << def << ";\n";

    source << "}\n";
}

//------------------------------------------------------------------------------

std::string CoGen::toCType(Type t) {
    // TODO: do something meaningful
    return "int";
}

std::string CoGen::toThorinType(Type t) {
    if (t->isa<MemTypeNode>())
        return "world.mem_type()";

    if (auto fn_t = t->isa<FnTypeNode>()) {
        std::string s = "world.fn_type({";
        std::string sep = "";
        for (auto arg : fn_t->args()) {
            s += sep + toThorinType(arg);
            sep = ", ";
        }
        s += "})";
        return s;
    }

    if (auto prim_t = t->isa<PrimTypeNode>()) {
        switch (prim_t->primtype_kind()) {
            case NodeKind::Node_PrimType_qs8:  return "world.type_qs8()";
            case NodeKind::Node_PrimType_qs16: return "world.type_qs16()";
            case NodeKind::Node_PrimType_qs32: return "world.type_qs32()";
            case NodeKind::Node_PrimType_qs64: return "world.type_qs64()";
            default: THORIN_UNREACHABLE; // not implemented
        }
    }

    THORIN_UNREACHABLE; // not implemented
}

//------------------------------------------------------------------------------

std::string CoGen::build(Type type) {
    auto it = type_map.find(type);
    if (type_map.end() != it)
        return it->second;
    auto var_t = get_next_variable("type");
    auto gen_t = toThorinType(type);
    decls.push_back(initialize(var_t, gen_t));
    type_map.insert(std::make_pair(type, var_t));
    return var_t;
}

std::string CoGen::build(DefNode const *def) {
    /* Lookup whether we already built `def'. */
    auto it = def_map.find(def);
    if (def_map.end() != it)
        return it->second;

    if (auto lambda = def->isa<Lambda>())
        return build(lambda);
    if (auto literal = def->isa<PrimLit>())
        return build(literal);

    THORIN_UNREACHABLE; // not implemented
}

std::string CoGen::build(Lambda const *lambda, std::string name) {
    auto fn_type = build(lambda->type());
    auto var_l   = get_next_variable("lambda");
    auto gen_l   = "world.lambda(" + fn_type + ", \"" + name + "\")";
    decls.push_back(initialize(var_l, gen_l));
    def_map.insert(std::make_pair(lambda, var_l));
    return var_l;
}

std::string CoGen::build(PrimLit const *literal) {
    switch (literal->primtype_kind()) {
        case NodeKind::Node_PrimType_qs8:  return "world.literal_qs8("  + std::to_string(literal->qs8_value())  + ")";
        case NodeKind::Node_PrimType_qs16: return "world.literal_qs16(" + std::to_string(literal->qs16_value()) + ")";
        case NodeKind::Node_PrimType_qs32: return "world.literal_qs32(" + std::to_string(literal->qs32_value()) + ")";
        case NodeKind::Node_PrimType_qs64: return "world.literal_qs64(" + std::to_string(literal->qs64_value()) + ")";

        default: THORIN_UNREACHABLE;
    }
}

}
