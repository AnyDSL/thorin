#include "thorin/transform/cogen/cogen.h"

#include "thorin/analyses/scope.h"
#include "thorin/world.h"
#include "thorin/type.h"
#include "thorin/def.h"
#include "thorin/lambda.h"
#include "thorin/primop.h"

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
        std::string sep = "";
        for (auto param : lambda->params()) {
            if (bta.get(param).isTop())
                continue;
            auto pname = param->name;
            def_map.insert(std::make_pair(param, pname));
            fn_proto << ", " << toCType(param->type()) << " " << pname;
        }
    }
    fn_proto << ")";

    header << fn_proto.str() << ";\n";
    source << fn_proto.str() << " {\n";

    get(lambda);
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

//------------------------------------------------------------------------------

std::string CoGen::get(Type type) {
    /* Lookup whether we already built `type'. */
    auto it = type_map.find(type);
    if (type_map.end() != it)
        return it->second;

    auto var = get_next_variable("type");
    auto gen = residualize(type);
    decls.push_back(initialize(var, gen));
    type_map.insert(std::make_pair(type, var));
    return var;
}

std::string CoGen::get(DefNode const *def) {
    /* Lookup whether we already built `def'. */
    auto it = def_map.find(def);
    if (def_map.end() != it)
        return it->second;

    auto var = get_next_variable("def");
    auto gen = residualize(def);
    decls.push_back(initialize(var, gen));
    def_map.insert(std::make_pair(def, var));
    return var;
}

FnType CoGen::extract_residual(Lambda const *lambda) {
    std::vector<Type> v;
    for (auto param : lambda->params()) {
        if (not bta.get(param).isTop())
            continue;
        v.push_back(param->type());
    }
    return world.fn_type(v);
}

std::string CoGen::residualize(Type type) {
    if (type->isa<MemTypeNode>())
        return "world.mem_type()";

    if (auto fn_t = type->isa<FnTypeNode>()) {
        std::string s = "world.fn_type({";
        std::string sep = "";
        for (auto arg : fn_t->args()) {
            s += sep + residualize(arg);
            sep = ", ";
        }
        s += "})";
        return s;
    }

    if (auto prim_t = type->isa<PrimTypeNode>()) {
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

std::string CoGen::residualize(DefNode const *def) {
    if (auto lambda = def->isa<Lambda>())
        return residualize(lambda);
    if (auto literal = def->isa<PrimLit>())
        return residualize(literal);
    if (auto arithOp = def->isa<ArithOp>())
        return residualize(arithOp);

    THORIN_UNREACHABLE; // not implemented
}

std::string CoGen::residualize(Lambda  const *lambda) { return residualize(lambda, lambda->unique_name()); }

std::string CoGen::residualize(Lambda const *lambda, std::string name) {
    auto fn_type = get(extract_residual(lambda));
    auto gen_l   = "world.lambda(" + fn_type + ", \"" + name + "\")";
    return gen_l;
}

std::string CoGen::residualize(PrimLit const *literal) {
    std::string s = "world.literal_";
    switch (literal->primtype_kind()) {
        case NodeKind::Node_PrimType_qs8:  s += "qs8("  + std::to_string(literal->qs8_value());  break;
        case NodeKind::Node_PrimType_qs16: s += "qs16(" + std::to_string(literal->qs16_value()); break;
        case NodeKind::Node_PrimType_qs32: s += "qs32(" + std::to_string(literal->qs32_value()); break;
        case NodeKind::Node_PrimType_qs64: s += "qs64(" + std::to_string(literal->qs64_value()); break;

        default: THORIN_UNREACHABLE;
    }
    return s + ")";
}

std::string CoGen::residualize(ArithOp const *arithOp) {
    auto lhs = get(arithOp->op(0));
    auto rhs = get(arithOp->op(1));

    std::string op;
    switch (arithOp->arithop_kind()) {
        case ArithOpKind::ArithOp_add: op = "+";  break;
        case ArithOpKind::ArithOp_sub: op = "-";  break;
        case ArithOpKind::ArithOp_mul: op = "*";  break;
        case ArithOpKind::ArithOp_div: op = "/";  break;
        case ArithOpKind::ArithOp_rem: op = "%";  break;
        case ArithOpKind::ArithOp_and: op = "&";  break;
        case ArithOpKind::ArithOp_or:  op = "|";  break;
        case ArithOpKind::ArithOp_xor: op = "^";  break;
        case ArithOpKind::ArithOp_shl: op = "<<"; break;
        case ArithOpKind::ArithOp_shr: op = ">>"; break;

        default: THORIN_UNREACHABLE;
    }

    return lhs + " " + op + " " + rhs;
}

}
