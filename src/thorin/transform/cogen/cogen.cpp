#include "thorin/transform/cogen/cogen.h"

#include <cassert>
#include "thorin/analyses/scope.h"
#include "thorin/world.h"
#include "thorin/type.h"
#include "thorin/def.h"
#include "thorin/lambda.h"
#include "thorin/primop.h"


namespace {
    std::string assign(std::string lhs, std::string rhs) {
        return lhs + " = " + rhs;
    }

    std::string initialize(std::string lhs, std::string rhs) {
        return "auto " + assign(lhs, rhs);
    }
}

namespace thorin {

void CoGen::run(World &world) {
    bta.run(world);

    emit_preamble();

    /* Emit a generator for every top-level lambda. */
    Scope::for_each(world, [&](Scope const &s){
            count = 0;
            def_map_specialized.clear();
            def_map_residualized.clear();
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

void CoGen::emit_generator(Lambda const *lambda) {
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
            def_map_specialized.insert(std::make_pair(param, pname));
            fn_proto << ", " << specialize(param->type()) << " " << pname;
        }
    }
    fn_proto << ")";

    header << fn_proto.str() << ";\n";
    source << fn_proto.str() << " {\n";

    emit_code(lambda);

    /* Emit code.*/
    for (auto decl : decls)
        source << "    " << decl << ";\n";
    if (not decls.empty() && not defs.empty())
        source << "\n";
    for (auto def : defs)
        source << "    " << def << ";\n";

    source << "}\n\n";
}

void CoGen::emit_code(Lambda const *lambda) {
    for (auto op : lambda->ops())
        get_residualized(op);

    auto to = lambda->to();
    if (bta.get(to).isTop()) {
        defs.push_back("jump_to");
    } else {
        defs.push_back("goto " + def_map_specialized[to]);
    }
}


//------------------------------------------------------------------------------

FnType CoGen::extract(Lambda const *lambda, bool const residual) {
    std::vector<Type> v;
    for (auto param : lambda->params()) {
        if (residual == bta.get(param).isTop())
            v.push_back(param->type());
    }
    return world.fn_type(v);
}

std::string CoGen::get(Type type) {
    /* Lookup whether we already built `type'. */
    auto it = type_map.find(type);
    if (type_map.end() != it)
        return it->second;

    auto var = get_next("type");
    auto gen = specialize(type);
    decls.push_back(initialize(var, gen));
    type_map.insert(std::make_pair(type, var));
    return var;
}

std::string CoGen::get_residualized(DefNode const *def) {
    { /* Check residual map */
        auto it = def_map_residualized.find(def);
        if (def_map_residualized.end() != it)
            return it->second;
    }

    auto res = residualize(def);
    def_map_residualized.insert(std::make_pair(def, res));
    return res;
}

std::string CoGen::get_specialized(DefNode const *def) {
    assert(not bta.get(def).isTop() && "cannot specialze non-static program");

    auto it = def_map_specialized.find(def);
    if (def_map_specialized.end() != it)
        return it->second;

    auto var = get_next("var");
    auto gen = specialize(def);
    def_map_specialized.insert(std::make_pair(def, var));
    return var;
}

//------------------------------------------------------------------------------

std::string CoGen::specialize(Type type) {
    if (auto prim_t = type->isa<PrimTypeNode>()) {
        switch (prim_t->primtype_kind()) {
            case NodeKind::Node_PrimType_bool: return "bool";
            case NodeKind::Node_PrimType_ps8:
            case NodeKind::Node_PrimType_ps16:
            case NodeKind::Node_PrimType_ps32: return "int *";
            case NodeKind::Node_PrimType_ps64: return "long *";
            case NodeKind::Node_PrimType_pu8:
            case NodeKind::Node_PrimType_pu16:
            case NodeKind::Node_PrimType_pu32: return "unsigned *";
            case NodeKind::Node_PrimType_pu64: return "unsigned long *";
            case NodeKind::Node_PrimType_qs8:
            case NodeKind::Node_PrimType_qs16:
            case NodeKind::Node_PrimType_qs32: return "int";
            case NodeKind::Node_PrimType_qs64: return "long";
            case NodeKind::Node_PrimType_qu8:
            case NodeKind::Node_PrimType_qu16:
            case NodeKind::Node_PrimType_qu32: return "unsigned";
            case NodeKind::Node_PrimType_qu64: return "unsigned long";
            case NodeKind::Node_PrimType_pf32: return "float *";
            case NodeKind::Node_PrimType_pf64: return "double *";
            case NodeKind::Node_PrimType_qf32: return "float";
            case NodeKind::Node_PrimType_qf64: return "double";

            default: THORIN_UNREACHABLE;
        }
    }

    // TODO: implement
    THORIN_UNREACHABLE;
}

std::string CoGen::specialize(DefNode const *def) {
    assert(not bta.get(def).isTop() && "cannot specialze non-static program");

    if (auto lambda = def->isa<Lambda>())
        return specialize(lambda);
    if (auto param = def->isa<Param>())
        return specialize(param);
    if (auto literal = def->isa<PrimLit>())
        return specialize(literal);
    if (auto arithOp = def->isa<ArithOp>())
        return specialize(arithOp);

    THORIN_UNREACHABLE; // not implemented
}

std::string CoGen::specialize(Lambda const *lambda) {
    return "label_" + lambda->name + ":";
}

std::string CoGen::specialize(Param const *param) {
    return specialize(param->type()) + " " + param->unique_name();
}

std::string CoGen::specialize(PrimLit const *literal) {
    switch (literal->primtype_kind()) {
        case NodeKind::Node_PrimType_qs8:  return std::to_string(literal->qs8_value());
        case NodeKind::Node_PrimType_qs16: return std::to_string(literal->qs16_value());
        case NodeKind::Node_PrimType_qs32: return std::to_string(literal->qs32_value());
        case NodeKind::Node_PrimType_qs64: return std::to_string(literal->qs64_value());

        default: THORIN_UNREACHABLE;
    }
}

std::string CoGen::specialize(ArithOp const *arithOp) {
    auto lhs = get_specialized(arithOp->op(0));
    auto rhs = get_specialized(arithOp->op(1));

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

//------------------------------------------------------------------------------

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
            case NodeKind::Node_PrimType_bool: return "world.type_bool()";
            case NodeKind::Node_PrimType_ps8:  return "world.type_ps8()";
            case NodeKind::Node_PrimType_ps16: return "world.type_ps16()";
            case NodeKind::Node_PrimType_ps32: return "world.type_ps32()";
            case NodeKind::Node_PrimType_ps64: return "world.type_ps64()";
            case NodeKind::Node_PrimType_pu8:  return "world.type_pu8()";
            case NodeKind::Node_PrimType_pu16: return "world.type_pu16()";
            case NodeKind::Node_PrimType_pu32: return "world.type_pu32()";
            case NodeKind::Node_PrimType_pu64: return "world.type_pu64()";
            case NodeKind::Node_PrimType_qs8:  return "world.type_qs8()";
            case NodeKind::Node_PrimType_qs16: return "world.type_qs16()";
            case NodeKind::Node_PrimType_qs32: return "world.type_qs32()";
            case NodeKind::Node_PrimType_qs64: return "world.type_qs64()";
            case NodeKind::Node_PrimType_qu8:  return "world.type_qu8()";
            case NodeKind::Node_PrimType_qu16: return "world.type_qu16()";
            case NodeKind::Node_PrimType_qu32: return "world.type_qu32()";
            case NodeKind::Node_PrimType_qu64: return "world.type_qu64()";
            case NodeKind::Node_PrimType_pf32: return "world.type_pf32()";
            case NodeKind::Node_PrimType_pf64: return "world.type_pf64()";
            case NodeKind::Node_PrimType_qf32: return "world.type_qf32()";
            case NodeKind::Node_PrimType_qf64: return "world.type_qf64()";

            default: THORIN_UNREACHABLE;
        }
    }

    if (auto array_t = type->isa<ArrayTypeNode>()) {
        auto elem_t = residualize(array_t->elem_type());

        if (array_t->isa<IndefiniteArrayTypeNode>())
            return "world.indefinite_array_type(" + elem_t + ")";
        return "world.definite_array_type(" + elem_t + ")";
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

std::string CoGen::residualize(Lambda const *lambda) {
    auto fn_type = get(extract(lambda, true));
    return "world.lambda(" + fn_type + ", \"" + lambda->name + "\")";
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
    auto lhs = get_residualized(arithOp->op(0));
    auto rhs = get_residualized(arithOp->op(1));

    std::string op;
    switch (arithOp->arithop_kind()) {
        case ArithOpKind::ArithOp_add: op = "add"; break;
        case ArithOpKind::ArithOp_sub: op = "sub"; break;
        case ArithOpKind::ArithOp_mul: op = "mul"; break;
        case ArithOpKind::ArithOp_div: op = "div"; break;
        case ArithOpKind::ArithOp_rem: op = "rem"; break;
        case ArithOpKind::ArithOp_and: op = "and"; break;
        case ArithOpKind::ArithOp_or:  op = "or";  break;
        case ArithOpKind::ArithOp_xor: op = "xor"; break;
        case ArithOpKind::ArithOp_shl: op = "shl"; break;
        case ArithOpKind::ArithOp_shr: op = "shr"; break;

        default: THORIN_UNREACHABLE;
    }

    return "world.arithop_" + op + "(" + lhs + ", " + rhs + ")";
}

}
