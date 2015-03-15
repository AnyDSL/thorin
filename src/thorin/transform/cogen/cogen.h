#ifndef THORIN_TRANSFORM_COGEN_H
#define THORIN_TRANSFORM_COGEN_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include "thorin/analyses/bta.h"

namespace thorin {

struct DefNode;
struct World;
struct Lambda;
struct PrimOp;
struct PrimLit;
struct ArithOp;

struct CoGen {
    CoGen(World &world) : world(world) { }
    void run(World &world);

    private:
    World &world;
    BTA bta;
    size_t varCount;
    size_t labelCount;
    std::ostringstream header;
    std::ostringstream source;
    std::vector<std::string> decls;
    std::vector<std::string> defs;
    std::map<DefNode const *, std::string> def_map;
    std::map<Type, std::string> type_map;

    /* Helper functions. */
    std::string get_next_variable(std::string var   = "v")     { return var   + std::to_string(varCount++); }
    std::string get_next_label   (std::string label = "label") { return label + std::to_string(labelCount++); }
    std::string assign(std::string lhs, std::string rhs) { return lhs + " = " + rhs; }
    std::string initialize(std::string lhs, std::string rhs) { return "auto " + assign(lhs, rhs); }
    void emit_preamble();
    void emit_epilogue();
    void emit_generator(Lambda *lambda);

    /* Static */
    std::string toCType(Type t);

    /* Residual */
    std::string get(Type type);
    std::string get(DefNode const *def);

    FnType extract_residual(Lambda const *lambda);

    std::string residualize(Type type);
    std::string residualize(DefNode const *def);
    std::string residualize(Lambda  const *lambda);
    std::string residualize(Lambda  const *lambda, std::string name);
    std::string residualize(PrimLit const *literal);
    std::string residualize(ArithOp const *arithOp);
};

}

#endif
