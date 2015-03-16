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
    size_t count;
    std::ostringstream header;
    std::ostringstream source;
    std::vector<std::string> decls;
    std::vector<std::string> defs;
    std::map<Type, std::string> type_map;
    std::map<DefNode const *, std::string> def_map_specialized;
    std::map<DefNode const *, std::string> def_map_residualized;

    /* Helper functions. */
    std::string get_next(std::string name = "v") { return name + std::to_string(count++); }
    void emit_preamble();
    void emit_epilogue();
    void emit_generator(Lambda const *lambda);
    void emit_code(Lambda const *lambda);

    /// Returns a function type with only residual (or spectime) parameter types.
    FnType extract(Lambda const *lambda, bool const residual);

    /// Returns the C++ variable holding the Thorin-type `type'.  If no such variable exists yet, one is created and
    /// initialized.
    std::string get(Type type);
    std::string get(DefNode const *def);
    std::string get_residualized(DefNode const *def);
    std::string get_specialized(DefNode const *def);

    /* Static */
    std::string specialize(Type type);
    std::string specialize(DefNode const *def);
    /// The specialization of a lambda is a goto-label
    std::string specialize(Lambda  const *lambda);
    /// The specialization of a parameter is a C++ variable
    std::string specialize(Param   const *param);
    std::string specialize(PrimLit const *literal);
    std::string specialize(ArithOp const *arithOp);

    /* Residual */
    std::string residualize(Type type);
    std::string residualize(DefNode const *def);
    /// A residualized lambda is C++ code that creates a lambda
    std::string residualize(Lambda  const *lambda);
    /// A residualized param is C++ code that references the param in a lambda
    std::string residualize(Param   const *param);
    std::string residualize(PrimLit const *literal);
    std::string residualize(ArithOp const *arithOp);
};

}

#endif
