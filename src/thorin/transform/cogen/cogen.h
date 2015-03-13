#ifndef THORIN_TRANSFORM_COGEN_H
#define THORIN_TRANSFORM_COGEN_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include "thorin/def.h"
#include "thorin/analyses/bta.h"
#include "thorin/type.h"
#include "thorin/lambda.h"
#include "thorin/primop.h"

namespace thorin {

struct CoGen {
    void run(World &world);

    private:
    BTA bta;
    size_t varCount;
    size_t labelCount;
    std::ostringstream header;
    std::ostringstream source;
    std::vector<std::string> decls;
    std::vector<std::string> defs;
    std::map<DefNode const *, std::string> vars;

    /* Helper functions. */
    std::string get_next_variable(std::string var   = "v")     { return var   + std::to_string(varCount++); }
    std::string get_next_label   (std::string label = "label") { return label + std::to_string(labelCount++); }
    std::string assign(std::string lhs, std::string rhs) { return lhs + " = " + rhs; }
    void emit_preamble();
    void emit_epilogue();
    void emit_generator(Lambda *lambda);

    std::string toCType     (Type t);   // static
    std::string toThorinType(Type t);   // residual

    /* Residual */
    std::string build_fn_type(Lambda *lambda);
    std::string build_lambda(Lambda *lambda) { return build_lambda(lambda, lambda->unique_name()); }
    std::string build_lambda(Lambda *lambda, std::string name);
};

}

#endif
