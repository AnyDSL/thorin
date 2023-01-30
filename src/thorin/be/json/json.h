#ifndef THORIN_BE_JSON_H
#define THORIN_BE_JSON_H

#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>

#include "thorin/be/codegen.h"

namespace thorin {

class Thorin;

namespace json {

using json = nlohmann::json;

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(Thorin& thorin, bool debug, std::string& target_triple, std::string& target_cpu, std::string& target_attr)
        : thorin::CodeGen(thorin, debug)
        , target_triple(target_triple)
        , target_cpu(target_cpu)
        , target_attr(target_attr)
    {}
    
    void emit_stream(std::ostream& stream) override;

    const char* file_ext() const override {
        return ".thorin.json";
    }
private:
    std::string& target_triple;
    std::string& target_cpu;
    std::string& target_attr;
};

}

}

#endif
