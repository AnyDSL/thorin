#ifndef THORIN_BE_JSON_H
#define THORIN_BE_JSON_H

#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>

#include "thorin/be/codegen.h"

namespace thorin {

class World;

namespace json {

using json = nlohmann::json;

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World& world, const Cont2Config& kernel_config, bool debug)
        : thorin::CodeGen(world, debug)
        , kernel_config_(kernel_config)
    {}
    
    void emit_stream(std::ostream& stream) override;

    const char* file_ext() const override {
        return ".thorin.json";
    }

private:
    const Cont2Config& kernel_config_;
};

}

}

#endif
