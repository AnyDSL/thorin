#ifndef THORIN_BE_CONFIG_H
#define THORIN_BE_CONFIG_H

#include <cstdint>
#include <iostream>

#include "thorin/be/codegen.h"

namespace thorin {

class World;

namespace config_script {

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World& world, bool debug)
        : thorin::CodeGen(world, debug) {}

    void emit_stream(std::ostream& stream) override;
    const char* file_ext() const override { return ".cfg"; }
};

}

}

#endif
