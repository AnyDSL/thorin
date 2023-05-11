#ifndef THORIN_BE_CONFIG_H
#define THORIN_BE_CONFIG_H

#include <cstdint>
#include <iostream>

#include "thorin/be/codegen.h"
#include "thorin/transform/hls_dataflow.h"
#include "thorin/transform/cgra_dataflow.h"

namespace thorin {

class World;

namespace config_script {
    //static std::pair<int, int> test;
    //TODO: we need to fill test_pair when making the script codegen obj in codegen.cpp!
    // on frontend side we use default param

class CodeGen : public thorin::CodeGen {
public:
    //CodeGen(World& world, bool debug, std::pair<int, int> test_pair = std::make_pair(0,0))
    CodeGen(World& world, bool debug, Ports& hls_cgra_ports, std::string& flags)
    //CodeGen(World& world, bool debug, Ports& hls_cgra_ports)
        : thorin::CodeGen(world, debug)
        , hls_cgra_ports_(hls_cgra_ports)
        , flags_(flags)
    {}

    void emit_stream(std::ostream& stream) override;
    const char* file_ext() const override { return ".cfg"; }

private:
    //std::pair<int, int> test_pair_;
    Ports hls_cgra_ports_;
    std::string flags_;
};

}

}

#endif
