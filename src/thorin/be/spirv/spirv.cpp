#include "thorin/be/spirv/spirv.h"

#include <spirv/unified1/spirv.hpp>

#include <iostream>
#include <fstream>

struct SpvFileBuilder {
    SpvFileBuilder(std::ostream& output) : output(output) {}

    std::ostream& output;
    uint32_t bound = 0;

    void finish() {
        output << spv::MagicNumber;
        output << spv::Version; // TODO: target a specific spirv version
        output << uint32_t(0); // TODO get a magic number ?
        output << bound;
        output << uint32_t(0); // instruction schema padding
    }
};

struct SpvMethodBuilder {

};

thorin::SpirVCodeGen::SpirVCodeGen(thorin::World& world)
    : world_(world)
{}

void thorin::SpirVCodeGen::emit() {
    std::ofstream myfile;
    myfile.open ("test.spv");
    auto builder = SpvFileBuilder(myfile);
    builder.finish();
    myfile.close();
}