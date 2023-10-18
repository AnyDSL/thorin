#ifndef THORIN_BE_C_H
#define THORIN_BE_C_H

#include <cstdint>
#include <iostream>

#include "thorin/be/codegen.h"

namespace thorin {

using Continuations = Schedule; // vector of continuation*
using FuncMode = ChannelMode;

class World;

namespace c {

enum class Lang : uint8_t { C99, HLS, CGRA, CUDA, OpenCL };

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World& world, const Cont2Config& kernel_config, Lang lang, bool debug, std::string& flags)
        : thorin::CodeGen(world, debug)
        , kernel_config_(kernel_config)
        , lang_(lang)
        , debug_(debug)
        , flags_(flags)
    {}

    void emit_stream(std::ostream& stream) override;

    const char* file_ext() const override {
        switch (lang_) {
            case Lang::C99:    return ".c";
            case Lang::HLS:    return ".hls";
            case Lang::CGRA:   return ".cgra";
            case Lang::CUDA:   return ".cu";
            case Lang::OpenCL: return ".cl";
            default: THORIN_UNREACHABLE;
        }
    }

private:
    const Cont2Config& kernel_config_;
    Lang lang_;
    bool debug_;
    std::string flags_;
};

void emit_c_int(World&, Stream& stream);

}

}

#endif
