#ifndef THORIN_BE_C_H
#define THORIN_BE_C_H

#include <cstdint>
#include <iostream>
#include <variant>

#include "thorin/be/codegen.h"
#include "thorin/analyses/schedule.h"

namespace thorin {

using Continuations = Schedule; // vector of continuation*
using FuncMode = ChannelMode;
using TempTypeParams = std::vector<std::pair<size_t, const Type*>>;// for CGRA - (type-param index, type)
using ApiConfig = std::pair<size_t, std::variant<const Type*, TempTypeParams>>; // for CGRA - (number of template params, (type-param index, type))

class World;

namespace c {

enum class Lang : uint8_t { C99, HLS, CGRA, CUDA, OpenCL };
inline const char* lang_to_ext (Lang lang);

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(Thorin& thorin, const Cont2Config& kernel_config, Lang lang, bool debug, std::string& flags)
        : thorin::CodeGen(thorin, debug)
        , kernel_config_(kernel_config)
        , lang_(lang)
        , debug_(debug)
        , flags_(flags)
    {}

    void emit_stream(std::ostream& stream) override;
    void emit_stream(std::ostream& stream1, std::ostream& stream2);

    Lang get_lang () const { return lang_; };

    const char* file_ext() const override {
        return lang_to_ext(lang_);
    }

private:
    const Cont2Config& kernel_config_;
    Lang lang_;
    bool debug_;
    std::string flags_;
};

void emit_c_int(Thorin&, Stream& stream);

}

}

#endif
