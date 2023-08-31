#ifndef THORIN_DEBUG_H
#define THORIN_DEBUG_H

#include <string>
#include <tuple>
#include <optional>

#include "thorin/util/stream.h"

namespace thorin {

class Def;
class World;

struct Pos {
    uint32_t row;
    uint32_t col;
};

struct Loc : public Streamable<Loc> {
    Loc() = default;
    Loc(std::string file, Pos begin, Pos finis)
        : file(file)
        , begin(begin)
        , finis(finis)
    {}
    Loc(std::string file, Pos pos)
        : Loc(file, pos, pos)
    {}
    Loc(const Def* dbg);

    Loc anew_begin() const { return {file, begin, begin}; }
    Loc anew_finis() const { return {file, finis, finis}; }

    std::string file;
    Pos begin;
    Pos finis;

    Stream& stream(Stream&) const;
};

struct Debug {
    std::string name = "";
    std::optional<Loc> loc = std::nullopt;

    inline Debug with_name(std::string new_name) {
        Debug d = *this;
        d.name = new_name;
        return d;
    }
};

}

#endif
