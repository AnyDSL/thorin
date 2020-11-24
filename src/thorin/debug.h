#ifndef THORIN_DEBUG_H
#define THORIN_DEBUG_H

#include <string>
#include <tuple>
#include <variant>

#include "thorin/util/stream.h"

namespace thorin {

class Def;
class World;

struct Pos {
    const uint32_t row = -1;
    const uint32_t col = -1;
};

struct Loc : public Streamable<Loc> {
    Loc() = default;
    Loc(std::string file, Pos begin, Pos finis)
        : file(file)
        , begin(begin)
        , finis(finis)
    {}

    const std::string file;
    const Pos begin = {uint32_t(-1), uint32_t(-1)};
    const Pos finis = {uint32_t(-1), uint32_t(-1)};

    Stream& stream(Stream&) const;
};

class Dbg {
private:
    struct Data {
        const std::string name;
        const Loc loc;
        const Def* meta = nullptr;
    };

public:
    Dbg(std::string name, Loc loc = {}, const Def* meta = nullptr)
        : data_((Data){name, loc, meta})
    {}
    Dbg(Loc loc)
        : Dbg("", loc)
    {}
    Dbg(const Def* def = nullptr)
        : data_(def)
    {}

    /// @name getters
    //@{
    std::string name() const;
    Loc loc() const;
    const Def* meta() const;
    //@}

    const Def* convert(World&) const;

private:
    std::variant<Data, const Def*> data_;
};

}

#endif
