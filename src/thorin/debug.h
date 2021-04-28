#ifndef THORIN_DEBUG_H
#define THORIN_DEBUG_H

#include <string>
#include <tuple>

#include "thorin/util/stream.h"

namespace thorin {

class Def;
class World;

struct Pos {
    uint32_t row = -1;
    uint32_t col = -1;
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
    Pos begin = {uint32_t(-1), uint32_t(-1)};
    Pos finis = {uint32_t(-1), uint32_t(-1)};

    Stream& stream(Stream&) const;
};

inline bool operator==(Pos p1, Pos p2) { return p1.row == p2.row && p1.col == p2.col; }
inline bool operator==(Loc l1, Loc l2) { return l1.begin == l2.begin && l1.finis == l2.finis && l1.file == l2.file; }

class Debug {
public:
    Debug(std::string name, Loc loc = {}, const Def* meta = nullptr)
        : name(name)
        , loc(loc)
        , meta(meta)
    {}
    Debug(const char* name, Loc loc = {}, const Def* meta = nullptr)
        : Debug(std::string(name), loc, meta)
    {}
    Debug(Loc loc)
        : Debug("", loc)
    {}
    Debug(const Def*);

    std::string name;
    Loc loc;
    const Def* meta = nullptr;
};

}

#endif
