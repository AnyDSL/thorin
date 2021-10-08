#ifndef THORIN_DEBUG_H
#define THORIN_DEBUG_H

#include <string>
#include <tuple>

#include "thorin/util/hash.h"
#include "thorin/util/stream.h"

namespace thorin {

class Def;

struct Pos : public Streamable<Pos> {
    Pos() = default;
    Pos(uint32_t row, uint32_t col)
        : row(row)
        , col(col)
    {}

    Stream& stream(Stream&) const;

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

class Sym : public Streamable<Sym> {
public:
    Sym() {}
    Sym(const Def* def)
        : def_(def)
    {}

    const Def* def() const { return def_; }
    bool operator==(Sym other) const { return this->def() == other.def(); }
    Stream& stream(Stream& s) const;

private:
    const Def* def_;
};

struct SymHash {
    static hash_t hash(Sym sym);
    static bool eq(Sym a, Sym b) { return a == b; }
    static Sym sentinel() { return Sym((const Def*) 1); }
};

template<class Val>
using SymMap = HashMap<Sym, Val, SymHash>;
using SymSet = HashSet<Sym, SymHash>;

}

#endif
