#ifndef THORIN_UTIL_LOCATION_H
#define THORIN_UTIL_LOCATION_H

#include <ostream>
#include <string>

#include "thorin/util/symbol.h"

namespace thorin {

//------------------------------------------------------------------------------

class Location {
public:
    Location() = default;

    Location(const char* filename, uint32_t front_line, uint32_t front_col, uint32_t back_line, uint32_t back_col)
        : filename_(filename)
        , front_line_(front_line)
        , front_col_(front_col)
        , back_line_(back_line)
        , back_col_(back_col)
    {}

    Location(const char* filename, uint32_t line, uint32_t col)
        : Location(filename, line, col, line, col)
    {}

    Location(Location front, Location back)
        : Location(front.filename(), front.front_line(), front.front_col(), back.back_line(), back.back_col())
    {}

    const char* filename() const { return filename_; }
    uint32_t front_line() const { return front_line_; }
    uint32_t front_col() const { return front_col_; }
    uint32_t back_line() const { return back_line_; }
    uint32_t back_col() const { return back_col_; }

    Location front() const { return {filename_, front_line(), front_col(), front_line(), front_col()}; }
    Location back() const { return {filename_, back_line(), back_col(), back_line(), back_col()}; }
    bool is_set() const { return filename_ != nullptr; }

protected:
    const char* filename_ = nullptr;
    uint16_t front_line_ = 1, front_col_ = 1, back_line_ = 1, back_col_ = 1;
};

Location operator+(Location l1, Location l2);
std::ostream& operator<<(std::ostream&, Location);

class Debug : public Location {
public:
    Debug() = default;
    Debug(Debug&&) = default;
    Debug(const Debug&) = default;
    Debug& operator=(const Debug&) = default;

    Debug(Location location, Symbol name)
        : Location(location)
        , name_(name)
    {}
    Debug(Symbol name)
        : name_(name)
    {}
    Debug(Location location)
        : Location(location)
    {}

    Symbol name() const { return name_; }
    Location location() { return *this; }
    void set(Symbol name) { name_= name; }
    void set(Location location) { *static_cast<Location*>(this) = location; }

private:
    Symbol name_;
};

inline Debug operator+(Debug dbg, Symbol s)             { return {dbg, dbg.name() + s}; }
inline Debug operator+(Debug dbg, const char* s)        { return {dbg, dbg.name() + s}; }
inline Debug operator+(Debug dbg, const std::string& s) { return {dbg, dbg.name() + s}; }
Debug operator+(Debug d1, Debug d2);

std::ostream& operator<<(std::ostream&, Debug);

}

#endif
