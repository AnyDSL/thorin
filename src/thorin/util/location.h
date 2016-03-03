#ifndef THORIN_UTIL_LOCATION_H
#define THORIN_UTIL_LOCATION_H

#include <ostream>

namespace thorin {

//------------------------------------------------------------------------------

class Position {
public:
    Position()
        : filename_(nullptr)
        , line_(-1)
        , col_(-1)
    {}
    Position(const char* filename, int line, int col)
        : filename_(filename)
        , line_(line)
        , col_(col)
    {}

    const char* filename() const { return filename_ ? filename_ : "<unset>"; }
    int line() const { return line_; }
    int col() const { return col_; }
    bool is_set() const { return filename_ != nullptr; }

    void inc_line(int inc = 1) { line_ += inc; }
    void dec_line(int dec = 1) { line_ -= dec; }
    void inc_col(int inc = 1)  { col_ += inc; }
    void dec_col(int dec = 1)  { col_ -= dec; }
    void reset_col() { col_ = 1; }
    void reset_line() { line_ = 1; }

private:
    const char* filename_;
    int line_;
    int col_;
};

//------------------------------------------------------------------------------

class Location {
public:
    Location() {}
    Location(const Position& begin, const Position& end)
        : begin_(begin)
        , end_(end)
    {}
    Location(const Position& begin)
        : begin_(begin)
        , end_(begin)
    {}
    Location(const char* filename, int line1, int col1, int line2, int col2)
        : begin_(filename, line1, col1)
        , end_(filename, line2, col2)
    {}

    const Position& begin() const { return begin_; }
    const Position& end() const { return end_; }
    void set_begin(const Position& begin) { begin_ = begin; }
    void set_end(const Position& end) { end_ = end; }
    bool is_set() const { return begin().is_set() && end().is_set(); }

protected:
    Position begin_;
    Position end_;

    friend class HasLocation;
};

//------------------------------------------------------------------------------

class HasLocation {
public:
    HasLocation() {}
    HasLocation(const Position& pos)
        : loc_(pos, pos)
    {}
    HasLocation(const Position& begin, const Position& end)
        : loc_(begin, end)
    {}
    HasLocation(const Location& loc)
        : loc_(loc)
    {}

    const Location& loc() const  { return loc_; }
    void set_begin(const Position& begin) { loc_.begin_ = begin; }
    void set_end(const Position& end) { loc_.end_ = end; }
    void set_loc(const Location& loc) { loc_ = loc; }
    void set_loc(const Position& begin, const Position& end) { loc_.begin_ = begin; loc_.end_ = end; }
    void set_loc(const char* filename, int line1, int col1, int line2, int col2) {
        loc_ = Location(filename, line1, col1, line2, col2);
    }

protected:
    Location loc_;
};

//------------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, const Position& pos);
std::ostream& operator<<(std::ostream& os, const Location& loc);

//------------------------------------------------------------------------------

}

#endif
