#ifndef ANYDSL2_LOCATION_H
#define ANYDSL2_LOCATION_H

#include <string>

namespace anydsl2 {

//------------------------------------------------------------------------------

class Position {
public:
    Position() 
        : filename_("<unset>")
        , line_(-1)
        , col_(-1)
    {}
    Position(const std::string& filename, int line, int col)
        : filename_(filename)
        , line_(line)
        , col_(col)
    {}

    bool operator == (const Position& pos) const {
        return filename_ == pos.filename() && line_ == pos.line_ && col_ == pos.col_;
    }

    const std::string& filename() const { return filename_; }
    int line() const { return line_; }
    int col() const { return col_; }

    void inc_line(int inc = 1) { line_ += inc; }
    void dec_line(int dec = 1) { line_ -= dec; }
    void inc_col(int inc = 1)  { col_ += inc; }
    void dec_col(int dec = 1)  { col_ -= dec; }
    void reset_col() { col_ = 1; }
    void reset_line() { line_ = 1; }
    std::ostream& line_col(std::ostream& os) const;
    std::ostream& error() const;
    std::ostream& warning() const;
    bool is_set() const { return line_ != -1; }

private:
    std::string filename_;
    int line_;
    int col_;
};

//------------------------------------------------------------------------------

class Location {
public:
    Location() {}
    Location(const Position& pos1, const Position& pos2)
        : pos1_(pos1)
        , pos2_(pos2)
    {}
    Location(const Position& pos1)
        : pos1_(pos1)
        , pos2_(pos1)
    {}
    Location(const std::string& filename, int line1, int col1, int line2, int col2)
        : pos1_( Position(filename, line1, col1) )
        , pos2_( Position(filename, line2, col2) )
    {}

    bool operator == (const Location& loc) const { return pos1_ == loc.pos1() && pos2_ == loc.pos2(); }
    bool is_set() const { return pos1_.is_set() && pos2_.is_set(); }
    const Position& pos1() const { return pos1_; }
    const Position& pos2() const { return pos2_; }
    void set_pos1(const Position& pos1) { pos1_ = pos1; }
    void set_pos2(const Position& pos2) { pos2_ = pos2; }
    std::ostream& error() const;
    std::ostream& warning() const;

protected:
    Position pos1_;
    Position pos2_;

    friend class HasLocation;
};

//------------------------------------------------------------------------------

class HasLocation {
public:
    HasLocation() {}
    HasLocation(const Position& pos)
        : loc_(pos, pos)
    {}
    HasLocation(const Position& pos1, const Position& pos2)
        : loc_(pos1, pos2)
    {}
    HasLocation(const Location& loc)
        : loc_(loc)
    {}

    const Location& loc() const  { return loc_; }
    const Position& pos1() const { return loc_.pos1(); }
    const Position& pos2() const { return loc_.pos2(); }
    void set_pos1(const Position& pos1) { loc_.pos1_ = pos1; }
    void set_pos2(const Position& pos2) { loc_.pos2_ = pos2; }
    void set_loc(const Location& loc) { loc_ = loc; }
    void set_loc(const Position& pos1, const Position& pos2) { loc_.pos1_ = pos1; loc_.pos2_ = pos2; }
    std::ostream& error() const;
    std::ostream& warning() const;

protected:
    Location loc_;
};

//------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& os, const Position& pos);
std::ostream& operator << (std::ostream& os, const Location& loc);

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif // DSLU_LOCATION_H
