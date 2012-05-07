#ifndef DSLU_LOCATION_H
#define DSLU_LOCATION_H

#include <string>

namespace anydsl {

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

    bool operator == (const Position& pos) const;

    const std::string& filename() const { return filename_; }
    int line() const { return line_; }
    int col() const { return col_; }

    void incLine(int inc = 1) { line_ += inc; }
    void decLine(int dec = 1) { line_ -= dec; }
    void incCol(int inc = 1)  { col_ += inc; }
    void decCol(int dec = 1)  { col_ -= dec; }
    void resetCol() { col_ = 1; }

    std::ostream& line_col(std::ostream& os) const;
    std::ostream& error() const;

    bool isSet() const { return line_ != -1; }

private:

    std::string filename_;
    int line_;
    int col_;
};

//------------------------------------------------------------------------------

class Location {
public:

    Location() 
        : pos1_("<unknown>", -1, -1)
        , pos2_(pos1_)
    {}
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
    Location(const std::string& s) { if (!parseLocation(s)) *this = Location(); }

    bool operator == (const Location& loc) const;
    bool isSet() const { return pos1_.isSet(); }

    const Position& pos1() const { return pos1_; }
    const Position& pos2() const { return pos2_; }

    void setPos1(const Position& pos1) { pos1_ = pos1; }
    void setPos2(const Position& pos2) { pos2_ = pos2; }

    std::ostream& error() const;

protected:

    Position pos1_;
    Position pos2_;

private:

    bool parseLocation(const std::string& s);

    friend class HasLocation;
};

//------------------------------------------------------------------------------

#if 0

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
    void setPos2(const Position& pos2) { loc_.pos2_ = pos2; }
    void setLoc(const Location& loc) { loc_ = loc; }

    std::ostream& error() const;

protected:

    Location loc_;
};

#endif

//------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& os, const Position& pos);
std::ostream& operator << (std::ostream& os, const Location& loc);

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // DSLU_LOCATION_H
