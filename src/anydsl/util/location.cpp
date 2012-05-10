#include "anydsl/util/location.h"

#include <cctype>
#include <iostream>

#include "anydsl/util/lexhelpers.h"
#include "anydsl/util/stdlib.h"

namespace anydsl {

//------------------------------------------------------------------------------

template<class Pred>
static bool accept(Pred pred, const char*& i, std::string& str) {
    if (*i != 0 && pred(*i)) {
        str += i++;
        return true;
    }
    return false;
}

static void lexFilename(const char*& i, std::string& filename) {
    while (*i != 0 && *i != ':' && *i != ' ') filename += *i++;
}

static void lexNunmber(const char*& i, std::string& filename) {
    while (*i != 0 && std::isdigit(*i))
        filename += *i++;
}

static void eatCol(const char*& i) {
    if (*i == ' ') ++i;
    if (*i == 'c') ++i;
    if (*i == 'o') ++i;
    if (*i == 'l') ++i;
    if (*i == ' ') ++i;
}

static void eatDash(const char*& i) {
    if (*i == ' ') ++i;
    if (*i == '-') ++i;
    if (*i == ' ') ++i;
}

bool Location::parseLocation(const std::string& s) {
    std::string filename1, filename2;
    std::string item;
    int line1, line2, col1, col2;

    const char* i = &*s.begin();

    lexFilename(i, filename1);
    if (*i == ':') ++i;
    lexNunmber(i, item);
    line1 = strtol(item.c_str(), 0, 0);
    eatCol(i);
    item.clear();
    lexNunmber(i, item);
    col1 = strtol(item.c_str(), 0, 0);

    if (*i == 0) {      // filename1:line1 col col2
        pos1_ = pos2_ = Position(filename1, line1, col1);
        return true;
    }

    eatDash(i);

    // may be a filename or a number
    item.clear();
    lexFilename(i, item);
    if (*i == ':') {        // filename1:line1 col col1 - filename2.:line2 col col2
        ++i; // eat ':'
        filename2 = item;

        item.clear();
        lexNunmber(i, item);
        line2 = strtol(item.c_str(), 0, 0);

        eatCol(i);

        item.clear();
        lexNunmber(i, item);
        col2 = strtol(item.c_str(), 0, 0);
    } else if (*i == ' ') { // filename1:line1 col col1 - line2. col col2
        eatCol(i);

        line2 = strtol(item.c_str(), 0, 0);

        item.clear();
        lexNunmber(i, item);
        col2 = strtol(item.c_str(), 0, 0);
        filename2 = filename1;
    } else {                // filename1:line1 col col1 - col2.
        col2 = strtol(item.c_str(), 0, 0);
        line2 = line1;
        filename2 = filename1;
    }

    if (*i != 0)
        return false;

    pos1_ = Position(filename1, line1, col1);
    pos2_ = Position(filename2, line2, col2);

    return true;
}

//------------------------------------------------------------------------------

bool Position::operator == (const Position& pos) const {
    return filename_ == pos.filename() && line_ == pos.line_ && col_ == pos.col_;
}

std::ostream& Position::line_col(std::ostream& os) const {
    return os << line_ << " col " << col_;
}

bool Location::operator == (const Location& loc) const {
    return pos1_ == loc.pos1() && pos2_ == loc.pos2();
}

//------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& os, const Position& pos) {
    return pos.line_col( os << pos.filename() << ':' );
}

std::ostream& operator << (std::ostream& os, const Location& loc) {
    const Position& pos1 = loc.pos1();
    const Position& pos2 = loc.pos2();

    if (pos1.filename() != pos2.filename())
        return os << pos1 << " - " << pos2;

    os << pos1.filename() << ':';

    if (pos1.line() != pos2.line())
        return pos2.line_col( pos1.line_col(os) << " - " );

    os << pos1.line() << " col ";

    if (pos1.col() != pos2.col())
        return os << pos1.col() << " - " << pos2.col();

    return os << pos1.col();
}

//------------------------------------------------------------------------------

std::ostream& Position::error() const {
    return std::cerr << *this << ": error: ";
}

std::ostream& Location::error() const {
    return std::cerr << *this << ": error: ";
}

std::ostream& HasLocation::error() const {
    return std::cerr << loc_ << ": error: ";
}

//------------------------------------------------------------------------------

} // namespace anydsl
