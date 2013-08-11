#include "anydsl2/util/location.h"

#include <cctype>
#include <iostream>

#include "anydsl2/util/stdlib.h"

namespace anydsl2 {

//------------------------------------------------------------------------------

std::ostream& Position::line_col(std::ostream& os) const { return os << line_ << " col " << col_; }

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

std::ostream& Position::error() const { return std::cerr << *this << ": error: "; }
std::ostream& Position::warning() const { return std::cerr << *this << ": warning: "; }
std::ostream& Location::error() const { return std::cerr << *this << ": error: "; }
std::ostream& Location::warning() const { return std::cerr << *this << ": warning: "; }
std::ostream& HasLocation::error() const { return std::cerr << loc_ << ": error: "; }
std::ostream& HasLocation::warning() const { return std::cerr << loc_ << ": warning: "; }

//------------------------------------------------------------------------------

} // namespace anydsl2
