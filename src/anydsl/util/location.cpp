#include "anydsl/util/location.h"

#include <cctype>
#include <iostream>

#include "anydsl/util/stdlib.h"

namespace anydsl {

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

std::ostream& Position::emitError() const {
    return std::cerr << *this << ": error: ";
}

std::ostream& Position::emitWarning() const {
    return std::cerr << *this << ": warning: ";
}

std::ostream& Location::emitError() const {
    return std::cerr << *this << ": error: ";
}

std::ostream& Location::emitWarning() const {
    return std::cerr << *this << ": warning: ";
}

std::ostream& HasLocation::emitError() const {
    return std::cerr << loc_ << ": error: ";
}

std::ostream& HasLocation::emitWarning() const {
    return std::cerr << loc_ << ": warning: ";
}

//------------------------------------------------------------------------------

} // namespace anydsl
