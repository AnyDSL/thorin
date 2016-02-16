#include "thorin/util/location.h"

namespace thorin {

static std::ostream& line_col(const Position& pos, std::ostream& os) { return os << pos.line() << " col " << pos.col(); }
std::ostream& operator << (std::ostream& os, const Position& pos) { return line_col(pos, os << pos.filename() << ':'); }

std::ostream& operator << (std::ostream& os, const Location& loc) {
    const Position& first = loc.begin();
    const Position& end = loc.end();

    if (first.filename() != end.filename())
        return os << first << " - " << end;

    os << first.filename() << ':';

    if (first.line() != end.line())
        return line_col(end, line_col(first, os) << " - ");

    os << first.line() << " col ";

    if (first.col() != end.col())
        return os << first.col() << " - " << end.col();

    return os << first.col();
}

}
