#include "thorin/util/location.h"

namespace thorin {

static std::ostream& line_col(const Position& pos, std::ostream& os) { return os << pos.line() << " col " << pos.col(); }
std::ostream& operator<<(std::ostream& os, const Position& pos) { return line_col(pos, os << pos.filename() << ':'); }

std::ostream& operator<<(std::ostream& os, const Location& loc) {
    const Position& begin = loc.begin();
    const Position& end = loc.end();

    if (begin.filename() != end.filename())
        return os << begin << " - " << end;

    os << begin.filename() << ':';

    if (begin.line() != end.line())
        return line_col(end, line_col(begin, os) << " - ");

    os << begin.line() << " col ";

    if (begin.col() != end.col())
        return os << begin.col() << " - " << end.col();

    return os << begin.col();
}

}
