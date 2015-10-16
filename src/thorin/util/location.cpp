#include "thorin/util/location.h"

#include <cctype>
#include <iostream>

namespace thorin {

//------------------------------------------------------------------------------

static std::ostream& line_col(const Position& pos, std::ostream& os) { return os << pos.line() << " col " << pos.col(); }

std::ostream& operator << (std::ostream& os, const Position& pos) {
    return line_col(pos, os << pos.filename() << ':');
}

std::ostream& operator << (std::ostream& os, const Location& loc) {
    const Position& pos1 = loc.pos1();
    const Position& pos2 = loc.pos2();

    if (pos1.filename() != pos2.filename())
        return os << pos1 << " - " << pos2;

    os << pos1.filename() << ':';

    if (pos1.line() != pos2.line())
        return line_col(pos2, line_col(pos1, os) << " - ");

    os << pos1.line() << " col ";

    if (pos1.col() != pos2.col())
        return os << pos1.col() << " - " << pos2.col();

    return os << pos1.col();
}

}
