#include "thorin/util/location.h"
#include "thorin/util/stream.h"

namespace thorin {

const std::string Debug::empty_;

const std::string& Debug::name() const {
    return name_.get() ? *name_ : empty_;
}

Location operator+(Location l1, Location l2) {
    return {l1.filename(), l1.front_line(), l1.front_col(), l2.back_line(), l2.back_col()};
}

std::ostream& operator<<(std::ostream& os, Location l) {
#ifdef _MSC_VER
    return os << l.filename() << "(" << l.front_line() << ")";
#else // _MSC_VER
    os << l.filename() << ':';

    if (l.front_col() == uint16_t(-1) || l.back_col() == uint16_t(-1)) {
        if (l.front_line() != l.back_line())
            return streamf(os, "{} - {}", l.front_line(), l.back_line());
        else
            return streamf(os, "{}", l.front_line());
    }

    if (l.front_line() != l.back_line())
        return streamf(os, "{} col {} - {} col {}", l.front_line(), l.front_col(), l.back_line(), l.back_col());

    if (l.front_col() != l.back_col())
        return streamf(os, "{} col {} - {}", l.front_line(), l.front_col(), l.back_col());

    return streamf(os, "{} col {}", l.front_line(), l.front_col());
#endif // _MSC_VER
}

Debug operator+(Debug dbg, const char* s) { return {dbg, dbg.name() + s}; }
Debug operator+(Debug dbg, const std::string& s) { return {dbg, dbg.name() + s}; }

Debug operator+(Debug d1, Debug d2) {
    return {(Location)d1 + (Location)d2, d1.name() + std::string(".") + d2.name()};
}

std::ostream& operator<<(std::ostream& os, Debug dbg) {
    return streamf(os, "{{{}, {}}}", (Location)dbg, dbg.name());
}

}
