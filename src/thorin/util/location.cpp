#include "thorin/util/location.h"
#include "thorin/util/stream.h"

namespace thorin {

std::ostream& operator<<(std::ostream& os, Location l) {
    os << l.filename() << ':';

    if (l.front_line() != l.back_line())
        return streamf(os, "% col % - % col %", l.front_line(), l.front_col(), l.back_line(), l.back_col());

    if (l.front_col() != l.back_col())
        return streamf(os, "% col % - %", l.front_line(), l.front_col(), l.back_col());

    return streamf(os, "% col %", l.front_line(), l.front_col());
}

}
