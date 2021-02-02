#include "thorin/debug.h"

#include "thorin/world.h"

namespace thorin {

Loc::Loc(const Def* /*dbg*/) {
#if 0
    if (dbg != nullptr) {
        auto [d_file, d_begin, d_finis] = dbg->out(1)->split<3>();
        const_cast<std::string&>(file) = tuple2str(d_file);
        const_cast<u32&>(begin.row) = as_lit(d_begin) >> 32_u64;
        const_cast<u32&>(begin.col) = as_lit(d_begin);
        const_cast<u32&>(finis.row) = as_lit(d_finis) >> 32_u64;
        const_cast<u32&>(finis.col) = as_lit(d_finis);
    }
#endif
    assert(false && "TODO");
}

Stream& Loc::stream(Stream& s) const {
    s.fmt("{}:", file);

    if (begin.col == u32(-1) || finis.col == u32(-1)) {
        if (begin.row != finis.row)
            s.fmt("{} - {}", begin.row, finis.row);
        else
            s.fmt("{}", begin.row);
    } else if (begin.row != finis.row) {
        s.fmt("{} col {} - {} col {}", begin.row, begin.col, finis.row, finis.col);
    } else if (begin.col != finis.col) {
        s.fmt("{} col {} - {}", begin.row, begin.col, finis.col);
    } else {
        s.fmt("{} col {}", begin.row, begin.col);
    }

    return s;
}

#if 0
Debug::Debug(const Def* dbg)
    : name(dbg ? tuple2str(dbg->out(0)) : std::string{})
    , loc(dbg)
    , meta(dbg ? dbg->out(2) : nullptr)
{}
#endif

}
