#include "thorin/debug.h"

#include "thorin/world.h"

namespace thorin {

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

std::string Dbg::name() const {
    if (const auto& d = std::get_if<Data>(&data_)) return d->name;
    if (auto data = std::get<const Def*>(data_)) return tuple2str(data->out(0));
    return {};
}

Loc Dbg::loc() const {
    if (const auto& d = std::get_if<Data>(&data_)) return d->loc;

    if (auto data = std::get<const Def*>(data_)) {
        auto [f, begin, finis] = data->out(1)->split<3>();
        auto file = tuple2str(f);
        u32 begin_row = as_lit(begin) >> 32_u64;
        u32 begin_col = as_lit(begin);
        u32 finis_row = as_lit(finis) >> 32_u64;
        u32 finis_col = as_lit(finis);
        return Loc{file, {begin_row, begin_col}, {finis_row, finis_col}};
    }

    return {};
}

const Def* Dbg::meta() const {
    if (const auto& d = std::get_if<Data>(&data_)) return d->meta;
    if (auto data = std::get<const Def*>(data_)) return data->out(2);
    return nullptr;
}

const Def* Dbg::convert(World& w) const {
    auto pos2def = [&](Pos pos) { return w.lit_nat((u64(pos.row) << 32_u64) | (u64(pos.col))); };

    if (const auto& d = std::get_if<Data>(&data_)) {
        auto name = w.tuple_str(d->name);
        auto file = w.tuple_str(d->loc.file);
        auto begin = pos2def(d->loc.begin);
        auto finis = pos2def(d->loc.finis);
        auto loc = w.tuple({file, begin, finis});
        return w.tuple({name, loc, d->meta ? d->meta : w.bot(w.bot_kind()) });
    }

    return std::get<const Def*>(data_);
}

}
