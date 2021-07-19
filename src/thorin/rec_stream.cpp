#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/util/utility.h"

namespace thorin {

class RecStreamer {
public:
    RecStreamer(Stream& s, size_t max)
        : s(s)
        , max(max)
    {}

    void run();
    void run(const Def*);

    Stream& s;
    size_t max;
    unique_queue<ContinuationSet> conts;
    DefSet defs;
};

void RecStreamer::run(const Def* def) {
    if (def->no_dep() || !defs.emplace(def).second) return;

    for (auto op : def->ops()) { // for now, don't include debug info and type
        if (auto cont = op->isa_continuation()) {
            if (max != 0) {
                if (conts.push(cont)) --max;
            }
        } else {
            run(op);
        }
    }

    if (auto cont = def->isa_continuation())
        s.fmt("{}: {} = {}({, })", cont, cont->type(), cont->callee(), cont->args());
    else if (!def->no_dep() && !def->isa<Param>())
        def->stream_let(s);
}

void RecStreamer::run() {
    while (!conts.empty()) {
        auto cont = conts.pop();
        s.endl().endl();

        if (!cont->empty()) {
            s.fmt("{}: {} = {{\t\n", cont->unique_name(), cont->type());
            run(cont);
            s.fmt("\b\n}}");
        } else {
            s.fmt("{}: {} = {{ <unset> }}", cont->unique_name(), cont->type());
        }
    }
}

//------------------------------------------------------------------------------

void Def::dump() const { dump(0); }
void Def::dump(size_t max) const { Stream s(std::cout); stream(s, max).endl(); }

Stream& Def::stream(Stream& s) const {
    if (isa<Param>() || no_dep()) return stream1(s);
    return s << unique_name();
}

Stream& Def::stream(Stream& s, size_t max) const {
    switch (max) {
        case 0: return stream(s);
        case 1: return stream1(s);
        default:
            max -= 2;
            RecStreamer rec(s, max);

            if (auto cont = isa_continuation()) {
                rec.conts.push(cont);
                rec.run();
            } else {
                rec.run(this);
                if (max != 0) rec.run();
            }

            return s;
    }
}

Stream& Def::stream1(Stream& s) const {
    if (auto param = isa<Param>()) {
        return s.fmt("{}[{}]", param->unique_name(), param->continuation());
    } else if (auto cont = isa<Continuation>()) {
        return s.fmt("{}({, })", cont->callee(), cont->args());
    } else if (auto lit = isa<PrimLit>()) {
        // print i8 as ints
        switch (lit->tag()) {
            case PrimType_qs8: return s.fmt("{}∷qs8", (int)      lit->qs8_value());
            case PrimType_ps8: return s.fmt("{}∷ps8", (int)      lit->ps8_value());
            case PrimType_qu8: return s.fmt("{}∷qu8", (unsigned) lit->qu8_value());
            case PrimType_pu8: return s.fmt("{}∷pu8", (unsigned) lit->pu8_value());
            default:
                switch (lit->tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: return s.fmt("{}∷{}", lit->value().get_##M(), #T);
#include "thorin/tables/primtypetable.h"
                    default: THORIN_UNREACHABLE;
                }
        }
    } else if (auto ass = isa<Assembly>()) {
        s.fmt("{} {} = asm \"{}\"\t\n", ass->type(), ass->unique_name(), ass->asm_template());
        s.fmt(": ({, })\n", ass->output_constraints());
        s.fmt(": ({, })\n", ass->input_constraints());
        s.fmt(": ({, })\n", ass->clobbers());
        s.fmt(": ({, })\b", ass->ops());
        return s;
    } else if (auto primop = isa<PrimOp>()) {
        return s.fmt("{}({, }))", primop->op_name(), ops());
    }
    THORIN_UNREACHABLE;
}

Stream& Def::stream_let(Stream& s) const {
    return stream1(s.fmt("{}: {} = ", this, type())).endl();
}

Stream& World::stream(Stream& s) const {
    RecStreamer rec(s, std::numeric_limits<size_t>::max());
    s << "module '" << name();

    for (auto&& [_, cont] : externals()) {
        rec.conts.push(cont);
        rec.run();
    }

    return s.endl();
}

THORIN_INSTANTIATE_STREAMABLE(World)

}
