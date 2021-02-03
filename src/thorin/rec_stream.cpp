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
    if (is_const(def) || !defs.emplace(def).second) return;

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
    else
        def->stream_let(s.endl());
}

void RecStreamer::run() {
    while (!conts.empty()) {
        auto cont = conts.pop();
        s.endl().endl();

        if (!cont->empty()) {
            s.fmt("{}: {} = {{\t", cont->unique_name(), cont->type());
            run(cont);
            s.fmt("\b\n}}");
        } else {
            s.fmt("{}: {} = {{ <unset> }}", cont->unique_name(), cont->type());
        }
    }
}

//------------------------------------------------------------------------------

Stream& Def::stream(Stream& s) const { return s << unique_name(); }

Stream& Def::stream_let(Stream& s) const {
    s.fmt("{}: {} =", this, type());

    if (auto lit = isa<PrimLit>()) {
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
        s.fmt(" : ({, })\n", ass->output_constraints());
        s.fmt(" : ({, })\n", ass->input_constraints());
        s.fmt(" : ({, })\n", ass->clobbers());
        s.fmt(   "({, })\b", ass->ops());
        return s;
    } else if (auto primop = isa<PrimOp>()) {
        return s.fmt("{}({, }))", primop->op_name(), ops());
    }
    THORIN_UNREACHABLE;
}

/*
 * Scope
 */

#if 0
std::ostream& Scope::stream(std::ostream& os) const {
    for (auto cont : schedule(*this)) {
    }

    return os; /* TODO return schedule(*this).stream(os); */
}

void Scope::write_thorin([[maybe_unused]] const char* filename) const { /* TODO return schedule(*this).write_thorin(filename); */ }
void Scope::thorin() const { /* TODO schedule(*this).thorin(); */ }
#endif

/*
 * World
 */

//------------------------------------------------------------------------------

Stream& World::stream(Stream& s) const {
    return s;
}

#if 0
std::ostream& World::stream(std::ostream& os) const {
    os << "module '" << name() << "'\n\n";

    for (auto primop : primops()) {
        if (auto global = primop->isa<Global>())
            global->stream_assignment(os);
    }

    Scope::for_each<false>(*this, [&] (const Scope& scope) { scope.stream(os); });
    return os;
}

void World::write_thorin(const char* filename) const { std::ofstream file(filename); stream(file); }

void World::thorin() const {
    auto filename = name() + ".thorin";
    write_thorin(filename.c_str());
}
#endif

}
