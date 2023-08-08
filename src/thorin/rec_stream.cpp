#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/world.h"
#include "thorin/analyses/scope.h"
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
        if (auto cont = op->isa_nom<Continuation>()) {
            if (max != 0) {
                if (conts.push(cont)) --max;
            }
        } else {
            run(op);
        }
    }

    if (auto cont = def->isa_nom<Continuation>()) {
        assert(cont->has_body());
        s.fmt("{}: {} = {}({, })", cont, cont->type(), cont->body()->callee(), cont->body()->args());
        run(cont->body());
    } else if (!def->no_dep() && !def->isa<Param>())
        def->stream_let(s);
}

void RecStreamer::run() {
    while (!conts.empty()) {
        auto cont = conts.pop();
        s.endl().endl();

        if (cont->world().is_external(cont))
            s.fmt("extern ");

        if (cont->has_body()) {
            std::vector<std::string> param_names;
            for (auto param : cont->params()) param_names.push_back(param->unique_name());
            s.fmt("{}: {} = ({, }) => {{\t\n", cont->unique_name(), cont->type(), param_names);
            run(cont->body()); // TODO app node
            s.fmt("\b\n}}");
        } else {
            s.fmt("{}: {} = {{ <no body> }}", cont->unique_name(), cont->type());
        }
    }
}

//------------------------------------------------------------------------------

void Def::dump() const { dump(0); }
void Def::dump(size_t max) const { Stream s(std::cout); stream(s, max).endl(); }

void Type::dump() const { Stream s(std::cout); stream(s).endl(); }

Stream& Def::stream(Stream& s) const {
    if (isa<Param>() || isa<App>() || no_dep()) return stream1(s);
    return s << unique_name();
}

Stream& Def::stream(Stream& s, size_t max) const {
    switch (max) {
        case 0: return stream(s);
        case 1: return stream1(s);
        default:
            max -= 2;
            RecStreamer rec(s, max);

            if (auto cont = isa_nom<Continuation>()) {
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
        return s.fmt("{}.{}", param->continuation(), param->unique_name());
    } else if (isa<Continuation>()) {
#if THORIN_ENABLE_CREATION_CONTEXT
        if (debug().creation_context != "")
            return s.fmt("cont {} [{}]", unique_name(), debug().creation_context);
        else
#endif
            return s.fmt("cont {}", unique_name());
    } else if (auto app = isa<App>()) {
        return s.fmt("{}({, })", app->callee(), app->args());
    } else if (isa<Filter>()) {
        return s.fmt("filter({, })", ops());
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
    }

    return s.fmt("{}({, }))", op_name(), ops());
}

Stream& Def::stream_let(Stream& s) const {
    return stream1(s.fmt("{}: {} = ", this, type())).endl();
}

Stream& World::stream(Stream& s) const {
    RecStreamer rec(s, std::numeric_limits<size_t>::max());
    s << "module '" << name() << "'";

    for (auto&& [_, cont] : externals()) {
        rec.conts.push(cont);
        rec.run();
    }

    return s.endl();
}

Stream& Scope::stream(Stream& s) const {
    THORIN_UNREACHABLE;
}

}
