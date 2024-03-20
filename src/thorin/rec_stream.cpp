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

        if (cont->world().is_external(cont)) {
            if (cont->attributes().cc == CC::Thorin)
                s.fmt("intern ");
            else
                s.fmt("extern ");
        }

        Scope scope(cont);
        if (!scope.has_free_params())
            s.fmt("top_level ");
        else {
            s.fmt("// free variables: {, }\n", scope.free_params());
            s.fmt("// free frontier: {, }\n", scope.free_frontier());
        }

        if (cont->has_body()) {
            std::vector<std::string> param_names;
            for (auto param : cont->params()) param_names.push_back(param->unique_name());
            s.fmt("{}: {} = ({, }) @({}) => {{\t\n", cont->unique_name(), cont->type(), param_names, cont->filter());
            run(cont->filter());
            if (defs.contains(cont->body())) {
                auto body = cont->body();
                if (auto cont2 = body->isa_nom<Continuation>()) {
                    s.fmt("{}: {} = {}({, })", cont2->unique_name(), cont2->type(), cont2->body()->callee(), cont2->body()->args());
                } else if (!body->no_dep() && !body->isa<Param>())
                    body->stream_let(s);
            } else {
                run(cont->body()); // TODO app node
            }
            s.fmt("\b\n}}");
        } else {
            s.fmt("{}: {} = {{ <no body> }}", cont->unique_name(), cont->type());
        }
    }
}

//------------------------------------------------------------------------------

void Def::dump() const { dump(0); }
void Def::dump(size_t max) const { Stream s(std::cout); stream(s, max).endl(); }

Stream& Def::stream(Stream& s) const {
    if (isa<Type>()) return ((Type*)this)->stream(s);
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
    } else if (auto global = isa<Global>()) {
        if (global->is_external())
            return s.fmt("{}", unique_name());
        else
            return s.fmt("{}({, }))", op_name(), ops());
    }

    return s.fmt("{}({, }))", op_name(), ops());
}

Stream& Def::stream_let(Stream& s) const {
    return stream1(s.fmt("{}: {} = ", this->unique_name(), type())).endl();
}

Stream& World::stream(Stream& s) const {
    RecStreamer rec(s, std::numeric_limits<size_t>::max());
    s << "module '" << name() << "'";

    for (auto&& [_, def] : externals()) {
        auto cont = def->isa<Continuation>();
        if (cont) {
            rec.conts.push(cont);
            rec.run();
        } else {
            s.fmt("\n{} = {}({, })\n", def->unique_name(), def->op_name(), def->ops());
        }
    }

    return s.endl();
}

Stream& Scope::stream(Stream&) const {
    THORIN_UNREACHABLE;
}

Stream& Type::stream(Stream& s) const {
    if (false) {}
    else if (isa<Star>()) return s.fmt("*");
    else if (isa<BottomType>()) return s.fmt("!!");
    else if (isa<   MemType>()) return s.fmt("mem");
    else if (isa< FrameType>()) return s.fmt("frame");
    else if (auto t = isa<DefiniteArrayType>()) {
        return s.fmt("[{} x {}]", t->dim(), t->elem_type());
    } else if (auto t = isa<ClosureType>()) {
        return s.fmt("closure [{, }]", t->ops());
    } else if (auto t = isa<FnType>()) {
        return s.fmt("fn[{, }]", t->ops());
    } else if (auto t = isa<IndefiniteArrayType>()) {
        return s.fmt("[{}]", t->elem_type());
    } else if (auto t = isa<StructType>()) {
        return s.fmt("struct {}", t->name());
    } else if (auto t = isa<VariantType>()) {
        return s.fmt("variant {}", t->name());
    } else if (auto t = isa<TupleType>()) {
        return s.fmt("[{, }]", t->ops());
    } else if (auto t = isa<PtrType>()) {
        if (t->is_vector()) s.fmt("<{} x", t->length());
        s.fmt("{}*", t->pointee());
        if (t->is_vector()) s.fmt(">");
        if (t->device() != -1) s.fmt("[{}]", t->device());

        switch (t->addr_space()) {
            case AddrSpace::Global:   s.fmt("[Global]");   break;
            case AddrSpace::Texture:  s.fmt("[Tex]");      break;
            case AddrSpace::Shared:   s.fmt("[Shared]");   break;
            case AddrSpace::Constant: s.fmt("[Constant]"); break;
            default: /* ignore unknown address space */    break;
        }
        return s;
    } else if (auto t = isa<PrimType>()) {
        if (t->is_vector()) s.fmt("<{} x", t->length());

        switch (t->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case Node_PrimType_##T: s.fmt(#T); break;
#include "thorin/tables/primtypetable.h"
            default: THORIN_UNREACHABLE;
        }

        if (t->is_vector()) s.fmt(">");
        return s;
    }
    THORIN_UNREACHABLE;
}

}
