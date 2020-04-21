#include "thorin/def.h"
#include "thorin/util.h"
#include "thorin/world.h"

namespace thorin {

/*
 * prefixes for identifiers:
 * foobar: nominal - no prefix
 * .foobar: node
 * :foobar: axiom
 * %foobar: structural
 */

Stream& stream(Stream& s, const Def* def) {
    if (false) {}
    else if (def->isa<Universe>()) return s.fmt("□");
    else if (def->isa<Nat>())      return s.fmt("nat");
    else if (auto bot = def->isa<Bot>()) return s.fmt("⊥∷{}", bot->type());
    else if (auto top = def->isa<Top>()) return s.fmt("⊤∷{}", top->type());
    else if (auto axiom = def->isa<Axiom>()) return s.fmt(":{}", axiom->name());
    else if (auto kind = def->isa<Kind>()) {
        switch (kind->tag()) {
            case Kind::Star:  return s.fmt("★");
            case Kind::Multi: return s.fmt("•");
            case Kind::Arity: return s.fmt("◦");
            default: THORIN_UNREACHABLE;
        }
    } else if (auto lit = def->isa<Lit>()) {
        if (auto real = thorin::isa<Tag::Real>(lit->type())) {
            switch (as_lit<nat_t>(real->arg())) {
                case 16: return s.fmt("{}∷r16", lit->get<r16>());
                case 32: return s.fmt("{}∷r32", lit->get<r32>());
                case 64: return s.fmt("{}∷r64", lit->get<r64>());
                default: THORIN_UNREACHABLE;
            }
        } else if (isa<Kind>(Kind::Arity, lit->type()->type())) {
            if (!lit->type()->isa<Top>()) {
                // append utf-8 subscripts in reverse order
                std::string str;
                for (size_t aa = as_lit<nat_t>(lit->type()); aa > 0; aa /= 10)
                    ((str += char(char(0x80) + char(aa % 10))) += char(0x82)) += char(0xe2);
                std::reverse(str.begin(), str.end());

                return s.fmt("{}{}", lit->get(), str);
            }
        }

        return s.fmt("{}∷{}", lit->get(), lit->type());
    } else if (auto pi = def->isa<Pi>()) {
        if (pi->is_cn()) {
            return s.fmt("cn {}", pi->domain());
        } else {
            return s.fmt("{} -> {}", pi->domain(), pi->codomain());
        }
    } else if (auto lam = def->isa<Lam>()) {
        return s.fmt("λ@({}) {}", lam->filter(), lam->body());
    } else if (auto app = def->isa<App>()) {
        if (auto w = get_width(app)) {
            if (auto _int = thorin::isa<Tag:: Int>(app)) return s.fmt("i{}", *w);
            if (auto sint = thorin::isa<Tag::SInt>(app)) return s.fmt("s{}", *w);
            if (auto real = thorin::isa<Tag::Real>(app)) return s.fmt("r{}", *w);
            THORIN_UNREACHABLE;
        } else if (auto ptr = thorin::isa<Tag::Ptr>(app)) {
            auto [pointee, addr_space] = ptr->args<2>();
            if (auto as = isa_lit<nat_t>(addr_space); as && *as == 0)
                return s.fmt("{}*", (const Def*) pointee); // TODO why the cast???
        }

        return s.fmt("{} {}", app->callee(), app->arg());
    } else if (auto sigma = def->isa<Sigma>()) {
        return s.fmt("[{, }]", sigma->ops());
    } else if (auto tuple = def->isa<Tuple>()) {
        auto& world = def->world();
        for (size_t i = 0; i != 16; ++i) {
            if (tuple == world.table(Bit(i)))
                return s.fmt("{}", op2str(Bit(i)));
        }
        s.fmt("({, })", tuple->ops());
        return tuple->type()->isa_nominal() ? s.fmt("∷{}", tuple->type()) : s;
    } else if (auto arr = def->isa<Arr>()) {
        return s.fmt("«{}; {}»", arr->domain(), arr->codomain());
    } else if (auto pack = def->isa<Pack>()) {
        return s.fmt("‹{}; {}›", pack->domain(), pack->body());
    } else if (auto succ = def->isa<Succ>()) {
        return s.fmt("succ{}{}{}", succ->tuplefy() ? '(' : '[', succ->type(), succ->tuplefy() ? ')' : ']');
    } else if (auto union_ = def->isa<Union>()) {
        if (union_->isa_nominal()) s.fmt("{}: {}", union_->unique_name(), union_->type());
        return s.fmt("⋃({, })", union_->ops());
    }

    // other
    if (def->fields() == 0)
        return s.fmt(".{} {, }", def->node_name(), def->ops());
    return s.fmt(".{}#{} {, }", def->node_name(), def->fields(), def->ops());
}

//------------------------------------------------------------------------------

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
    unique_queue<DefSet> nominals;
    DefSet defs;
};

void RecStreamer::run(const Def* def) {
    if (def->is_const() || !defs.emplace(def).second) return;

    for (auto op : def->ops()) { // for now, don't include debug info and type
        if (auto nom = op->isa_nominal()) {
            if (max != 0) {
                if (nominals.push(nom)) --max;
            }
        } else {
            run(op);
        }
    }

    if (auto nom = def->isa_nominal())
        thorin::stream(s.endl().fmt("-> "), nom).fmt(";");
    else
        def->stream_assignment(s.endl());
}

void RecStreamer::run() {
    while (!nominals.empty()) {
        auto nom = nominals.pop();
        s.endl().endl();

        if (nom->is_set()) {
            s.fmt("{}: {} = {{\t", nom->unique_name(), nom->type());
            run(nom);
            s.fmt("\b\n}}");
        } else {
            s.fmt("{}: {} = {{ <unset> }}", nom->unique_name(), nom->type());
        }
    }
}

//------------------------------------------------------------------------------

Stream& operator<<(Stream& s, const Def* def) {
    if (def == nullptr) return s << "<nullptr>";
    return def->stream(s);
}

Stream& Def::stream(Stream& s) const {
    if (is_const()) return thorin::stream(s, this);
    return s << unique_name();
}

Stream& Def::stream(Stream& s, size_t max) const {
    if (max == 0) return stream_assignment(s);
    RecStreamer rec(s, --max);

    if (auto nom = isa_nominal()) {
        rec.nominals.push(nom);
        rec.run();
    } else {
        rec.run(this);
        if (max != 0) rec.run();
    }

    return s;
}

Stream& Def::stream_assignment(Stream& s) const {
    return thorin::stream(s.fmt("{}: {} = ", unique_name(), type()), this).fmt(";");
}

void Def::dump() const { Streamable<Def>::dump(); }

void Def::dump(size_t max) const {
    Stream s(std::cout);
    stream(s, max).endl();
}

Stream& World::stream(Stream& s) const {
    RecStreamer rec(s, std::numeric_limits<size_t>::max());
    s << "module '" << name();

    for (const auto& [name, nom] : externals()) {
        rec.nominals.push(nom);
        rec.run();
    }

    return s;
#if 0
    std::vector<const Global*> globals;

    for (auto def : defs()) {
        if (auto global = def->isa<Global>())
            globals.emplace_back(global);
    }

    for (auto global : globals)
        stream_assignment(s, global).endl();

    visit<false>([&] (const Scope& scope) {
        if (scope.entry()->isa<Axiom>()) return;
        scope.stream(s);
    });
    return s;
#endif
}

}
