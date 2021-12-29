#include "thorin/def.h"
#include "thorin/world.h"
#include "thorin/analyses/deptree.h"
#include "thorin/util/container.h"

namespace thorin {

/*
 * prefixes for identifiers:
 * foobar: nom - no prefix
 * .foobar: node
 * :foobar: axiom
 * %foobar: structural
 */

bool Def::unwrap() const {
    if (isa_nom()) return false;
    if (isa<Global>()) return false;
    //if (def->no_dep()) return true;
    if (auto app = isa<App>()) {
        if (app->type()->isa<Pi>()) return true; // curried apps are printed inline
        if (app->type()->isa<Kind>() || app->type()->isa<Space>()) return true;
        if (app->callee()->isa<Axiom>()) {
            return app->callee_type()->num_doms() <= 1;
        }
        return false;
    }
    return true;
}

enum class Prec {   //  left    right
    Bottom,         //  Bottom  Bottom
    Pi,             //  App     Pi
    App,            //  App     Extract
    Extract,        //  Extract Lit
    Lit,            //  -       Lit
};

static Prec prec(const Def* def) {
    if (def->isa<Pi>())      return Prec::Pi;
    if (def->isa<App>())     return Prec::App;
    if (def->isa<Extract>()) return Prec::Extract;
    if (def->isa<Lit>())     return Prec::Lit;
    return Prec::Bottom;
}

static Prec prec_l(const Def* def) {
    assert(!def->isa<Lit>());
    if (def->isa<Pi>())      return Prec::App;
    if (def->isa<App>())     return Prec::App;
    if (def->isa<Extract>()) return Prec::Extract;
    return Prec::Bottom;
}

static Prec prec_r(const Def* def) {
    if (def->isa<Pi>())      return Prec::Pi;
    if (def->isa<App>())     return Prec::Extract;
    if (def->isa<Extract>()) return Prec::Lit;
    return Prec::Bottom;
}

template<bool L>
struct LRPrec {
    LRPrec(const Def* l, const Def* r)
        : l_(l)
        , r_(r)
    {}

    friend Stream& operator<<(Stream& s, const LRPrec& p) {
        if constexpr (L) {
            if (p.l_->unwrap() && prec_l(p.r_) > prec(p.r_)) return s.fmt("({})", p.l_);
            return s.fmt("{}", p.l_);
        } else {
            if (p.r_->unwrap() && prec(p.l_) > prec_r(p.l_)) return s.fmt("({})", p.r_);
            return s.fmt("{}", p.r_);
        }
    }

private:
    const Def* l_;
    const Def* r_;
};

using LPrec = LRPrec<true>;
using RPrec = LRPrec<false>;

Stream& Def::unwrap(Stream& s) const {
    if (false) {}
    else if (isa<Space>()) return s.fmt("□");
    else if (isa<Kind>())  return s.fmt("★");
    else if (isa<Nat>())   return s.fmt("nat");
    else if (auto bot = isa<Bot>()) return s.fmt("⊥∷{}", bot->type());
    else if (auto top = isa<Top>()) return s.fmt("⊤∷{}", top->type());
    else if (auto axiom = isa<Axiom>()) return s.fmt(":{}", axiom->debug().name);
    else if (auto lit = isa<Lit>()) {
        if (auto real = thorin::isa<Tag::Real>(lit->type())) {
            switch (as_lit(real->arg())) {
                case 16: return s.fmt("{}∷r16", lit->get<r16>());
                case 32: return s.fmt("{}∷r32", lit->get<r32>());
                case 64: return s.fmt("{}∷r64", lit->get<r64>());
                default: THORIN_UNREACHABLE;
            }
        }
        return s.fmt("{}∷{}", lit->get(), lit->type());
    } else if (auto ex = isa<Extract>()) {
        if (ex->tuple()->isa<Var>() && ex->index()->isa<Lit>()) return s.fmt("{}", ex->unique_name());
        return s.fmt("{}#{}", ex->tuple(), ex->index());
    } else if (auto var = isa<Var>()) {
        return s.fmt("@{}", var->nom());
    } else if (auto pi = isa<Pi>()) {
        if (pi->is_cn()) {
            return s.fmt("Cn {}", pi->dom());
        } else {
            return s.fmt("{} -> {}", pi->dom(), pi->codom());
        }
    } else if (auto lam = isa<Lam>()) {
        return s.fmt("λ@({}) {}", lam->filter(), lam->body());
    } else if (auto app = isa<App>()) {
        if (auto size = isa_lit(isa_sized_type(app))) {
            if (auto real = thorin::isa<Tag::Real>(app)) return s.fmt("r{}", *size);
            if (auto _int = thorin::isa<Tag:: Int>(app)) {
                if (auto width = mod2width(*size)) return s.fmt("i{}", *width);

                // append utf-8 subscripts in reverse order
                std::string str;
                for (size_t mod = *size; mod > 0; mod /= 10)
                    ((str += char(char(0x80) + char(mod % 10))) += char(0x82)) += char(0xe2);
                std::reverse(str.begin(), str.end());

                return s.fmt("i{}", str);
            }
            THORIN_UNREACHABLE;
        }
        return s.fmt("{} {}", LPrec(app->callee(), app), RPrec(app, app->arg()));
    } else if (auto sigma = isa<Sigma>()) {
        return s.fmt("[{, }]", sigma->ops());
    } else if (auto tuple = isa<Tuple>()) {
        s.fmt("({, })", tuple->ops());
        return tuple->type()->isa_nom() ? s.fmt("∷{}", tuple->type()) : s;
    } else if (auto arr = isa<Arr>()) {
        return s.fmt("«{}; {}»", arr->shape(), arr->body());
    } else if (auto pack = isa<Pack>()) {
        return s.fmt("‹{}; {}›", pack->shape(), pack->body());
    } else if (auto proxy = isa<Proxy>()) {
        return s.fmt(".proxy#{}#{} {, }", proxy->id(), proxy->flags(), proxy->ops());
    } else if (auto bound = isa_bound(this)) {
        const char* op = bound->isa<Join>() ? "∪" : "∩";
        if (isa_nom()) s.fmt("{}{}: {}", op, unique_name(), type());
        return s.fmt("{}({, })", op, ops());
    }

    // other
    if (fields() == 0)
        return s.fmt(".{} {, }", node_name(), ops());
    return s.fmt(".{}#{} {, }", node_name(), fields(), ops());
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
    unique_queue<NomSet> noms;
    DefSet defs;
};

void RecStreamer::run(const Def* def) {
    if (!defs.emplace(def).second) return;

    for (auto op : def->ops()) { // for now, don't include debug info and type
        if (auto nom = op->isa_nom()) {
            if (max != 0) {
                if (noms.push(nom)) --max;
            }
        } else {
            run(op);
        }
    }

    if (auto nom = def->isa_nom())
        nom->unwrap(s.endl());
    else if (!def->unwrap())
        def->let(s.endl());
}

void RecStreamer::run() {
    while (!noms.empty()) {
        auto nom = noms.pop();
        s.endl().endl();

        if (nom->is_set()) {
            s.fmt("{}: {}", nom->unique_name(), nom->type());
            if (nom->has_var()) {
                auto e = nom->num_vars();
                s.fmt(": {}", e == 1 ? "" : "(");
                s.range(nom->vars(), [&](const Def* def) { s << def->unique_name(); });
                if (e != 1) s.fmt(")");
            }
            s.fmt(" = {{\t");
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

Stream& operator<<(Stream& s, std::pair<const Def*, const Def*> p) { return s.fmt("({}, {})", p.first, p.second); }

Stream& Def::stream(Stream& s) const {
    if (unwrap()) return unwrap(s);
    return s << unique_name();
}

Stream& Def::stream(Stream& s, size_t max) const {
    if (max == 0) return let(s);
    RecStreamer rec(s, --max);

    if (auto nom = isa_nom()) {
        rec.noms.push(nom);
        rec.run();
    } else {
        rec.run(this);
        if (max != 0) rec.run();
    }

    return s;
}

Stream& Def::let(Stream& s) const {
    return unwrap(s.fmt("{}: {} = ", unique_name(), type())).fmt(";");
}

void Def::dump() const { Streamable<Def>::dump(); }

void Def::dump(size_t max) const {
    Stream s(std::cout);
    stream(s, max).endl();
}

// TODO polish this
Stream& World::stream(Stream& s) const {
    //auto old_gid = curr_gid();
#if 1
    DepTree dep(*this);

    RecStreamer rec(s, 0);
    s << "module '" << name();

    stream(rec, dep.root()).endl();
    //assert_unused(old_gid == curr_gid());
    return s;
#else
    RecStreamer rec(s, std::numeric_limits<size_t>::max());
    s << "module '" << name();

    for (const auto& [name, nom] : externals()) {
        rec.noms.push(nom);
        rec.run();
    }

    return s.endl();
#endif
}

Stream& World::stream(RecStreamer& rec, const DepNode* n) const {
    rec.s.indent();

    if (auto nom = n->nom()) {
        rec.noms.push(nom);
        rec.run();
    }

    for (auto child : n->children())
        stream(rec, child);

    return rec.s.dedent();
}

void World::debug_stream() {
    if (min_level() == LogLevel::Debug) stream(stream());
}

template void Streamable<Def>::dump() const;
template void detail::HashTable<const Def*, void, GIDHash<const Def*>, 4>::dump() const;

}
