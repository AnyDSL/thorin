#include "thorin/def.h"
#include "thorin/util.h"
#include "thorin/world.h"

namespace thorin {

Stream& Def::stream(Stream& s) const {
    return thorin::stream(s, this, Recurse::OneLevel);
}

Stream& operator<<(Stream& s, const Def* def) {
    if (def == nullptr)     return s << "<nullptr>";
    if (def->isa<Axiom>())  return s << def->name();
    if (is_const(def))      return stream(s, def, Recurse::No);
    return s << def->unique_name();
}

Stream& stream(Stream& s, const Def* def, Recurse recurse) {
    if (recurse == Recurse::No && def->isa_nominal()) return s << def->unique_name();

    if (false) {}
    else if (def->isa<Universe>())  return s.fmt("□");
    else if (def->isa<KindStar>())  return s.fmt("★");
    else if (def->isa<KindMulti>()) return s.fmt("•");
    else if (def->isa<KindArity>()) return s.fmt("◦");
    else if (def->isa<Mem>())       return s.fmt("mem");
    else if (def->isa<Nat>())       return s.fmt("nat");
    else if (auto bot = def->isa<Bot>()) return s.fmt("⊥∷{}", bot->type());
    else if (auto top = def->isa<Top>()) return s.fmt("⊤∷{}", top->type());
    else if (auto axiom = def->isa<Axiom>()) return s.fmt("{}", axiom->name());
    else if (auto lit = def->isa<Lit>()) {
        if (auto real = thorin::isa<Tag::Real>(lit->type())) {
            switch (as_lit<nat_t>(real->arg())) {
                case 16: return s.fmt("{}∷r16", lit->get<r16>());
                case 32: return s.fmt("{}∷r32", lit->get<r32>());
                case 64: return s.fmt("{}∷r64", lit->get<r64>());
                default: THORIN_UNREACHABLE;
            }
        } else if (lit->type()->type()->isa<KindArity>()) {
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
            if (auto nom_pi = pi->isa_nominal<Pi>())
                return s.fmt("cn {}:{}", nom_pi->param(), nom_pi->domain());
            else
                return s.fmt("cn {}", pi->domain());
        } else {
            if (auto nom_pi = pi->isa_nominal<Pi>())
                return s.fmt("Π{}:{} -> {}", nom_pi->param(), nom_pi->domain(), nom_pi->codomain());
            else
                return s.fmt("Π{} -> {}", pi->domain(), pi->codomain());
        }
    } else if (def->isa<Lam>()) {
        // TODO
    } else if (auto app = def->isa<App>()) {
        if (auto w = get_width(app)) {
            if (auto _int = thorin::isa<Tag:: Int>(app)) return s.fmt("i{}", *w);
            if (auto sint = thorin::isa<Tag::SInt>(app)) return s.fmt("s{}", *w);
            if (auto real = thorin::isa<Tag::Real>(app)) return s.fmt("r{}", *w);
            THORIN_UNREACHABLE;
        } else if (auto ptr = thorin::isa<Tag::Ptr>(app)) {
            auto [pointee, addr_space] = ptr->args<2>();
            if (auto as = isa_lit<nat_t>(addr_space); as && *as == 0) return s.fmt("{}*", pointee);
        }

        return s.fmt("{} {}", app->callee(), app->arg());
    } else if (auto sigma = def->isa<Sigma>()) {
        if (sigma->isa_nominal()) s.fmt("{}: {}", sigma->unique_name(), sigma->type());
        return s.fmt("[{, }]", sigma->ops());
    } else if (auto tuple = def->isa<Tuple>()) {
        if (tuple == def->world().table_and()) return s.fmt("AND");
        if (tuple == def->world().table_or ()) return s.fmt( "OR");
        if (tuple == def->world().table_xor()) return s.fmt("XOR");
        if (tuple == def->world().table_not()) return s.fmt("NOT");
        s.fmt("({, })", tuple->ops());
        return tuple->type()->isa_nominal() ? s.fmt("∷{}", tuple->type()) : s;
    } else if (auto variadic = def->isa<Variadic>()) {
        if (auto nom_variadic = variadic->isa_nominal<Variadic>())
            return s.fmt("«{}: {}; {}»", nom_variadic->param(), nom_variadic->domain(), nom_variadic->codomain());
        return s.fmt("«{}; {}»", variadic->domain(), variadic->codomain());
    } else if (auto pack = def->isa<Pack>()) {
        if (auto nom_pack = pack->isa_nominal<Pack>())
            return s.fmt("‹{}: {}; {}›", nom_pack->param(), nom_pack->domain(), nom_pack->body());
        return s.fmt("‹{}; {}›", pack->domain(), pack->body());
    } else if (auto union_ = def->isa<Union>()) {
        if (union_->isa_nominal()) s.fmt("{}: {}", union_->unique_name(), union_->type());
        return s.fmt("⋃({, }]", union_->ops());
    }

    // unknown node type
    if (def->fields() == 0)
        return s.fmt("{} {, }", def->node_name(), def->ops());
    return s.fmt("{}#{} {, }", def->node_name(), def->fields(), def->ops());
}

Stream& stream_assignment(Stream& s, const Def* def) {
    return stream(s.fmt("{}: {} = ", def->unique_name(), def->type()), def, Recurse::OneLevel);
}

}
