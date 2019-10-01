#include "thorin/def.h"
#include "thorin/util.h"

namespace thorin {

Stream& Stream::endl() {
    os_ << '\n';
    for (size_t i = 0; i != level_; ++i) os_ << tab_;
    return *this;
}

//------------------------------------------------------------------------------

Stream& operator<<(Stream& s, const Def* def) {
    if (def == nullptr) return s << "<nullptr>";
    if (is_const(def)) return stream(s, def, Recurse::No);
    return s << def->unique_name();
}

Stream& stream(Stream& s, const Def* def, Recurse recurse) {
    if (recurse == Recurse::No && def->isa_nominal()) return s << def->unique_name();

    if (false) {}
    else if (def->isa<Universe>())  return s.fmt("□");
    else if (def->isa<KindStar>())  return s.fmt("*");
    else if (def->isa<KindMulti>()) return s.fmt("*M");
    else if (def->isa<KindArity>()) return s.fmt("*A");
    else if (def->isa<Mem>())       return s.fmt("mem");
    else if (def->isa<Nat>())       return s.fmt("nat");
    else if (auto bot = def->isa<Bot>()) return s.fmt("{{⊥: {}}}", bot->type());
    else if (auto top = def->isa<Top>()) return s.fmt("{{⊤: {}}}", top->type());
    else if (auto axiom = def->isa<Axiom>()) return s.fmt("{}", axiom->name());
    else if (auto lit = def->isa<Lit>()) {
        if (lit->type()->isa<KindArity>()) return s.fmt("{}ₐ", lit->get());

        if (lit->type()->type()->isa<KindArity>()) {
            if (lit->type()->isa<Top>()) return s.fmt("{}T", lit->get());

            // append utf-8 subscripts in reverse order
            std::string str;
            for (size_t aa = as_lit<nat_t>(lit->type()); aa > 0; aa /= 10)
                ((str += char(char(0x80) + char(aa % 10))) += char(0x82)) += char(0xe2);
            std::reverse(str.begin(), str.end());

            return s.fmt("{}{}", lit->get(), str);
        } else if (lit->type()->isa<Nat>()) {
            return s.fmt("{}_nat", lit->get());
        } else if (auto int_ = thorin::isa<Tag::Int >(lit->type())) {
            return s.fmt("{}_i{}", lit->get(), as_lit<nat_t>(int_->arg()));
        } else if (auto real = thorin::isa<Tag::Real>(lit->type())) {
            switch (as_lit<nat_t>(real->arg())) {
                case 16: return s.fmt("{}_r16", lit->get<r16>());
                case 32: return s.fmt("{}_r32", lit->get<r32>());
                case 64: return s.fmt("{}_r64", lit->get<r64>());
                default: THORIN_UNREACHABLE;
            }
        }

        return s.fmt("{{{}: {}}}", lit->type(), lit->get());
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
            if (auto real = thorin::isa<Tag::Real>(app)) return s.fmt("r{}", *w);
            return s.fmt("i{}", *w);
        } else if (auto ptr = thorin::isa<Tag::Ptr>(app)) {
            auto [pointee, addr_space] = ptr->args<2>();
            s.fmt("{}*", pointee);
            if (auto as = isa_lit<nat_t>(addr_space)) {
                switch (*as) {
                    case AddrSpace::Generic:  return s.fmt("");
                    case AddrSpace::Global:   return s.fmt("[Global]");
                    case AddrSpace::Texture:  return s.fmt("[Tex]");
                    case AddrSpace::Shared:   return s.fmt("[Shared]");
                    case AddrSpace::Constant: return s.fmt("[Constant]");
                    default:;
                }
            }
            return s.fmt("[{}]", addr_space);
        }

        if (app->arg()->isa<Tuple>() || app->arg()->isa<Pack>())
            return s.fmt("{}{}", app->callee(), app->arg());
        return s.fmt("{}({})", app->callee(), app->arg());
    } else if (auto sigma = def->isa<Sigma>()) {
        if (sigma->isa_nominal()) s.fmt("{}: {}", sigma->unique_name(), sigma->type());
        return s.list(sigma->ops(), [&](const Def* def) { return s << def; }, "[", "]");
    } else if (auto tuple = def->isa<Tuple>()) {
#if 0
        // special case for string
        if (std::all_of(ops().begin(), ops().end(), [&](const Def* op) { return op->isa<Lit>(); })) {
            if (auto variadic = type()->isa<Variadic>()) {
                if (auto i = variadic->body()->isa<Sint>()) {
                    if (i->lit_num_bits() == 8) {
                        for (auto op : ops()) os << as_lit<char>(op);
                        return os;
                    }
                }
            }
        }
#endif
        s.list(tuple->ops(), [&](const Def* def) { return s << def; }, "(", ")");
        return tuple->type()->isa_nominal() ? s.fmt(": {}", tuple->type()) : s;
    } else if (auto variadic = def->isa<Variadic>()) {
        if (auto nom_variadic = variadic->isa_nominal<Variadic>())
            return s.fmt("«{}: {}; {}»", nom_variadic->param(), nom_variadic->domain(), nom_variadic->codomain());
        return s.fmt("«{}; {}»", variadic->domain(), variadic->codomain());
    } else if (auto pack = def->isa<Pack>()) {
#if 0
        // special case for string
        if (auto variadic = type()->isa<Variadic>()) {
            if (auto i = variadic->body()->isa<Sint>()) {
                if (i->lit_num_bits() == 8) {
                    if (auto a = isa_lit<u64>(arity())) {
                        if (auto lit = body()->isa<Lit>()) {
                            for (size_t i = 0, e = *a; i != e; ++i) os << lit->get<char>();
                            return os;
                        }
                    }
                }
            }
        }
#endif
        if (auto nom_pack = pack->isa_nominal<Pack>())
            return s.fmt("‹{}: {}; {}›", nom_pack->param(), nom_pack->domain(), nom_pack->body());
        return s.fmt("‹{}; {}›", pack->domain(), pack->body());
    } else if (auto union_ = def->isa<Union>()) {
        if (union_->isa_nominal()) s.fmt("{}: {}", union_->unique_name(), union_->type());
        return s.fmt("⋃").list(union_->ops(), [&](const Def* def) { return s << def; }, "(", ")");
    }

    // unknown node type
    if (def->fields() != 0)
        s.fmt("({} {})", def->node_name(), def->fields());
    else
        s.fmt("{}", def->node_name());
    return s.list(def->ops(), [&](const Def* def) { return s << def; }, "(", ")");
}

Stream& stream_assignment(Stream& s, const Def* def) {
    return stream(s.fmt("{}: {} = ", def->unique_name(), def->type()), def, Recurse::OneLevel);
}

}
