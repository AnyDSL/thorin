#include "scoped_dump.h"

namespace thorin {

#define COLORS(C) \
C(Black, "\u001b[30m") \
C(Red, "\u001b[31m") \
C(Green, "\u001b[32m") \
C(Yellow, "\u001b[33m") \
C(Blue, "\u001b[34m") \
C(Magenta, "\u001b[35m") \
C(Cyan, "\u001b[36m") \
C(White, "\u001b[37m") \
C(Reset, "\u001b[0m")  \

// #define N(n, _) COLOR_INDEX_##n,
// enum Colors {
//     COLORS(N)
// };

#define T(n, c) static const char* n = c;
COLORS(T)

void ScopedWorld::stream_cont(thorin::Stream& s, Continuation* cont) const {
    s.fmt(Magenta);
    if (cont->is_external())
        s.fmt("extern ");
    if (cont->is_intrinsic())
        s.fmt("intrinsic ");

    s.fmt(Red);
    s.fmt("{}", cont->unique_name());
    s.fmt(Reset);
    s.fmt("(");
    const FnType* t = cont->type();
    int ret_pi = t->ret_param_index();
    for (size_t i = 0; i < cont->num_params(); i++) {
        s.fmt(Yellow);
        s.fmt("{}: ", cont->param(i)->unique_name());
        s.fmt(Blue);
        s.fmt("{}", t->types()[i]);
        s.fmt(Reset);
        if (i + 1 < cont->num_params())
            s.fmt(", ");
    }
    s.fmt(")");
    if (!cont->has_body()) {
        s.fmt(";");
        return;
    }

    s.fmt(" = {{\t\n");

    for (auto p : cont->params())
        done_.insert(p);

    Scope& sc = forest_.get_scope(cont);
    scopes_to_defs_[cont] = std::make_unique<std::vector<const Def*>>();
    auto children = sc.children_scopes();
    size_t i = 0;
    for (auto child : children) {
        stream_cont(s, child);
        //if (i + 1 < children.size())
            s.fmt("\n\n");
        i++;
    }

    prepare_def(cont, cont->body());

    auto defs = *scopes_to_defs_[cont];
    stream_defs(s, defs);

    s.fmt("\b\n}}");
}

void ScopedWorld::prepare_def(Continuation* in, const thorin::Def* def) const {
    if (done_.contains(def))
        return;
    done_.insert(def);
    if (def->isa_nom<Continuation>())
        return;

    while (in) {
        Scope* scope = &forest_.get_scope(in);
        if (scope->contains(def))
            break;
        in = scope->parent_scope();
    }

    for (auto op : def->ops())
        prepare_def(in, op);

    if (!in)
        top_lvl_.push_back(def);
    else
        scopes_to_defs_[in]->push_back(def);
}

void ScopedWorld::stream_op(thorin::Stream& s, const thorin::Def* op) {
    //if (is_mem(op))
    //    s.fmt(Gray);
    if (op->isa<PrimLit>())
        s.fmt(Cyan);
    if (op->isa<Param>())
        s.fmt(Yellow);
    if (op->isa<Continuation>())
        s.fmt(Red);
    s.fmt("{}", op);
    s.fmt(Reset);
}

void ScopedWorld::stream_ops(thorin::Stream& s, Defs ops) {
    s.fmt("(");
    size_t j = 0;
    for (auto op : ops) {
        stream_op(s, op);
        if (j + 1 < ops.size())
            s.fmt(", ");
        j++;
    }
    s.fmt(")");
}

void ScopedWorld::stream_def(thorin::Stream& s, const thorin::Def* def) {
    if (auto app = def->isa<App>()) {
        stream_op(s, app->callee());
        stream_ops(s, app->args());
        if (!app->filter()->is_empty()) {
            s.fmt(" with {}", app->filter());
        }
        return;
    }

    s.fmt(Green);
    s.fmt("{}", def->op_name());
    s.fmt(Reset);
    stream_ops(s, def->ops());
}

void ScopedWorld::stream_defs(thorin::Stream& s, std::vector<const Def*>& defs) {
    size_t i = 0;
    for (auto def : defs) {
        s.fmt("{}: ", def->unique_name());
        s.fmt(Blue);
        s.fmt("{}", def->type());
        s.fmt(Reset);
        s.fmt(" = ");
        stream_def(s, def);
        if (i + 1 < defs.size())
            s.fmt("\n");
        i++;
    }
}

Stream& ScopedWorld::stream(thorin::Stream& s) const {
    auto tl = forest_.top_level_scopes();
    size_t i = 0;
    for (auto root : tl) {
        stream_cont(s, root);
        //if (i + 1 < tl.size())
            s.fmt("\n\n");
        i++;
    }

    stream_defs(s, top_lvl_);

    return s;
}

std::unique_ptr<ScopedWorld> scoped_world(World& w) {
    return std::make_unique<ScopedWorld>(w);
}

}
