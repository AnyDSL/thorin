#include "thorin/world.h"
#include "thorin/util/stream.h"
#include "thorin/analyses/scope.h"

namespace thorin {

struct ScopedWorld : public Streamable<ScopedWorld> {
    ScopedWorld(World& w) : world_(w), forest_(w) {
    }

    World& world_;
    mutable ScopesForest forest_;

    mutable DefSet done_;
    mutable ContinuationMap<std::unique_ptr<std::vector<const Def*>>> scopes_to_defs_;
    mutable std::vector<const Def*> top_lvl_;

    Stream& stream(Stream&) const;
private:

    void stream_cont(thorin::Stream& s, Continuation* cont) const;
    void prepare_def(Continuation* in, const Def* def) const;
    static void stream_defs(thorin::Stream& s, std::vector<const Def*>& defs);
};

void ScopedWorld::stream_cont(thorin::Stream& s, Continuation* cont) const {
    if (cont->is_external())
        s.fmt("extern ");
    if (cont->is_intrinsic())
        s.fmt("intrinsic ");

    s.fmt("{}(", cont->unique_name());
    const FnType* t = cont->type();
    int ret_pi = t->ret_param_index();
    for (size_t i = 0; i < cont->num_params(); i++) {
        s.fmt("{}: {}", cont->param(i)->unique_name(), t->types()[i]);
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

void ScopedWorld::stream_defs(thorin::Stream& s, std::vector<const Def*>& defs) {
    size_t i = 0;
    for (auto def : defs) {
        s.fmt("{}: {} = {}({, })", def->unique_name(), def->type(), def->op_name(), def->ops());
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