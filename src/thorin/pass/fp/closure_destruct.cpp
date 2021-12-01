
#include <functional>

#include "thorin/pass/fp/closure_destruct.h"
#include "thorin/transform/untype_closures.h"

namespace thorin {

static const Var* isa_var(const Def* def) {
    if (auto proj = def->isa<Extract>())
        def = proj->tuple();
    return def->isa<Var>();
}

ClosureDestruct::Node::Node(const Def* def, bool esc, undo_t undo)
    : repr_(this)
    , def_(def)
    , esc_(esc)
    , undo_(undo)
    , points_to_() {
    if (!def || def->is_external()) { // ⊤, external lam or Global
        esc_ = true;
    } else if (def->isa<Global>()) {
        esc_ = true;
        add_pointee(top(), 0);
    } else if (auto var = isa_var(def)) {
        if (auto lam = var->nom()->isa_nom<Lam>(); lam && lam->is_external()) {
            esc_ |= !lam->is_set(); // imported lams always escape their args
            add_pointee(top(), 0);
        }
    }
}

undo_t ClosureDestruct::Node::mark_esc() {
    if (!is_repr())
        return repr()->mark_esc();
    if (is_esc())
        return No_Undo;
    esc_ = true;
    auto undo = undo_;
    for (auto p : points_to_)
        undo = std::min(undo, p->mark_esc());
    return undo;
}

undo_t ClosureDestruct::Node::add_pointee(Node* pointee, size_t iter) {
    if (!is_repr())
        return repr()->add_pointee(pointee, iter);
    if (def_) {
        auto [p, inserted] = points_to_.emplace(pointee, iter);
        if (!inserted) {
            p->set_iter(iter);
            return No_Undo;
        }
    }
    return (is_esc()) ? pointee->mark_esc() : No_Undo;
}

undo_t ClosureDestruct::Node::unify(Node* other) {
    auto a = repr();
    auto b = other->repr();
    if (a == b)
        return No_Undo;
    auto res = No_Undo;
    if (!a->is_esc())
        std::swap(a, b);
    if (a->esc_ != b->esc_)
        res = b->mark_esc();
    a->esc_ &= b->is_esc();
    a->undo_ = std::min(a->undo_, b->undo_);
    a->points_to_.insert(b->points_to_.begin(), b->points_to_.end());
    b->points_to_.clear();
    return res;
}

ClosureDestruct::Node ClosureDestruct::Node::top_(nullptr);

void ClosureDestruct::Node::dump(Stream& s, std::set<Node*>& visited) {
    if (!def_) {
        s.fmt("⊤");
    } if (!is_repr()) {
        repr_->dump(s, visited);
    } else {
        s.fmt("{}: [{}, {}]\n", this, def_, (is_esc()) ? "⊤" : "⊥");
        if (visited.count(this) != 0) {
            visited.insert(this);
            s.indent();
            for (auto& edge: points_to_)
               edge.dump(s, visited);
            s.dedent();
        }
    }
}

void ClosureDestruct::Edge::dump(Stream& s, std::set<Node*>& visited) const {
    s.fmt("{}: ", iter_);
    node_->dump(s, visited);
}


ClosureDestruct::Node* ClosureDestruct::get_node(const Def* def, undo_t undo) { 
    auto [p, inserted] = def2node_.emplace(def, nullptr);
    if (inserted)
        p->second = std::make_unique<Node>(def, false, undo);
    return p->second.get();
}

undo_t ClosureDestruct::add_pointee(ClosureDestruct::Node* node, const Def* def) {
    if (def->isa_nom<Lam>() || isa_var(def)) {
        return node->add_pointee(get_node(def), iter_);
    } else if (auto closure = UntypeClosures::isa_closure(def)) {
        return add_pointee(node, closure->op(1_u64));
    } else if (auto proj = def->isa<Extract>()) {
        return add_pointee(node, proj->tuple());
    } else if (auto pack = def->isa<Pack>()) {
        return add_pointee(node, pack->body());
    } else if (auto tuple = def->isa<Tuple>()) {
        auto undo = No_Undo;
        for (auto op: tuple->ops())
            undo = std::min(undo, add_pointee(node, def));
    } else {
        return No_Undo;
    }
}

const Def* ClosureDestruct::rewrite(const Def* def) {
    if (auto closure = UntypeClosures::isa_closure(def)) {
        auto env = closure->op(0);
        auto lam = closure->op(1)->isa_nom<Lam>();
        auto lam_node = get_node(lam);
        if (!lam_node->is_esc() && lam->dom(0) != world().sigma()) { // TODO: Check Basic-Block-Like
            auto& [old_env, dropped] = clos2dropped_[lam];
            if (!dropped || old_env != env) {
                auto doms = world().sigma(Array<const Def*>(lam->num_doms(), [&](auto i) {
                    return (i == 0) ? world().sigma() : lam->dom(i);
                }));
                dropped = lam->stub(world(), world().cn(doms), lam->dbg());
                world().DLOG("drop ({}, {}) => {}", env, lam, dropped);
                auto new_vars = Array<const Def*>(dropped->num_doms(), [&](auto i) {
                    return (i == 0) ? env : dropped->var(i); 
                });
                lam_node->unify(get_node(def, curr_undo()));
            }
            return world().tuple(closure->type(), {world().tuple(), dropped}, closure->dbg());
        }
    }
    return def;
}

undo_t ClosureDestruct::analyze_call(const Def* callee, size_t i, const Def* arg) {
    if (auto lam = callee->isa_nom<Lam>()) {
        return add_pointee(lam->var(i), arg);
    } else if (auto proj = callee->isa<Extract>()) {
        return analyze_call(proj->tuple(), i, arg);
    } else if (auto pack = callee->isa<Pack>()) {
        return analyze_call(pack->body(), i, arg);
    } else if (auto tuple = callee->isa<Tuple>()) {
        auto undo = No_Undo;
        for (auto op: tuple->ops())
            undo = std::min(undo, analyze_call(callee, i, arg));
        return undo;
    } else {
        return add_pointee(Node::top(), arg);
    }
}

undo_t ClosureDestruct::analyze(const Def* def) {
    if (auto closure = UntypeClosures::isa_closure(def)) {
        return add_pointee(closure->op(1), closure->op(0));
    } else if (auto app = def->isa<App>(); app && app->callee_type()->is_cn()) {
        auto undo = No_Undo;
        for (size_t i = 0; i < app->num_args(); i++)
            undo = std::min(undo, analyze_call(app->callee(), i, app->arg(i)));
        return undo;
    }
    return No_Undo;
}

void ClosureDestruct::unify(const Def* a, const Def* b) {
    get_node(a)->unify(get_node(b));
}

void ClosureDestruct::dump_node(Node* node) {
    world().stream().fmt("{}", *node);
}

void ClosureDestruct::dump_graph() {
    auto s = world().stream();
    auto v = std::set<Node*>();
    for (auto& [def, node]: def2node_) {
        s.fmt("{def} => ");
        node->dump(s, v);
        s.fmt("\n");
    }
}

} // namespace thorin
