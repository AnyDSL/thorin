
#include <functional>

#include "thorin/pass/fp/closure_destruct.h"
#include "thorin/transform/untype_closures.h"

namespace thorin {

template<class I>
static inline undo_t min_undo(I& cont, std::function<undo_t (typename I::value_type)> f) {
    auto undo_t = No_Undo;
    for (auto elem: cont)
        std::min(undo_t, f(elem));
    return No_Undo;
}

static const Var* isa_var(const Def* def) {
    if (auto proj = def->isa<Extract>())
        def = proj->tuple();
    return def->isa<Var>();
}

static bool is_exp_imp(Lam *lam) {
    return lam->is_external() || !lam->is_set();
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
        if (auto lam = var->nom()->isa_nom<Lam>(); lam && is_exp_imp(lam)) {
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
    return min_undo(points_to_, [&] (auto edge) {
        return edge->mark_esc();
    });
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
    a->points_to_.insert({b, 0});
    b->repr_ = a;
    b->points_to_.clear();
    return res;
}

ClosureDestruct::Node ClosureDestruct::Node::top_(nullptr);

void ClosureDestruct::Node::dump(Stream& s, std::set<Node*>& visited) {
    if (!def_) {
        s.fmt("<top>");
    } else {
        s.fmt("[{}, {}, {}] \n", repr(), def_, (is_esc()) ? "⊤" : "⊥");
        if (visited.count(this) == 0) {
            visited.insert(this);
            s.indent(1);
            s.fmt("\n");
            for (auto& edge: points_to_) {
                s.fmt("[{}] {} ", edge.iter(), (edge->repr() == this) ? "=" : "->");
                edge->dump(s, visited);
            }
            s.fmt("\n");
            s.dedent(1);
        }
    }
}

ClosureDestruct::Node* ClosureDestruct::get_node(const Def* def, undo_t undo) { 
    auto [p, inserted] = def2node_.emplace(def, nullptr);
    if (inserted)
        p->second = std::make_unique<Node>(def, false, undo);
    return p->second.get();
}

static bool interesting_type_b(const Def* type) {
    return UntypeClosures::isa_pct(type) != nullptr;
}

static bool interesting_type(const Def* type) {
    if (interesting_type_b(type))
        return true;
    if (auto sigma = type->isa<Sigma>())
        return std::any_of(sigma->ops().begin(), sigma->ops().end(), interesting_type);
    if (auto arr = type->isa<Arr>())
        return interesting_type(arr->body());
    return false;
}

undo_t ClosureDestruct::add_pointee(ClosureDestruct::Node* node, const Def* def) {
    if (def->isa_nom<Lam>() || def->isa<Var>()) {
        return node->add_pointee(get_node(def), iter_);
    } else if (auto proj = def->isa<Extract>()) {
        if (auto lam = proj->tuple()->isa_nom<Lam>(); lam && interesting_type(def->type()))
            return node->add_pointee(node, iter_);
        else
            return add_pointee(node, proj->tuple());
    } else if (auto closure = UntypeClosures::isa_closure(def)) {
        return add_pointee(node, closure->op(1_u64));
    } else if (auto pack = def->isa<Pack>()) {
        return add_pointee(node, pack->body());
    } else if (auto tuple = def->isa<Tuple>()) {
        auto undo = No_Undo;
        for (auto op: tuple->ops())
            undo = std::min(undo, add_pointee(node, op));
        return undo;
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
                dropped->set(lam->apply(world().tuple(new_vars)));
                auto dropped_node = get_node(dropped, curr_undo());
                lam_node->unify(dropped_node);
            }
            return world().tuple(closure->type(), {world().tuple(), dropped}, closure->dbg());
        }
    }
    return def;
}


// TODO: Interprocedural part?
undo_t ClosureDestruct::analyze_call(const Def* callee, ArgVec& args) {
    if (auto lam = callee->isa_nom<Lam>()) {
        return min_undo(args, [&] (auto arg) { 
            world().DLOG("call: {} -> {}", lam, arg.second);
            return add_pointee(lam->var(arg.first), arg.second);
        });
    } else if (auto closure = UntypeClosures::isa_closure(callee)) {
        return analyze_call(closure->op(1_u64), args);
    } else if (auto proj = callee->isa<Extract>()) {
        return analyze_call(proj->tuple(), args);
    } else if (auto pack = callee->isa<Pack>()) {
        return analyze_call(pack->body(), args);
    } else if (auto tuple = callee->isa<Tuple>()) {
        auto undo = No_Undo;
        for (auto op: tuple->ops())
            undo = std::min(undo, analyze_call(op, args));
        return undo;
    } else {
        return min_undo(args, [&] (auto arg) {
            world().DLOG("call {}: escape {}", callee, arg.second);
            return add_pointee(Node::top(), arg.second);
        });
    }
}

// TODO: Analyze store
undo_t ClosureDestruct::analyze(const Def* def) {
    auto undo = No_Undo;
    if (auto closure = UntypeClosures::isa_closure(def)) {
        undo = add_pointee(closure->op(1), closure->op(0));
        // FIXME: env also flows into the env-argument: #x @{f, 0} for 0 <= x <= size env
    } else if (auto app = def->isa<App>(); app && app->callee_type()->is_cn()) {
        auto args = ArgVec();
        for (size_t i = 0; i < app->num_args(); i++) 
            if (interesting_type(app->callee_type()->dom(i)))
                args.emplace_back(i, app->arg(i));
        return analyze_call(app->callee(), args);
    }
    return undo;
}

void ClosureDestruct::unify(const Def* a, const Def* b) {
    get_node(a)->unify(get_node(b));
}

void ClosureDestruct::dump_node(Node* node) {
    world().stream().fmt("{}\n", *node);
}

void ClosureDestruct::dump_graph() {
    auto s = world().stream();
    auto v = std::set<Node*>();
    s.fmt("-----------------\n");
    for (auto& [def, node]: def2node_) {
        if (v.count(node.get()) == 0) {
            s.fmt("{} =>\n", def);
            node->dump(s, v);
            s.fmt("\n");
        }
    }
    s.fmt("\n");
}

} // namespace thorin
