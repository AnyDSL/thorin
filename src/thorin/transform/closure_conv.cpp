
#include "thorin/transform/closure_conv.h"
#include "thorin/analyses/scope.h"

namespace thorin {

static auto num_doms(const Def *def) {
    auto pi = def->type()->isa<Pi>();
    assert(pi && "cc: num_doms(): def not of pi type");
    return pi->num_doms();
}

void ClosureConv::run() {
    auto externals = std::vector(world().externals().begin(), world().externals().end());
    auto subst = Def2Def();
    world().DLOG("===== ClosureConv: start =====");
    for (auto [_, ext_def]: externals) {
        rewrite(ext_def, subst);
    }
    while (!worklist_.empty()) {
        auto def = worklist_.front();
        subst = Def2Def();
        worklist_.pop();
        if (auto closure = closures_.lookup(def)) {
            auto new_fn = closure->fn;
            auto env = closure->env;
            auto num_fvs = closure->num_fvs;
            auto old_fn = closure->old_fn;

            world().DLOG("RUN: closure body {} [old={}, env={}]\n\t", new_fn, old_fn, env);
            auto env_param = new_fn->var(0_u64);
            if (num_fvs == 1) {
                subst.emplace(env, env_param);
            } else {
                for (size_t i = 0; i < num_fvs; i++) {
                    subst.emplace(env->op(i), world().extract(env_param, i, world().dbg("cc_fv")));
                }
            }

            auto params = 
                world().tuple(Array<const Def*>(old_fn->num_doms(), [&] (auto i) {
                    return new_fn->var(i + 1); 
                }), world().dbg("cc_param"));
            subst.emplace(old_fn->var(), params);

            auto filter = (new_fn->filter()) 
                ? rewrite(new_fn->filter(), subst) 
                : nullptr; // extern function
            
            auto body = (new_fn->body())
                ? rewrite(new_fn->body(), subst)
                : nullptr;

            new_fn->set_body(body);
            new_fn->set_filter(filter);
        }
        else {
            world().DLOG("RUN: rewrite def {}\t", def);
            rewrite(def, subst);
        }
        world().DLOG("\b");
    }
    world().DLOG("===== ClosureConv: done ======");
    // world().debug_stream();
}

const Def* ClosureConv::rewrite(const Def* def, Def2Def& subst) {
    switch(def->node()) {
        case Node::Kind:
        case Node::Space:
        case Node::Nat:
        case Node::Bot:
        case Node::Top:
        case Node::Axiom:
            return def;
        default:
            break;
    }

    auto map = [&](const Def* new_def) {
        subst.emplace(def, new_def);
        return new_def;
    };

    if (subst.contains(def)) {
        return map(*subst.lookup(def));
    } else if (auto pi = def->isa<Pi>(); pi && pi->is_cn()) {
        /* Types: rewrite dom, codom \w susbt here */
        return map(closure_type(pi, subst));
    } else if (auto lam = def->isa_nom<Lam>(); lam && lam->type()->is_cn()) {
        auto [old, num, fv_env, fn] = make_closure(lam, subst);
        auto closure_type = rewrite(lam->type(), subst);
        auto env = rewrite(fv_env, subst);
        auto closure = world().tuple(closure_type, {env, fn});
        world().DLOG("RW: pack {} ~> {} : {}", lam, closure, closure_type);
        return map(closure);
    } 

    auto new_type = rewrite(def->type(), subst);
    auto new_dbg = (def->dbg()) ? rewrite(def->dbg(), subst) : nullptr;

    if (auto nom = def->isa_nom()) {
        // TODO: Test this
        world().DLOG("RW: nom {}", nom);
        auto new_nom = nom->stub(world(), new_type, new_dbg);
        subst.emplace(nom->var(), new_nom->var());
        for (size_t i = 0; i < nom->num_ops(); i++) {
            if (def->op(i))
                new_nom->set(i, rewrite(def->op(i), subst));
        }
        if (auto restruct = new_nom->restructure())
            return map(restruct);
        return map(new_nom);
    } else {
        auto new_ops = Array<const Def*>(def->num_ops(), [&](auto i) {
            return rewrite(def->op(i), subst);
        });
        if (auto app = def->isa<App>(); app && new_ops[0]->type()->isa<Sigma>()) {
            auto closure = new_ops[0];
            auto args = new_ops[1];
            auto env = world().extract(closure, 0_u64, world().dbg("cc_app_env"));
            auto fn = world().extract(closure, 1_u64, world().dbg("cc_app_f"));
            world().DLOG("RW: call {} ~> APP {} {} {}", closure, fn, env, args);
            return map(world().app(fn, Array<const Def*>(num_doms(fn), [&](auto i) {
                return (i == 0) ? env : world().extract(args, i - 1);
            })));
        } else {
            return map(def->rebuild(world(), new_type, new_ops, new_dbg));
        }
    }
}

const Def* ClosureConv::closure_type(const Pi* pi, Def2Def& subst, const Def* env_type) {
    if (!env_type) {
        if (auto pct = closure_types_.lookup(pi))
            return* pct;
        auto sigma = world().nom_sigma(world().kind(), 2_u64, world().dbg("cc_pct"));
        auto new_pi = closure_type(pi, subst, sigma->var());
        sigma->set(0, sigma->var());
        sigma->set(1, new_pi);
        closure_types_.emplace(pi, sigma);
        world().DLOG("C-TYPE: pct {} ~~> {}", pi, sigma);
        return sigma;
    } else {
        auto dom = world().sigma(Array<const Def*>(pi->num_doms() + 1, [&](auto i) {
            return (i == 0) ? env_type : rewrite(pi->dom(i - 1), subst);
        }));
        auto new_pi = world().cn(dom, world().dbg("cc_ct"));
        world().DLOG("C-TYPE: ct {}, env = {} ~~> {}", pi, env_type, new_pi);
        return new_pi;
    }
}


void FVA::split_fv(const Def* def, DefSet& out) {
    if (def->no_dep() || def->isa<Lam>() || def->is_external() || def->isa<Axiom>()) {
        return;
    } else if (auto tuple = def->isa<Tuple>()) {
        for (auto op: tuple->ops())
            split_fv(op, out);
    } else {
        out.emplace(def);
    }
}

std::pair<FVA::Node*, bool> FVA::build_node(Lam *lam, NodeQueue& worklist) {
    auto [p, inserted] = lam2nodes_.emplace(lam, nullptr);
    if (!inserted) 
        return {p->second.get(), false};
    world().DLOG("FVA: create node: {}", lam);
    p->second = std::make_unique<Node>();
    auto node = p->second.get();
    node->lam = lam;
    node->pass_id = 0;
    auto scope = Scope(lam);
    node->fvs = DefSet();
    for (auto v: scope.free_defs()) {
        split_fv(v, node->fvs);
    }
    node->preds = Nodes();
    node->succs = Nodes();
    bool init_node = false;
    for (auto n: scope.free_noms()) {
        if (auto pred = n->isa_nom<Lam>(); pred && pred != lam) {
            auto [pnode, inserted] = build_node(pred, worklist);
            node->preds.push_back(pnode);
            pnode->succs.push_back(node);
            init_node |= inserted;
        }
    }
    if (!init_node) {
        worklist.push(node);
        world().DLOG("FVA: init {}", lam);
    }
    return {node, true};
}


void FVA::run(NodeQueue& worklist) {
    int iter = 0;
    while(!worklist.empty()) {
        auto node = worklist.front();
        worklist.pop();
        world().DLOG("FA: iter {}: {}", iter, node->lam);
        if (is_done(node))
            continue;
        auto changed = is_bot(node);
        mark(node);
        for (auto p: node->preds) {
            auto& pfvs = p->fvs;
            changed |= node->fvs.insert(pfvs.begin(), pfvs.end());
            world().DLOG("\tFV({}) ∪= FV({}) = {{{, }}}\b", node->lam, p->lam, pfvs);
        }
        if (changed) {
            for (auto s: node->succs) {
                worklist.push(s);
            }
        }
        iter++;
    }
    world().DLOG("FVA: done");
}

DefSet& FVA::run(Lam *lam) {
    auto worklist = NodeQueue();
    auto [node, _] = build_node(lam, worklist);
    if (!is_done(node)) {
        cur_pass_id++;
        run(worklist);
    }
    return node->fvs;
}


ClosureConv::Closure ClosureConv::make_closure(Lam* fn, Def2Def& subst) {
    if (auto closure = closures_.lookup(fn))
        return* closure;

    auto& fv_set = fva_.run(fn);
    auto fvs = std::vector<const Def*>();
    auto fvs_types = std::vector<const Def*>();
    for (auto fv: fv_set) {
        fvs.emplace_back(fv);
        fvs_types.emplace_back(rewrite(fv->type(), subst));
    }
    auto env = world().tuple(fvs);
    auto env_type = world().sigma(fvs_types);

    /* Types: rewrite function type here \w fv s */
    auto new_fn_type = closure_type(fn->type(), subst, env_type)->as<Pi>();
    auto new_lam = world().nom_lam(new_fn_type, world().dbg(fn->name()));
    new_lam->set_body(fn->body());
    new_lam->set_filter(fn->filter());
    if (fn->is_external()) { 
        fn->make_internal();
        new_lam->make_external();
    }

    world().DLOG("STUB {} ~~> ({}, {})", fn, new_lam, env);

    auto closure = Closure{fn, fv_set.size(), env, new_lam};
    closures_.emplace(fn, closure);
    closures_.emplace(new_lam, closure);
    worklist_.emplace(new_lam);
    return closure;
}

}
