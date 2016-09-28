#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/verify.h"
#include "thorin/transform/import.h"
#include "thorin/util/queue.h"

namespace thorin {

class Cleaner {
public:
    Cleaner(World& world)
        : world_(world)
    {}

    World& world() { return world_; }
    void cleanup();
    void merge_continuations();
    void eliminate_params();
    void rebuild();
    void verify_closedness();
    void within(const Def*);

private:
    World& world_;
};

class Merger {
public:
    Merger(const Scope& scope)
        : scope(scope)
        , cfg(scope.f_cfg())
        , domtree(cfg.domtree())
    {
        merge(domtree.root());
    }

    void merge(const CFNode* n);
    const CFNode* dom_succ(const CFNode* n);
    World& world() { return scope.world(); }

    const Scope& scope;
    const F_CFG& cfg;
    const DomTree& domtree;
};

const CFNode* Merger::dom_succ(const CFNode* n) {
    const auto& succs = cfg.succs(n);
    const auto& children = domtree.children(n);
    if (succs.size() == 1 && children.size() == 1 && *succs.begin() == (*children.begin())) {
        auto continuation = (*succs.begin())->continuation();
        if (continuation->num_uses() == 1 && continuation == n->continuation()->callee())
            return children.front();
    }
    return nullptr;
}

void Merger::merge(const CFNode* n) {
    auto cur = n;
    for (auto next = dom_succ(cur); next != nullptr; cur = next, next = dom_succ(next)) {
        assert(cur->continuation()->num_args() == next->continuation()->num_params());
        for (size_t i = 0, e = cur->continuation()->num_args(); i != e; ++i)
            next->continuation()->param(i)->replace(cur->continuation()->arg(i));
        cur->continuation()->destroy_body();
    }

    if (cur != n)
        n->continuation()->jump(cur->continuation()->callee(), cur->continuation()->type_args(), cur->continuation()->args(), cur->continuation()->jump_loc());

    for (auto child : domtree.children(cur))
        merge(child);
}

void Cleaner::merge_continuations() {
    Scope::for_each(world(), [] (const Scope& scope) { Merger merger(scope); });
}

void Cleaner::eliminate_params() {
    for (auto ocontinuation : world().copy_continuations()) {
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;

        if (!ocontinuation->empty() && !world().is_external(ocontinuation)) {
            for (auto use : ocontinuation->uses()) {
                if (use.index() != 0 || !use->isa_continuation())
                    goto next_continuation;
            }

            for (size_t i = 0, e = ocontinuation->num_params(); i != e; ++i) {
                auto param = ocontinuation->param(i);
                if (param->num_uses() == 0)
                    proxy_idx.push_back(i);
                else
                    param_idx.push_back(i);
            }

            if (!proxy_idx.empty() && ocontinuation->num_type_params() == 0) { // TODO do this for polymorphic functions, too
                auto new_fn_type = world().fn_type(ocontinuation->type()->args().cut(proxy_idx));
                auto ncontinuation = world().continuation(new_fn_type, ocontinuation->loc(), ocontinuation->cc(), ocontinuation->intrinsic(), ocontinuation->name);
                size_t j = 0;
                for (auto i : param_idx) {
                    ocontinuation->param(i)->replace(ncontinuation->param(j));
                    ncontinuation->param(j++)->name = ocontinuation->param(i)->name;
                }

                ncontinuation->jump(ocontinuation->callee(), ocontinuation->type_args(), ocontinuation->args(), ocontinuation->jump_loc());
                ocontinuation->destroy_body();

                for (auto use : ocontinuation->uses()) {
                    auto ucontinuation = use->as_continuation();
                    assert(use.index() == 0);
                    ucontinuation->jump(ncontinuation, ucontinuation->type_args(), ucontinuation->args().cut(proxy_idx), ucontinuation->jump_loc());
                }
            }
        }
next_continuation:;
    }
}

void Cleaner::rebuild() {
    World new_world(world().name());
    Def2Def old2new;
    Type2Type type_old2new;

    for (auto external : world().externals())
        import(new_world, type_old2new, old2new, external);

    swap(world_, new_world);
}

void Cleaner::verify_closedness() {
    auto check = [&](const Def* def) {
        size_t i = 0;
        for (auto op : def->ops()) {
            within(op);
            assert(op->uses_.find(Use(i++, def)) != op->uses_.end() && "can't find def in op's uses");
        }

        for (auto use : def->uses_) {
            within(use.def());
            assert(use->op(use.index()) == def && "can't use doesn't point to def");
        }
    };

    for (auto primop : world().primops())
        check(primop);
    for (auto continuation : world().continuations()) {
        check(continuation);
        for (auto param : continuation->params())
            check(param);
    }
}

void Cleaner::within(const Def* def) {
    assert(world().types().contains(def->type()));
    if (auto primop = def->isa<PrimOp>())
        assert_unused(world().primops().contains(primop));
    else if (auto continuation = def->isa_continuation())
        assert_unused(world().continuations().contains(continuation));
    else
        within(def->as<Param>()->continuation());
}

void Cleaner::cleanup() {
#ifndef NDEBUG
    for (const auto& p : world().trackers_)
        assert(p.second.empty() && "there are still live trackers before running cleanup");
#endif

    merge_continuations();
    eliminate_params();
    rebuild();

#ifndef NDEBUG
    verify_closedness();
    debug_verify(world());
#endif
}

void cleanup_world(World& world) { Cleaner(world).cleanup(); }

}
