#include "thorin/world.h"
#include "thorin/analyses/verify.h"

namespace thorin {

class Cleaner {
public:
    Cleaner(World& world)
        : world_(world)
    {}

    void cleanup();
    void eliminate_params();
    Def dead_code_elimination(Def, Def2Def&, TypeSet&);
    void unreachable_code_elimination(Lambda*, LambdaSet&, Def2Def&, TypeSet&);
    void unused_type_elimination(const Type*, TypeSet&);
    template<class S, class W> static void wipe_out(S& set, W wipe);

private:
    World& world_;
};

void Cleaner::eliminate_params() {
    for (auto olambda : world_.copy_lambdas()) {
        if (olambda->empty())
            continue;

        olambda->clear();
        std::vector<size_t> proxy_idx;
        std::vector<size_t> param_idx;
        size_t i = 0;
        for (auto param : olambda->params()) {
            if (param->is_proxy())
                proxy_idx.push_back(i++);
            else
                param_idx.push_back(i++);
        }

        if (proxy_idx.empty())
            continue;

        auto nlambda = world_.lambda(world_.pi(olambda->type()->elems().cut(proxy_idx)), olambda->attribute(), olambda->name);
        size_t j = 0;
        for (auto i : param_idx) {
            olambda->param(i)->replace(nlambda->param(j));
            nlambda->param(j++)->name = olambda->param(i)->name;
        }

        nlambda->jump(olambda->to(), olambda->args());
        olambda->destroy_body();

        for (auto use : olambda->uses()) {
            auto ulambda = use->as_lambda();
            assert(use.index() == 0 && "deleted param of lambda used as argument");
            ulambda->jump(nlambda, ulambda->args().cut(proxy_idx));
        }
    }
}

void Cleaner::cleanup() {
    eliminate_params();

    World::PrimOps old;
    swap(world_.primops_, old);
    LambdaSet uce_set;
    Def2Def dce_map;
    TypeSet type_set;

    for (size_t i = 0, e = sizeof(world_.keep_)/sizeof(const Type*); i != e; ++i)
        unused_type_elimination(world_.keep_[i], type_set);

    for (auto lambda : world_.lambdas())
        if (lambda->attribute().is(Lambda::Extern))
            unreachable_code_elimination(lambda, uce_set, dce_map, type_set);

    for (auto lambda : world_.lambdas()) {
        if (!uce_set.contains(lambda))
            lambda->destroy_body();
    }

    auto wipe_lambda = [&] (Lambda* lambda) {
        return !lambda->attribute().is(Lambda::Extern)
            && (  (!lambda->attribute().is(Lambda::Intrinsic) && lambda->empty())
                || (lambda->attribute().is(Lambda::Intrinsic) && lambda->num_uses() == 0));
    };
    auto unlink_representative = [&] (const DefNode* def) {
        if (def->is_proxy()) {
            auto num = def->representative_->representatives_of_.erase(def);
            assert(num == 1);
        }
    };

    for (auto primop : old) {
        for (size_t i = 0, e = primop->size(); i != e; ++i)
            primop->unregister_use(i);
        unlink_representative(primop);
    }

    for (auto lambda : world_.lambdas()) {
        if (wipe_lambda(lambda)) {
            for (auto param : lambda->params())
                unlink_representative(param);
            unlink_representative(lambda);
        }
    }

    wipe_out(world_.lambdas_, wipe_lambda);
    wipe_out(world_.types_, [&] (const Type* type) { return type_set.find(type) == type_set.end(); });
    verify_closedness(world_);
    debug_verify(world_);
}

void Cleaner::unreachable_code_elimination(Lambda* lambda, LambdaSet& uce_set, Def2Def& dce_map, TypeSet& type_set) {
    if (visit(uce_set, lambda)) return;

    for (auto lambda : world_.lambdas()) {
        unused_type_elimination(lambda->type(), type_set);
        for (auto param : lambda->params())
            unused_type_elimination(param->type(), type_set);
    }

    for (size_t i = 0, e = lambda->ops().size(); i != e; ++i)
        lambda->update_op(i, dead_code_elimination(lambda->op(i), dce_map, type_set));

    for (auto succ : lambda->succs())
        unreachable_code_elimination(succ, uce_set, dce_map, type_set);
}

Def Cleaner::dead_code_elimination(Def def, Def2Def& dce_map, TypeSet& type_set) {
    if (auto mapped = find(dce_map, def))
        return mapped;
    if (def->isa<Lambda>() || def->isa<Param>())
        return dce_map[def] = def;

    auto oprimop = def->as<PrimOp>();

    Array<Def> ops(oprimop->size());
    for (size_t i = 0, e = oprimop->size(); i != e; ++i)
        ops[i] = dead_code_elimination(oprimop->op(i), dce_map, type_set);

    auto nprimop = world_.rebuild(oprimop, ops);
    assert(nprimop != oprimop);
    unused_type_elimination(nprimop->type(), type_set);
    return dce_map[oprimop] = nprimop;
}

void Cleaner::unused_type_elimination(const Type* type, TypeSet& type_set) {
    assert(world_.types_.find(type) != world_.types_.end() && "not in map");
    if (type_set.find(type) != type_set.end()) return; // use visit
    type_set.insert(type);

    for (auto elem : type->elems())
        unused_type_elimination(elem, type_set);
}

template<class S, class W>
void Cleaner::wipe_out(S& set, W wipe) {
    for (auto i = set.begin(); i != set.end();) {
        auto j = i++;
        auto val = *j;
        if (wipe(val)) {
            set.erase(j);
            delete val;
        }
    }
}

void cleanup_world(World& world) { Cleaner(world).cleanup(); }

}
