#include "thorin/world.h"
#include "thorin/analyses/verify.h"

namespace thorin {

class Cleaner {
public:
    Cleaner(World& world)
        : world_(world)
    {}

    World::PrimOps& oprimops() { return oprimops_; }
    World::PrimOps& nprimops() { return world_.primops_; }
    LambdaSet& olambdas() { return olambdas_; }
    LambdaSet& nlambdas() { return world_.lambdas_; }
    World::Types& otypes() { return world_.types_; }
    TypeSet& ntypes() { return ntypes_; }

    void cleanup();
    void eliminate_params();
    Def dead_code_elimination(Def);
    void unreachable_code_elimination(Lambda*);
    void unused_type_elimination(const TypeNode*);

private:
    World& world_;
    World::PrimOps oprimops_;
    LambdaSet olambdas_;
    TypeSet ntypes_;
    Def2Def dce_map_;
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

        auto nlambda = world_.lambda(world_.fn_type(olambda->type()->args().cut(proxy_idx)), olambda->attribute(), olambda->intrinsic(), olambda->name);
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

    swap(world_.primops_, oprimops_);
    swap(world_.lambdas_, olambdas_);

    // collect all reachable and live stuff
    for (auto lambda : olambdas())
        if (lambda->attribute().is(Lambda::Extern))
            unreachable_code_elimination(lambda);

#if 0
    // collect all used types
    for (size_t i = 0, e = sizeof(world_.keep_)/sizeof(const TypeNode*); i != e; ++i)
        unused_type_elimination(world_.keep_[i]);
    for (auto primop : nprimops())
        unused_type_elimination(*primop->type());
    for (auto lambda : nlambdas()) {
        unused_type_elimination(*lambda->type());
        for (auto param : lambda->params())
            unused_type_elimination(*param->type());
    }
#endif

    // destroy bodies of unreachable lambdas
    for (auto lambda : olambdas()) {
        if (!nlambdas().contains(lambda))
            lambda->destroy_body();
    }

    auto unlink_representative = [&] (const DefNode* def) {
        if (def->is_proxy()) {
            auto num = def->representative_->representatives_of_.erase(def);
            assert(num == 1);
        }
    };

    // unlink dead primops from the rest
    for (auto primop : oprimops()) {
        for (size_t i = 0, e = primop->size(); i != e; ++i)
            primop->unregister_use(i);
        unlink_representative(primop);
    }

    // unlink unreachable lambdas from the rest
    for (auto lambda : olambdas()) {
        if (!nlambdas().contains(lambda)) {
            for (auto param : lambda->params())
                unlink_representative(param);
            unlink_representative(lambda);
        }
    }

    verify_closedness(world_);

    // delete dead primops
    for (auto primop : oprimops()) 
        delete primop;

    // delete unreachable lambdas
    for (auto lambda : olambdas()) {
        if (!nlambdas().contains(lambda))
            delete lambda;
    }

#if 0
    // delete unused types and remove from otypes map
    for (auto i = otypes().begin(); i != otypes().end();) {
        auto j = i++;
        auto type = *j;
        if (!ntypes().contains(type)) {
            otypes().erase(j);
            delete type;
        }
    }
#endif

    debug_verify(world_);
}

void Cleaner::unreachable_code_elimination(Lambda* lambda) {
    if (visit(nlambdas(), lambda)) return;

    for (size_t i = 0, e = lambda->ops().size(); i != e; ++i)
        lambda->update_op(i, dead_code_elimination(lambda->op(i)));

    for (auto succ : lambda->succs())
        unreachable_code_elimination(succ);
}

Def Cleaner::dead_code_elimination(Def def) {
    if (auto mapped = find(dce_map_, def))
        return mapped;
    if (def->isa<Lambda>() || def->isa<Param>())
        return dce_map_[def] = def;

    auto oprimop = def->as<PrimOp>();

    Array<Def> ops(oprimop->size());
    for (size_t i = 0, e = oprimop->size(); i != e; ++i)
        ops[i] = dead_code_elimination(oprimop->op(i));

    auto nprimop = world_.rebuild(oprimop, ops);
    assert(nprimop != oprimop);
    return dce_map_[oprimop] = nprimop;
}

void Cleaner::unused_type_elimination(const TypeNode* type) {
    if (visit(ntypes(), type)) return;
    for (auto arg : type->args())
        unused_type_elimination(*arg);
}

void cleanup_world(World& world) { Cleaner(world).cleanup(); }

}
