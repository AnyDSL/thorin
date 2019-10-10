#include "thorin/transform/compile_ptrns.h"
#include "thorin/world.h"
#include "thorin/rewrite.h"
#include "thorin/flatten.h"

namespace thorin {

struct PtrnCompiler {
    Ptrn* expand(Ptrn* ptrn) {
        auto& world = ptrn->world();
        Scope scope(ptrn);
        Rewriter rewriter(world, &scope);
        auto expanded = world.ptrn(world.case_(flattener.flatten(ptrn->type()->domain()), ptrn->type()->codomain()), ptrn->debug());
        rewriter.old2new.emplace(ptrn->param(), unflatten(expanded->param(), ptrn->type()->domain()));
        expanded->set(flattener.flatten(rewriter.rewrite(ptrn->matcher())), rewriter.rewrite(ptrn->body()));
        expanded->matcher()->dump();
        expanded->body()->dump();
        expanded->type()->dump();
        return expanded;
    }

    const Def* expand(const Match* match) {
        Array<const Def*> cases(match->cases().size(), [&] (size_t i) {
            return expand(match->cases()[i]->as_nominal<Ptrn>());
        });
        return match->world().match(flattener.flatten(match->arg()), cases);
    }

    const Def* compile(const Match* match) {
        return expand(match);
    }

    Flattener flattener;
};

void compile_ptrns(World& world) {
    PtrnCompiler compiler;
    world.rewrite("compile_ptrns",
        [&](const Scope& scope) {
            return scope.entry()->isa<Lam>();
        },
        [&](const Def* old) -> const Def* {
            if (auto match = old->isa<Match>()) 
                return compiler.compile(match);
            return nullptr;
        });
}

}
