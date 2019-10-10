#include "thorin/transform/compile_ptrns.h"
#include "thorin/world.h"
#include "thorin/rewrite.h"
#include "thorin/flatten.h"

namespace thorin {

/// Pattern matching compiler based on:
/// "Compiling Pattern Matching to Good Decision Trees"
/// by Luc Maranget
class PtrnCompiler {
private:
    /// Flattens the tuples/packs in a pattern
    Ptrn* flatten(Ptrn* ptrn) {
        auto& world = ptrn->world();
        Scope scope(ptrn);
        Rewriter rewriter(world, &scope);
        auto expanded = world.ptrn(world.case_(flattener.flatten(ptrn->type()->domain()), ptrn->type()->codomain()), ptrn->debug());
        rewriter.old2new.emplace(ptrn->param(), unflatten(expanded->param(), ptrn->type()->domain()));
        expanded->set(flattener.flatten(rewriter.rewrite(ptrn->matcher())), rewriter.rewrite(ptrn->body()));
        return expanded;
    }

    /// Eliminates element 'col' from @p def
    static const Def* eliminate(const Def* def, size_t col) {
        auto& world = def->world();
        if (auto tuple = def->isa<Tuple>()) {
            Array<const Def*> ops(tuple->num_ops() - 1);
            auto cur = std::copy(tuple->ops().begin(), tuple->ops().begin() + col, ops.begin());
            std::copy(tuple->ops().begin() + col + 1, tuple->ops().end(), cur);
            return world.tuple(ops, tuple->debug());
        } else if (auto pack = def->isa<Pack>()) {
            assert(pack->type()->arity()->isa<Lit>());
            return world.pack(pack->type()->lit_arity() - 1, pack->body(), pack->debug());
        } else {
            THORIN_UNREACHABLE;
        }
    }

    Ptrn* specialize(Ptrn* ptrn, size_t col, const Def* val = nullptr) {
        // TODO
        return nullptr;
    }

    Flattener flattener;

public:
    const Def* compile(const Def* arg, std::vector<Ptrn*>& ptrns, const Def* dbg) {
        assert(!ptrns.empty());

        // Flatten tuple patterns
        arg = flattener.flatten(arg);
        for (auto& ptrn : ptrns)
            ptrn = flatten(ptrn);

        // Select a column to specialize on
        size_t col = 0; // TODO: Heuristics
        auto col_type = proj(arg->type(), col);

        // Generate specialized patterns for each constructor pattern
        // and collect patterns that have no constructor.
        auto& world = arg->world();
        std::vector<Ptrn*> no_ctor;
        DefMap<std::vector<Ptrn*>> ctor2ptrns;
        for (auto ptrn : ptrns) {
            auto ptrn_col = ptrn->matcher()->out(col);
            if (auto lit = ptrn_col->isa<Lit>()) {
                ctor2ptrns[lit].push_back(specialize(ptrn, col, lit));
            } else {
                no_ctor.push_back(specialize(ptrn, col));
            }
        }

        // Generate a new match for each constructor
        std::vector<const Def*> compiled_ptrns;
        auto s_arg = eliminate(arg, col);
        for (auto& [ctor, ctor_ptrns] : ctor2ptrns) {
            auto value = compile(s_arg, ctor_ptrns, dbg);
            auto ptrn = world.ptrn(world.case_(col_type, value->type()), dbg);
            ptrn->set(ctor, value);
            compiled_ptrns.push_back(ptrn);
        }
        // TODO: Completeness/redundancy check
        if (!no_ctor.empty()) {
            auto value = compile(s_arg, no_ctor, dbg);
            auto ptrn = world.ptrn(world.case_(col_type, value->type()), dbg);
            ptrn->set(ptrn->param(), value);
            compiled_ptrns.push_back(ptrn);
        }

        return world.match(arg->out(col), compiled_ptrns, dbg);
    }

    const Def* compile(const Match* match) {
        std::vector<Ptrn*> ptrns(match->ptrns().size());
        for (size_t i = 0, n = ptrns.size(); i < n; ++i)
            ptrns[i] = match->ptrn(i)->as_nominal<Ptrn>();
        return compile(match->arg(), ptrns, match->debug());
    }
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
