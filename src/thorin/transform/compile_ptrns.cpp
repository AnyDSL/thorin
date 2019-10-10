#include "thorin/transform/compile_ptrns.h"
#include "thorin/world.h"
#include "thorin/rewrite.h"
#include "thorin/flatten.h"
#include "thorin/error.h"

namespace thorin {

/// Pattern matching compiler based on:
/// "Compiling Pattern Matching to Good Decision Trees"
/// by Luc Maranget
class PtrnCompiler {
private:
    /// Flattens the tuples/packs in a pattern
    Ptrn* flatten(Ptrn* ptrn) {
        Scope scope(ptrn);
        Rewriter rewriter(world_, &scope);
        auto f_ptrn = world_.ptrn(world_.case_(flattener_.flatten(ptrn->type()->domain()), ptrn->type()->codomain()), ptrn->debug());
        rewriter.old2new.emplace(ptrn->param(), unflatten(f_ptrn->param(), ptrn->type()->domain()));
        f_ptrn->set(flattener_.flatten(rewriter.rewrite(ptrn->matcher())), rewriter.rewrite(ptrn->body()));
        return f_ptrn;
    }

    /// Eliminates an element from a def
    const Def* eliminate(const Def* def, size_t col) {
        Array<const Def*> ops(def->type()->lit_arity() - 1);
        for (size_t i = 0; i < col; ++i)
            ops[i] = def->out(i);
        for (size_t i = col, n = ops.size(); i < n; ++i)
            ops[i] = def->out(i + 1);
        return world_.tuple(ops);
    }

    /// Introduces an element in a def
    const Def* introduce(const Def* def, size_t col, const Def* val) {
        Array<const Def*> ops(def->type()->lit_arity() + 1);
        for (size_t i = 0; i < col; ++i)
            ops[i] = def->out(i);
        ops[col] = val;
        for (size_t i = col + 1, n = ops.size(); i < n; ++i)
            ops[i] = def->out(i - 1);
        return world_.tuple(ops);
    }

    /// Specializes the pattern for one particular value
    Ptrn* specialize(Ptrn* ptrn, size_t col, const Def* val, const Def* s_type) {
        Scope scope(ptrn);
        Rewriter rewriter(world_, &scope);
        auto s_ptrn = world_.ptrn(world_.case_(s_type, ptrn->body()->type()), ptrn->debug());
        rewriter.old2new.emplace(ptrn->param(), introduce(s_ptrn->param(), col, val));
        s_ptrn->set(eliminate(rewriter.rewrite(ptrn->matcher()), col), rewriter.rewrite(ptrn->body()));
        return s_ptrn;
    }

    /// Returns whether the constructor patterns form a signature for the matched type
    bool is_complete(const Def* arg_type, const DefMap<std::vector<Ptrn*>>& ctor2ptrns) {
        return arg_type == world_.type_bool() && ctor2ptrns.size() == 2;
    }

    World& world_;
    Flattener flattener_;

public:
    PtrnCompiler(World& world)
        : world_(world)
    {}

    const Def* compile(const Match* match, const Def* arg, std::vector<Ptrn*>& ptrns, const Def* dbg) {
        assert(!ptrns.empty());
        if (arg->type()->lit_arity() == 1) {
            // The reinterpret_cast is need to case the Ptrn** into Def**,
            // which is not a valid C++ static_cast, but should be safe as a
            // reinterpret_cast because we do not modify the contents of the array.
            return world_.match(arg, Defs(ptrns.size(), reinterpret_cast<Def**>(ptrns.data())), dbg);
        }

        // Flatten tuple patterns
        arg = flattener_.flatten(arg);
        for (auto& ptrn : ptrns)
            ptrn = flatten(ptrn);

        // Select a column to specialize on
        size_t col = 0; // TODO: Heuristics
        auto col_arg = arg->out(col);
        auto col_type = col_arg->type();

        // Generate specialized patterns for each constructor pattern
        // and collect patterns that have no constructor.
        std::vector<Ptrn*> no_ctor;
        DefMap<std::vector<Ptrn*>> ctor2ptrns;
        auto s_arg = eliminate(arg, col);
        for (auto ptrn : ptrns) {
            auto ptrn_col = ptrn->matcher()->out(col);
            if (auto lit = ptrn_col->isa<Lit>())
                ctor2ptrns[lit].push_back(specialize(ptrn, col, lit, s_arg->type()));
            else
                no_ctor.push_back(specialize(ptrn, col, col_arg, s_arg->type()));
        }

        // Generate a new match for each constructor
        std::vector<const Def*> compiled_ptrns;
        for (auto& [ctor, ctor_ptrns] : ctor2ptrns) {
            ctor_ptrns.insert(ctor_ptrns.end(), no_ctor.begin(), no_ctor.end());
            auto value = compile(match, s_arg, ctor_ptrns, dbg);
            auto ptrn = world_.ptrn(world_.case_(col_type, value->type()), dbg);
            ptrn->set(ctor, value);
            compiled_ptrns.push_back(ptrn);
        }
        if (no_ctor.empty() && !is_complete(col_type, ctor2ptrns))
            world_.err()->incomplete_match(match);
        if (!no_ctor.empty()) {
            auto value = compile(match, s_arg, no_ctor, dbg);
            auto ptrn = world_.ptrn(world_.case_(col_type, value->type()), dbg);
            ptrn->set(ptrn->param(), value);
            compiled_ptrns.push_back(ptrn);
        }

        return world_.match(col_arg, compiled_ptrns, dbg);
    }

    const Def* compile(const Match* match) {
        std::vector<Ptrn*> ptrns(match->ptrns().size());
        for (size_t i = 0, n = ptrns.size(); i < n; ++i)
            ptrns[i] = match->ptrn(i)->as_nominal<Ptrn>();
        return compile(match, match->arg(), ptrns, match->debug());
    }
};

void compile_ptrns(World& world) {
    PtrnCompiler compiler(world);
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
