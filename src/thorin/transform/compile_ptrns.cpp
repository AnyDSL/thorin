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
        auto f_ptrn = world_.ptrn(world_.case_(flattener_.flatten(ptrn->type()->domain()), ptrn->type()->codomain()), ptrn->can_be_redundant(), ptrn->debug());
        rewriter.old2new.emplace(ptrn->param(), unflatten(f_ptrn->param(), ptrn->type()->domain()));
        f_ptrn->set(flattener_.flatten(rewriter.rewrite(ptrn->matcher())), rewriter.rewrite(ptrn->body()));
        return f_ptrn;
    }

    /// Eliminates an element from a tuple
    std::pair<const Def*, const Def*> eliminate(const Def* def, size_t col) {
        // Unions and other types may also have an arity.
        auto type = def->type()->reduce();
        if (!type->isa<Sigma>() || type->num_ops() == 0)
            return { def, world_.tuple() };
        Array<const Def*> ops(type->num_ops() - 1);
        for (size_t i = 0; i < col; ++i)
            ops[i] = def->out(i);
        for (size_t i = col, n = ops.size(); i < n; ++i)
            ops[i] = def->out(i + 1);
        return { def->out(col), world_.tuple(ops) };
    }

    /// Introduces an element in a tuple
    const Def* introduce(const Def* def, size_t col, const Def* val) {
        // See above.
        auto type = def->type()->reduce();
        if (!type->isa<Sigma>())
            return world_.tuple({col == 0 ? val : def, col == 0 ? def : val});
        if (type->num_ops() == 0)
            return val;
        Array<const Def*> ops(type->num_ops() + 1);
        for (size_t i = 0; i < col; ++i)
            ops[i] = def->out(i);
        ops[col] = val;
        for (size_t i = col + 1, n = ops.size(); i < n; ++i)
            ops[i] = def->out(i - 1);
        return world_.tuple(ops);
    }

    /// Specializes the pattern for one particular value
    Ptrn* specialize(Ptrn* ptrn, size_t col, const Def* ctor, const Def* arg_col, const Def* s_arg, bool can_be_redundant = false) {
        // Create a pattern with the specialized signature
        auto s_ptrn = world_.ptrn(world_.case_(s_arg->type(), ptrn->body()->type()), can_be_redundant, ptrn->debug());
        const Def* param = ptrn->param(), *s_param = s_ptrn->param();
        if (arg_col) {
            // We have (T, ...) and we want (E::A(T), ...)
            const Def* s_param_col = nullptr;
            std::tie(s_param_col, s_param) = eliminate(s_param, col);
            s_param = introduce(s_param, col, world_.insert(world_.bot(arg_col->type()), ctor, s_param_col));
        } else {
            s_param = introduce(s_param, col, ctor);
        }

        // Rewrite the body and matcher
        Scope scope(ptrn);
        Rewriter rewriter(world_, &scope);
        assert(param->type()->reduce() == s_param->type()->reduce());
        rewriter.old2new.emplace(param, s_param);
        auto s_matcher = rewriter.rewrite(ptrn->matcher());
        if (arg_col) {
            // We have (E::A(T), ...), and we want (T, ...)
            const Def* s_matcher_col = nullptr;
            std::tie(s_matcher_col, s_matcher) = eliminate(s_matcher, col);
            s_matcher = introduce(s_matcher, col, world_.extract(s_matcher_col, ctor));
        } else {
            s_matcher = std::get<1>(eliminate(s_matcher, col));
        }
        s_ptrn->set(s_matcher, rewriter.rewrite(ptrn->body()));
        return s_ptrn;
    }

    /// Returns whether the constructor patterns form a signature for the matched type
    bool is_complete(const Def* arg_type, const DefMap<std::vector<Ptrn*>>& ctor2ptrns) {
        if (auto lit = arg_type->isa<Lit>())
            return ctor2ptrns.size() == lit->get<nat_t>();
        return false;
    }

    /// Report an error message for each pattern that does not have the flag 'can_be_redundant' set
    void report_redundant_ptrns(const Match* match, ArrayRef<Ptrn*> ptrns) {
        if (world_.err()) {
            for (auto ptrn : ptrns) {
                if (!ptrn->can_be_redundant())
                    world_.err()->redundant_match_case(match, ptrn);
            }
        }
    }

    World& world_;
    Flattener flattener_;

public:
    PtrnCompiler(World& world)
        : world_(world)
    {}

    const Def* compile(const Match* match, const Def* arg, std::vector<Ptrn*>& ptrns, const Def* dbg) {
        assert(!ptrns.empty());
        // If the first pattern of the list matches everything, then no need for a match
        if (arg->type()->reduce()->lit_arity() == 0 || ptrns[0]->is_trivial()) {
            report_redundant_ptrns(match, ArrayRef<Ptrn*>(ptrns).skip_front());
            return ptrns[0]->apply(arg);
        }

        // Flatten tuple patterns
        arg = flattener_.flatten(arg);
        for (auto& ptrn : ptrns)
            ptrn = flatten(ptrn);

        // Select a column to specialize on
        size_t col = 0; // TODO: Heuristics
        auto [arg_col, d_arg] = eliminate(arg, col);
        bool has_union = arg_col->type()->reduce()->isa<Union>();
        auto ctor = has_union ? world_.variant(arg_col) : arg_col;
        auto ctor_type = ctor->type();

        // Generate specialized patterns for each constructor pattern
        // and collect patterns that have no constructor.
        std::vector<Ptrn*> no_ctor;
        DefMap<std::vector<Ptrn*>> ctor2ptrns;
        for (auto ptrn : ptrns) {
            auto [ptrn_col, _] = eliminate(ptrn->matcher(), col);
            if (auto lit = ptrn_col->isa<Lit>())
                ctor2ptrns[lit].push_back(ptrn);
            else if (auto insert = ptrn_col->isa<Insert>()) {
                assert(insert->type()->reduce()->isa<Union>());
                assert(insert->index()->isa<Lit>());
                ctor2ptrns[insert->index()].push_back(ptrn);
            } else
                no_ctor.push_back(ptrn);
        }

        // Generate a new match for each constructor
        std::vector<const Def*> compiled_ptrns;
        for (auto& [ctor, ctor_ptrns] : ctor2ptrns) {
            auto s_arg = d_arg;
            // Add the arguments of the union to the pattern if there are any
            if (has_union)
                s_arg = introduce(s_arg, col, world_.extract(arg_col, ctor));
            for (auto& ptrn : ctor_ptrns)
                ptrn = specialize(ptrn, col, ctor, has_union ? arg_col : nullptr, s_arg);
            for (auto ptrn : no_ctor)
                ctor_ptrns.push_back(specialize(ptrn, col, ctor, arg_col, s_arg, true));
            auto value = compile(match, s_arg, ctor_ptrns, dbg);
            auto ptrn = world_.ptrn(world_.case_(ctor_type, value->type()), false, dbg);
            ptrn->set(ctor, value);
            compiled_ptrns.push_back(ptrn);
        }
        auto complete = is_complete(ctor_type, ctor2ptrns);
        if (no_ctor.empty() && !complete) {
            if (world_.err()) world_.err()->incomplete_match(match);
            return world_.bot(match->type());
        }
        if (!no_ctor.empty()) {
            if (complete)
                report_redundant_ptrns(match, no_ctor);
            for (auto& ptrn : no_ctor)
                ptrn = specialize(ptrn, col, arg_col, nullptr, d_arg);
            auto value = compile(match, d_arg, no_ctor, dbg);
            auto ptrn = world_.ptrn(world_.case_(ctor_type, value->type()), false, dbg);
            ptrn->set(ptrn->param(), value);
            compiled_ptrns.push_back(ptrn);
        }

        return world_.match(ctor, compiled_ptrns, dbg);
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
        [&] (const Scope& scope) {
            return scope.entry()->isa<Lam>();
        },
        [&] (const Def* old) -> const Def* {
            if (auto match = old->isa<Match>())
                return compiler.compile(match);
            return nullptr;
        });
}

}
