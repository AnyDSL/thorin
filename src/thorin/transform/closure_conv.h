
#ifndef THORIN_CLOSURE_CONV_H
#define THORIN_CLOSURE_CONV_H

#include <queue>
#include <vector>
#include <set>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

/// Perform free variable analyses.
class FVA {
public:
    FVA(World& world)
        : world_(world)
        , cur_pass_id(1) 
        , lam2nodes_() {};

    /// @p run will compute free defs that appear transitively in @p lam%s body.
    /// Nominal @p Def%s are never considered free (but their free variables are).
    /// Structural @p Def%s containing nominals are broken up.
    /// The results are memorized.
    DefSet& run(Lam *lam);

private:
    struct Node;
    using NodeQueue = std::queue<Node*>;
    using Nodes = std::vector<Node*>;

    struct Node {
        Def *nom;
        DefSet fvs;
        Nodes preds;
        Nodes succs;
        unsigned pass_id;
    };

    bool is_bot(Node* node) { return node->pass_id == 0; }
    bool is_done(Node* node) { 
        return !is_bot(node) && node->pass_id < cur_pass_id; 
    }
    void mark(Node* node) { node->pass_id = cur_pass_id; }

    void split_fv(Def *nom, const Def* fv, DefSet& out);

    std::pair<Node*, bool> build_node(Def* nom, NodeQueue& worklist);
    void run(NodeQueue& worklist);

    World& world() { return world_; }

    World& world_;
    unsigned cur_pass_id;
    DefMap<std::unique_ptr<Node>> lam2nodes_;
};


/// Perform typed closure conversion.
/// Closures are represented using existential types <code>Σent_type.[env_type, cn[ent_type, Args..]]</code>
/// Only non-returning @p Lam%s are converted (i.e that have type cn[...])
/// This can lead to bugs in combinations with @p Axiom%s / @p Lam%s that are polymorphic in their arguments
/// return type:
/// <code>ax : ∀[B]. (int -> B) -> (int -> B)</code> won't be converted, possible arguments may.
/// Further, there is no machinery to handle free variables in a @p Lam%s type; this may lead to
/// problems with polymorphic functions.
/// Neither of this two cases is checked.
/// The type of higher-order @p Axiom%s is adjusted as well.

class ClosureConv {
    public:
        ClosureConv(World& world)
            : world_(world)
            , fva_(world)
            , closures_(DefMap<Closure>())
            , closure_types_(Def2Def())
            , worklist_(std::queue<const Def*>()) {};

        void run();

    private:
        struct Closure {
            Lam* old_fn;
            size_t num_fvs;
            const Def* env;
            Lam* fn;
        };


        const Def* rewrite(const Def* old_def, Def2Def& subst);

        const Def* closure_type(const Pi* pi, Def2Def& subst, const Def* ent_type = nullptr);

        Closure make_closure(Lam* lam, Def2Def& subst);

        World& world() { return world_; }

        World& world_;
        FVA fva_;
        DefMap<Closure> closures_;
        Def2Def closure_types_;
        std::queue<const Def*> worklist_;
};

/// Utils for working with closures

// Functions for matching closure types

Sigma* isa_pct(const Def* def);

const Sigma* isa_uct(const Def* def);

const Sigma* isa_ct(const Def* def, bool typed);

const Def* closure_env_type(World& world);

class ClosureWrapper {
public:
    ClosureWrapper(const Def* def, bool typed)
        : def_(def->isa<Tuple>() && isa_ct(def, typed) ? def->as<Tuple>() : nullptr) {}

    Lam* lam();

    const Def* env();

    operator bool() const {
        return def_ != nullptr;
    }

    operator const Tuple*() {
        return def_;
    }

    const Sigma* type() {
        assert(def_);
        return def_->type()->isa<Sigma>();
    }

    const Pi* old_type();

    unsigned int order() {
        return old_type()->order();
    }

    bool is_basicblock() {
        return old_type()->is_basicblock();
    }

private:
    const Tuple* def_;
};

ClosureWrapper isa_closure(const Def* def, bool typed = true);

};

#endif
