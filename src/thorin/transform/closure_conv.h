
#ifndef THORIN_CLOSURE_CONV_H
#define THORIN_CLOSURE_CONV_H

#include <queue>
#include <vector>
#include <set>

#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

class FVA {
public:
    FVA(World& world)
        : world_(world)
        , cur_pass_id(1) 
        , lam2nodes_() {};

    DefSet& run(Lam *lam);

private:
    struct Node;
    using NodeQueue = std::queue<Node*>;
    using Nodes = std::vector<Node*>;

    struct Node {
        Lam *lam;
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

    void split_fv(const Def* fv, DefSet& out);

    std::pair<Node*, bool> build_node(Lam* lam, NodeQueue& worklist);
    void run(NodeQueue& worklist);

    World& world() { return world_; }

    World& world_;
    unsigned cur_pass_id;
    DefMap<std::unique_ptr<Node>> lam2nodes_;
};

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

};

#endif
