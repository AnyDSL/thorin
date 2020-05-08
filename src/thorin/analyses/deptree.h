#ifndef THORIN_ANALYSES_DEPTREE_H
#define THORIN_ANALYSES_DEPTREE_H

#include "thorin/def.h"

namespace thorin {

class DepTree {
public:
    class Node {
    public:
        Node() = default;
        Node(Def* nominal, size_t depth)
            : nominal_(nominal)
            , depth_(depth)
        {}

        Def* nominal() const { return nominal_; }
        size_t depth() const { return depth_; }
        Node* parent() const { return parent_; }
        const std::vector<std::unique_ptr<Node>>& children() const { return children_; }

    private:
        Node* set_parent(Node* parent) {
            parent_ = parent;
            depth_ = parent->depth() + 1;
            parent->children_.emplace_back(this);
            return this;
        }

        Def* nominal_;
        size_t depth_;
        Node* parent_ = nullptr;
        std::vector<std::unique_ptr<Node>> children_;

        friend class DepTree;
    };

    DepTree(World& world)
        : world_(world)
        , root_(std::make_unique<Node>(nullptr, 0))
    {
        run();
    }

    World& world() { return world_; };

private:
    void run();
    ParamSet run(Def*);
    ParamSet run(Def*, const Def*);
    static void adjust_depth(Node* node, size_t depth);

    World& world_;
    std::unique_ptr<Node> root_;
    NomMap<Node*> nom2node_;
    DefMap<ParamSet> def2params_;
    std::deque<Node*> stack_;
};

}

#endif
