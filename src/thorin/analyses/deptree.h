#ifndef THORIN_ANALYSES_DEPTREE_H
#define THORIN_ANALYSES_DEPTREE_H

#include "thorin/def.h"

namespace thorin {

class DepTree {
public:
    class Node {
    public:
        Node() = default;
        Node(Def* nominal)
            : nominal_(nominal)
        {}

        Def* nominal() const { return nominal_; }
        Node* parent() const { return parent_; }
        size_t depth() const { return depth_; }

    private:
        Node* set_parent(Node* parent) {
            parent_ = parent;
            depth_ = parent->depth() + 1;
            parent->children_.emplace_back(this);
            return this;
        }

        Def* nominal_;
        Node* parent_ = nullptr;
        size_t depth_ = 0;
        std::vector<std::unique_ptr<Node>> children_;

        friend class DepTree;
    };

    DepTree(World& world)
        : world_(world)
    {
        run();
    }

    World& world() { return world_; };

private:
    void run();
    Node* run(Def*);
    Node* run(Def*, const Def*);

    World& world_;
    NomMap<Node*> nom2node_;
    DefSet done_;
    std::vector<std::unique_ptr<Node>> roots_;
};

}

#endif
