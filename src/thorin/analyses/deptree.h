#ifndef THORIN_ANALYSES_DEPTREE_H
#define THORIN_ANALYSES_DEPTREE_H

#include "thorin/def.h"

namespace thorin {

class DepNode {
public:
    DepNode(Def* nominal, size_t depth)
        : nominal_(nominal)
        , depth_(depth)
    {}

    Def* nominal() const { return nominal_; }
    size_t depth() const { return depth_; }
    DepNode* parent() const { return parent_; }
    const std::vector<DepNode*>& children() const { return children_; }

private:
    DepNode* set_parent(DepNode* parent) {
        parent_ = parent;
        depth_ = parent->depth() + 1;
        parent->children_.emplace_back(this);
        return this;
    }

    Def* nominal_;
    size_t depth_;
    DepNode* parent_ = nullptr;
    std::vector<DepNode*> children_;

    friend class DepTree;
};

class DepTree {
public:
    DepTree(const World& world)
        : world_(world)
        , root_(std::make_unique<DepNode>(nullptr, 0))
    {
        run();
    }

    const World& world() const { return world_; };
    const DepNode* root() const { return root_.get(); }
    const DepNode* nom2node(Def* nom) const { return nom2node_.find(nom)->second.get(); }
    bool depends(Def* a, Def* b) const; ///< Does @p a depend on @p b?

private:
    void run();
    ParamSet run(Def*);
    ParamSet run(Def*, const Def*);
    static void adjust_depth(DepNode* node, size_t depth);

    const World& world_;
    std::unique_ptr<DepNode> root_;
    NomMap<std::unique_ptr<DepNode>> nom2node_;
    DefMap<ParamSet> def2params_;
    std::deque<DepNode*> stack_;
};

}

#endif
