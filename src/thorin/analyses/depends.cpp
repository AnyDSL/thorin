#include "thorin/def.h"

namespace thorin {

class DepChecker {
public:
    struct Node {
        Node() = default;
        Node(Def* nom)
            : nom(nom)
            , parent(nullptr)
        {}
        Node(Def* nom, Node* parent)
            : nom(nom)
            , parent(parent)
        {
            parent->children.emplace_back(this);
        }

        Def* nom;
        Node* parent;
        std::vector<std::unique_ptr<Node>> children;
    };

    DepChecker(const Def* on)
        : on_(on)
    {}

    bool depends(const Def* def) {
        if (def->is_const() || !done_.emplace(def).second) return false;

        for (auto op : def->extended_ops()) {
            if (depends(op)) return true;
        }

        return false;
    }

private:
    std::vector<std::unique_ptr<Node>> children;
    const Def* on_;
    DefSet done_;
};

bool depends(const Def* def, const Def* on) {
    DepChecker checker(on);
    return checker.depends(def);
}

}
