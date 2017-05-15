#ifndef THORIN_ANALYSES_NEST_H
#define THORIN_ANALYSES_NEST_H

#include <memory>

#include "thorin/def.h"
#include "thorin/continuation.h"
#include "thorin/util/iterator.h"

namespace thorin {

class Nest {
public:
    class Node {
    public:
        Node(Continuation* continuation, const Node* parent, int depth)
            : continuation_(continuation)
            , parent_(parent)
            , depth_(depth)
        {}

        Continuation* continuation() const { return continuation_; }
        const Node* parant() const { return parent_; }
        ArrayRef<std::unique_ptr<const Node>> children() const { return children_; }
        int depth() const { return depth_; }

    private:
        /// Give birth to a new child containing @p continuation with current Node as parent.
        const Node* bear(Continuation* continuation) const {
            children_.emplace_back(std::make_unique<const Node>(continuation, this, depth() + 1));
            return children_.back().get();
        }

        Continuation* continuation_;
        const Node* parent_;
        mutable std::vector<std::unique_ptr<const Node>> children_;
        int depth_;

        friend class Nest;
    };

    Nest(const Scope&);

    const Scope& scope() const { return scope_; }
    ArrayRef<const Node*> top_down() const { return top_down_; }
    auto bottom_up() const { return range(top_down().rbegin(), top_down().rend()); }
    const Node* node(Continuation* continuation) const { auto i = map_.find(continuation); /*assert(i != map_.end());*/ return i == map_.end() ? nullptr : i->second; }
    int depth(Continuation* continuation) const { return node(continuation)->depth(); }

private:
    std::unique_ptr<const Node> run();
    const Node* def2node(const Def*);

    const Scope& scope_;
    DefMap<const Node*> def2node_;
    Array<const Node*> top_down_;
    ContinuationMap<const Node*> map_;
    std::unique_ptr<const Node> root_;
};


}

#endif
