#ifndef THORIN_CLOSURE_DESTRUCT_H
#define THORIN_CLOSURE_DESTRUCT_H

#include <set>
#include <map>
#include "thorin/pass/pass.h"

namespace thorin {

// class PTG;

class ClosureDestruct : public FPPass<ClosureDestruct, Lam> {
public:
    ClosureDestruct(PassMan& man) 
        : FPPass<ClosureDestruct, Lam>(man, "closure_destruct")
        , iter_(0), clos2dropped_() 
    {}

    void unify(const Def* a, const Def* b);

    void enter() override { iter_++; }
    const Def* rewrites(const Def*);
    undo_t analyze(const Def*) override;

    using Data = struct {};

private:
    class Edge;
    class Node;

    Node* get_node(const Def* def, undo_t undo = No_Undo);

    undo_t add_pointee(Node* node, const Def* def);
    undo_t add_pointee(const Def* a, const Def* b, undo_t undo = No_Undo) {
        return add_pointee(get_node(a, undo), b);
    }

    undo_t analyze_call(const Def* callee, size_t i, const Def* arg);

    size_t iter_;
    DefMap<std::unique_ptr<Node>> def2node_;
    LamMap<std::pair<const Def*, Lam*>> clos2dropped_;
};

}



#endif
