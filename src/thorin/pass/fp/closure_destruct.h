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
        , iter_(1), clos2dropped_() 
    {}

    void unify(const Def* a, const Def* b);

    void enter() override { 
        iter_++; 
        dump_graph();
    }

    const Def* rewrite(const Def*) override;
    undo_t analyze(const Def*) override;

    using Data = int;

private:
    class Edge;
    class Node;

    Node* get_node(const Def* def, undo_t undo = No_Undo);

    undo_t add_pointee(Node* node, const Def* def);
    undo_t add_pointee(const Def* a, const Def* b, undo_t undo = No_Undo) {
        return add_pointee(get_node(a, undo), b);
    }

    using ArgVec = std::vector<std::pair<size_t, const Def*>>;
    undo_t analyze_call(const Def* callee, ArgVec& vec);

    void dump_node(Node* node);
    void dump_graph();

    size_t iter_;
    DefMap<std::unique_ptr<Node>> def2node_;
    LamMap<std::pair<const Def*, Lam*>> clos2dropped_;
};

struct ClosureDestruct::Edge {
public:
    Edge(Node* node, size_t iter)
        : node_(node), iter_(iter) {};

    Node* operator*() const {
        return node_;
    }

    Node* operator->() const {
        return node_;
    }

    friend bool operator<(Edge a, Edge b) {
        return a.node_ < b.node_;
    }

    friend bool operator==(Edge a, Edge b) {
        return a.node_ == b.node_;
    }

    size_t iter() const { return iter_; }
    void set_iter(size_t new_iter) const {
        iter_ = std::min(iter_, new_iter);
    }

    void dump(Stream& s, std::set<Node*>& visited) const;
    
private:
    Node* const node_;
    mutable size_t iter_; 
};


class ClosureDestruct::Node {
public:
    Node(const Def* def, bool esc = false, undo_t undo = No_Undo);

    template<class T = const Def>
    T* def() { assert(def_); return def_->isa<T>(); };

    inline bool is_repr() {
        return this == repr_;
    }

    Node* repr() {
        if (!is_repr())
            repr_ = repr_->repr();
        return repr_;
    }

    bool is_esc() { 
        if (!is_repr())
            return repr()->is_esc();
        return esc_; 
    }

    undo_t mark_esc();

    undo_t add_pointee(Node* pointee, size_t iter);

    undo_t unify(Node* other);

    void dump(Stream& s, std::set<Node*>& visited);

    friend Stream& operator<<(Stream& s, Node& node) {
        auto visited = std::set<Node*>();
        node.dump(s, visited); 
        return s;
    }

    static Node* top() { return &top_; }

private:
    Node* repr_;
    const Def* def_;
    bool esc_;
    undo_t undo_;
    std::set<Edge> points_to_;

    static Node top_; 
};


}



#endif
