#ifndef ANYDSL2_NODE_H
#define ANYDSL2_NODE_H

#include <cassert>

#include "anydsl2/enums.h"
#include "anydsl2/util/array.h"
#include "anydsl2/util/cast.h"

namespace anydsl2 {

class Node : public MagicCast {
private:

    /// Do not copy-assign a \p Node instance.
    Node& operator = (const Node&);

protected:


    /// This variant leaves internal \p ops_ \p Array allocateble via ops_.alloc(size).
    Node(int kind) 
        : kind_(kind) 
        , cur_pass_(0)
    {}
    Node(int kind, size_t size)
        : kind_(kind)
        , ops_(size)
        , cur_pass_(0)
    {}
    Node(const Node& node)
        : debug(node.debug)
        , kind_(node.kind())
        , ops_(node.ops_.size())
        , cur_pass_(0)
    {}
    virtual ~Node() {}

    void set(size_t i, const Node* n) { ops_[i] = n; }

    virtual bool equal(const Node* other) const;
    virtual size_t hash() const;

public:

    int kind() const { return kind_; }
    bool is_corenode() const { return ::anydsl2::is_corenode(kind()); }
    NodeKind node_kind() const { assert(is_corenode()); return (NodeKind) kind_; }

    /// Just do what ever you want with this field.
    mutable std::string debug;

    template<class T>
    ArrayRef<T> ops_ref() const { return ops_.ref().cast<T>(); }

    void alloc(size_t size) { ops_.alloc(size); }
    void realloc(size_t size) { 
        ops_.~Array<const Node*>(); 
        new (&ops_) Array<const Node*>();
        alloc(size); 
    }
    void shrink(size_t newsize) { ops_.shrink(newsize); }
    size_t size() const { return ops_.size(); }
    bool empty() const { return ops_.empty(); }
    bool valid() const { return ops_.valid(); }

    /** 
     * Use this field in order to annotate information on this Def.
     * Various analyses have to memorize different stuff temporally.
     * Each analysis can use this field for its specific information. 
     * \attention { 
     *      Each pass/analysis simply overwrites this field again.
     *      So keep this in mind and perform copy operations in order to
     *      save your data before running the next pass/analysis.
     *      Also, keep in mind to perform clean up operations at the end 
     *      of your pass/analysis.
     * }
     */
    union {
        mutable void* ptr;
        mutable const void* cptr;
        mutable bool flags[sizeof(void*)/sizeof(bool)];
    };

    /*
     * scratch operations
     */

    size_t cur_pass() const { return cur_pass_; }
    bool visit(size_t pass) const { 
        assert(cur_pass_ <= pass); 
        if (cur_pass_ != pass) { 
            cur_pass_ = pass; 
            return false; 
        } 
        return true; 
    }
    bool is_visited(size_t pass) const { assert(cur_pass_ <= pass); return cur_pass_ == pass; }

private:

    int kind_;
    Array<const Node*> ops_;
    mutable size_t cur_pass_;

    friend class World;
    friend class NodeHash;
    friend class NodeEqual;
};

//------------------------------------------------------------------------------

struct NodeHash : std::unary_function<const Node*, size_t> {
    size_t operator () (const Node* n) const { return n->hash(); }
};

struct NodeEqual : std::binary_function<const Node*, const Node*, bool> {
    bool operator () (const Node* n1, const Node* n2) const { return n1->equal(n2); }
};

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif // ANYDSL2_NODE_H
