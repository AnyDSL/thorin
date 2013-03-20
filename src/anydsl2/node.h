#ifndef ANYDSL2_NODE_H
#define ANYDSL2_NODE_H

#include <cassert>
#include <vector>

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
    Node(int kind, const std::string& name)
        : kind_(kind) 
        , cur_pass_(0)
        , name(name)
    {}
    Node(int kind, size_t size, const std::string& name)
        : kind_(kind)
        , ops_(size)
        , cur_pass_(0)
        , name(name)
    {}
    Node(const Node& node)
        : kind_(node.kind())
        , ops_(node.ops_.size())
        , cur_pass_(0)
        , name(node.name)
    {}
    virtual ~Node() {}

    void set(size_t i, const Node* n) { ops_[i] = n; }
    void resize(size_t n) { ops_.resize(n, 0); }

public:

    int kind() const { return kind_; }
    bool is_corenode() const { return ::anydsl2::is_corenode(kind()); }
    NodeKind node_kind() const { assert(is_corenode()); return (NodeKind) kind_; }
    template<class T>
    ArrayRef<T> ops_ref() const { return ArrayRef<T>((T*) &ops_.front(), ops_.size()); }
    size_t size() const { return ops_.size(); }
    bool empty() const { return ops_.empty(); }

    /*
     * scratch operations
     */

    size_t cur_pass() const { return cur_pass_; }
    bool visit(const size_t pass) const { 
        assert(cur_pass_ <= pass); 
        if (cur_pass_ != pass) { 
            cur_pass_ = pass; 
            return false; 
        } 
        return true; 
    }
    void visit_first(const size_t pass) const { assert(!is_visited(pass)); cur_pass_ = pass; }
    void unvisit(const size_t pass) const { assert(cur_pass_ == pass); --cur_pass_; }
    bool is_visited(const size_t pass) const { assert(cur_pass_ <= pass); return cur_pass_ == pass; }

private:
    int kind_;
protected:
    std::vector<const Node*> ops_;
private:
    mutable size_t cur_pass_;

public:

    /// Just do what ever you want with this field.
    mutable std::string name;

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
        mutable size_t counter;
    };

    friend class World;
};

//------------------------------------------------------------------------------

template<class T, class U> inline
bool smart_eq(const T& t, const Node* other) { 
    if (const U* u = other->isa<U>()) 
        return smart_eq(t, u->as_tuple()); 
    return false;
}

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif // ANYDSL2_NODE_H
