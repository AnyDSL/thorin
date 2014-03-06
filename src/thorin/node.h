// TODO remove this stuff once new impala front-end is done

#ifndef THORIN_NODE_H
#define THORIN_NODE_H

#include <cassert>
#include <vector>

#include "thorin/enums.h"
#include "thorin/util/array.h"
#include "thorin/util/cast.h"

namespace thorin {

template<class Base>
class Node : public MagicCast<Base> {
private:
    Node& operator = (const Node&); ///< Do not copy-assign a \p Node instance.
    Node(const Node& node);         ///< Do not copy-construct a \p Node.

protected:
    Node(int kind, size_t size, const std::string& name)
        : kind_(kind)
        , ops_(size)
        , cur_pass_(0)
        , name(name)
    {}
    virtual ~Node() {}

    void set(size_t i, const Node* n) { ops_[i] = n; }
    void resize(size_t n) { ops_.resize(n, 0); }

public:
    int kind() const { return kind_; }
    bool is_corenode() const { return ::thorin::is_corenode(kind()); }
    NodeKind node_kind() const { assert(is_corenode()); return (NodeKind) kind_; }
    template<class T>
    ArrayRef<T> ops_ref() const { return size() ? ArrayRef<T>((T*) &ops_.front(), ops_.size()) : ArrayRef<T>(); }
    size_t size() const { return ops_.size(); }
    bool empty() const { return ops_.empty(); }
    virtual size_t hash() const {
        size_t seed = hash_combine(hash_value(kind()), size());
        for (auto op : ops_)
            seed = hash_combine(seed, op);
        return seed;
    }
    virtual bool equal(const Node* other) const {
        bool result = this->kind() == other->kind() && this->size() == other->size();
        for (size_t i = 0, e = size(); result && i != e; ++i)
            result &= this->ops_[i] == other->ops_[i];
        return result;
    }

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
    mutable std::string name; ///< Just do what ever you want with this field.

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
    };
    union {
        mutable bool flags[sizeof(size_t)/sizeof(bool)];
        mutable size_t counter;
    };

    friend class World;
};

//------------------------------------------------------------------------------

}

#endif
