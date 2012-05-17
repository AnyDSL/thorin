#ifndef ANYDSL_AIR_USE_H
#define ANYDSL_AIR_USE_H

#include "anydsl/air/airnode.h"
#include "anydsl/util/autoptr.h"

namespace anydsl {

class Def;

class Use : public AIRNode {
private:

    /// Do not create "default" \p Use instances
    Use();
    /// Do not copy-create a \p Use instance.
    Use(const Use&);
    /// Do not copy-assign a \p Use instance.
    Use& operator = (const Use&);

public:

    Use(Def* def, AIRNode* parent, const std::string& debug = "");
    virtual ~Use();

    const Def* def() const { return def_; }

    AIRNode* parent() { return parent_; }
    const AIRNode* parent() const { return parent_; }

private:

    Def* def_;
    AIRNode* parent_;

    friend class Args;
};

//------------------------------------------------------------------------------

/**
 * Circular doubly linked list of Use instances.
 *
 * NOTE: iterating using the FOREACH macro already yields a reference. 
 * In other words do it like this:
 *
 * FOREACH(use, args) // type of use is const Use&
 *
 * This won't even compile:
 *
 * FOREACH(& use, args) // type of use would be const Use&&
 */
class Args  {
private:

    /// Do not create "default" \p Args instances
    Args();
    /// Do not copy-create a \p Args instance.
    Args(const Args&);
    /// Do not copy-assign a \p Args instance.
    Args& operator = (const Args&);

    struct Node {
        Node* next_;
        Node* prev_;

#ifndef NDEBUG
        Node() : isSentinel_(true) {}
        bool isSentinel_;
#endif
    };

    struct UseNode : public Node {
        UseNode(Def* def, AIRNode* parent, const std::string& debug)
            : use_(def, parent, debug)
        {
#ifndef NDEBUG
            isSentinel_ = false;
#endif
        }

        Use use_;
    };

    template<class T, bool reverse>
    struct node_iter {
    public:

        typedef std::bidirectional_iterator_tag iterator_category;
        typedef const Use value_type;
        typedef ptrdiff_t difference_type;
        typedef const Use* pointer;
        typedef const Use& reference;

        node_iter() {}
        explicit node_iter(T n) : n_(n) {}
        template<class U> node_iter(node_iter<U, reverse> const& i) : n_(i.n_) {}

        const Use& operator * () const { assert(!n_->isSentinel_); return ((UseNode*) n_)->use_; }
        const Use* operator ->() const { assert(!n_->isSentinel_); return &((UseNode*) n_)->use_; }

        template<class U> bool operator == (const node_iter<U, reverse>& i) const { return n_ == i.n_; }
        template<class U> bool operator != (const node_iter<U, reverse>& i) const { return n_ != i.n_; }

        node_iter& operator ++() { n_ = reverse ? n_->prev_ : n_->next_; return *this; }
        node_iter& operator --() { n_ = reverse ? n_->next_ : n_->prev_; return *this; }
        node_iter operator ++(int) { node_iter<T, reverse> i(n_); n_ = n_->next_; return i; }
        node_iter operator --(int) { node_iter<T, reverse> i(n_); n_ = n_->next_; return i; }

    private:

        T n_;

        template <class, bool> friend class node_iter;
        friend class Args;
    };

public:

    typedef node_iter<Node*, false> iterator;
    typedef node_iter<const Node*, false> const_iterator;
    typedef node_iter<Node*, true> reverse_iterator;
    typedef node_iter<const Node*, true> const_reverse_iterator;

    /// Create empty argument list.
    Args(AIRNode* parent);
    ~Args();

    void append(Def* def, const std::string& debug = "") { insert(end(), def, debug); }
    void prepend(Def* def, const std::string& debug = "") { insert(begin(), def, debug); }

    iterator begin() { return iterator(head()); }
    iterator end() { return iterator(sentinel_); }
    const_iterator begin() const { return const_iterator(head()); }
    const_iterator end() const { return const_iterator(sentinel_); }

    reverse_iterator rbegin() { return reverse_iterator(tail()); }
    reverse_iterator rend() { return reverse_iterator(sentinel_); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(tail()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(sentinel_); }

    const Use& front() { return *begin(); }
    const Use& back() { return *rbegin(); }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    /** 
     * @brief Insert before given \p pos.
     * 
     * @param pos Insertion happens before this position.
     * @param def Create new Use of this Def.
     * @param debug Debug info for new Use.
     * 
     * @return iterator to new Node.
     */
    iterator insert(iterator pos, Def* def, const std::string& debug = "");

    /** 
     * Erase node at \p pos.
     * @return Returns interator to the node previously followed by \p pos.
     */
    iterator erase(iterator pos);

    /// Removes all elements from the list.
    void clear();

private:

    Node* head() { return sentinel_->next_; }
    const Node* head() const { return sentinel_->next_; }
    Node* tail() { return sentinel_->prev_; }
    const Node* tail() const { return sentinel_->prev_; }

    AIRNode* parent_;
    Node* sentinel_;
    size_t size_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_USE_H
