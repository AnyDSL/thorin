#ifndef ANYDSL_AIR_USE_H
#define ANYDSL_AIR_USE_H

#include "anydsl/air/airnode.h"

namespace anydsl {

class Def;
class Jump;

/**
 * Use encapsulates a use of an SSA value, i.e., a \p Def.
 *
 * This class is supposed to be embedded in other \p AIRNode%s.
 * \p Use already has enough encapsulation magic. 
 * No need to hammer further getters/setters around a Use aggregate within a class.
 * Just make it a public class member.
 */
class Use : public AIRNode {
private:

    /// Do not copy-create a \p Use instance.
    Use(const Use&);
    /// Do not copy-assign a \p Use instance.
    Use& operator = (const Use&);

public:

    /** 
     * @brief Construct a \p Use of the specified \p Def.
     * 
     * @param parent The class where \p Use is embedded in.
     * @param def 
     */
    Use(AIRNode* parent, Def* def);
    virtual ~Use();

    /// Get the definition \p Def of this \p Use.
    Def* def() { return def_; }
    const Def* def() const { return def_; }

    /// Get embedding ojbect.
    AIRNode* parent() { return parent_; }
    /// Get embedding ojbect.
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
 * This class is supposed to be embedded in a Terminator.
 * Args already have enough encapsulation magic. 
 * No need to hammer further getters/setters around an Args aggregate within a class.
 * Just make it a public class member.
 *
 * Iterators stay valid even if you put the whole list upside down 
 * and remove or insert many items.
 * Only erasing a node at all invalidates an interator.
 *
 * Iterating using the FOREACH macro already yields a reference. 
 * In other words do it like this:
 * \code
 * FOREACH(use, args) // type of use is const Use&
 * \endcode
 * This won't even compile:
 * \code
 * FOREACH(& use, args) // type of use would be const Use&&
 * \endcode
 */
class Args  {
private:

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
        UseNode(AIRNode* parent, Def* def)
            : use_(parent, def)
        {
#ifndef NDEBUG
            isSentinel_ = false;
#endif
        }

        Use use_;
    };


    /// This class implements four iterators; all combinations of const, non-const, reverse and non-reverse.
    template<class T, bool reverse>
    struct node_iter {
    public:

        /// typedefs are necessary for std::iterator_traits (needed by FOREACH)
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
        friend class Args;
    };

public:

    typedef node_iter<Node*, false> iterator;
    typedef node_iter<const Node*, false> const_iterator;
    typedef node_iter<Node*, true> reverse_iterator;
    typedef node_iter<const Node*, true> const_reverse_iterator;

    /// Create empty argument list.
    Args(Jump* parent);
    ~Args();

    iterator append(Def* def)  { return insert(end(), def); }
    iterator prepend(Def* def) { return insert(begin(), def); }

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
     * 
     * @return iterator to new Node.
     */
    iterator insert(iterator pos, Def* def);

    /** 
     * Erase node at \p pos.
     * The associated Use is also deleted (and thus unregistered from its Def).
     *
     * @return Returns iterator to the node previously followed by \p pos.
     */
    iterator erase(iterator pos);

    /** 
     * Removes all elements from the list.
     * All \p Use%s are destroyed in this process (and thus are unregistered from their Def).
     */
    void clear();

    /// Get Jump where this argument list is embedded in.
    Jump* parent() { return parent_; }
    /// Get Jump where this argument list is embedded in.
    const Jump* parent() const { return parent_; }

private:

    Node* head() { return sentinel_->next_; }
    const Node* head() const { return sentinel_->next_; }
    Node* tail() { return sentinel_->prev_; }
    const Node* tail() const { return sentinel_->prev_; }

    Jump* parent_;
    Node* sentinel_;
    size_t size_;
};

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_USE_H
