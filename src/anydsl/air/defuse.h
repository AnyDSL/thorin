#ifndef ANYDSL_AIR_DEF_H
#define ANYDSL_AIR_DEF_H

#include <cstring>
#include <string>

#include <boost/cstdint.hpp>
#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>

#include "anydsl/air/airnode.h"

namespace anydsl {

//------------------------------------------------------------------------------

class Lambda;
class Type;
class World;
class Use;
typedef boost::unordered_set<Use*> UseSet;
class Def;
class Jump;
class World;

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

    World& world();

private:

    Def* def_;
    AIRNode* parent_;

    friend class Ops;
};

//------------------------------------------------------------------------------

/**
 * Circular doubly linked list of Use instances.
 *
 * This class is supposed to be embedded in a Terminator.
 * Ops already have enough encapsulation magic. 
 * No need to hammer further getters/setters around an Ops aggregate within a class.
 * Just make it a public class member.
 *
 * Iterators stay valid even if you put the whole list upside down 
 * and remove or insert many items.
 * Only erasing a node at all invalidates an interator.
 */
class Ops  {
private:

    /// Do not copy-create a \p Ops instance.
    Ops(const Ops&);
    /// Do not copy-assign a \p Ops instance.
    Ops& operator = (const Ops&);

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
        typedef Use value_type;
        typedef ptrdiff_t difference_type;
        typedef Use* pointer;
        typedef Use& reference;

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

        node_iter<T, !reverse> switch_direction() const { return node_iter<T, !reverse>(n_); }

    private:

        T n_;
        friend class Ops;
    };

public:

    typedef node_iter<Node*, false> iterator;
    typedef node_iter<const Node*, false> const_iterator;
    typedef node_iter<Node*, true> reverse_iterator;
    typedef node_iter<const Node*, true> const_reverse_iterator;

    /// Create empty argument list.
    Ops();
    ~Ops();

    iterator append(Def* parent, Def* def)  { return insert(end(), parent, def); }
    iterator prepend(Def* parent, Def* def) { return insert(begin(), parent, def); }

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
    iterator insert(iterator pos, Def* parent, Def* def);

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

private:

    Node* head() { return sentinel_->next_; }
    const Node* head() const { return sentinel_->next_; }
    Node* tail() { return sentinel_->prev_; }
    const Node* tail() const { return sentinel_->prev_; }

    Node* sentinel_;
    size_t size_;
};

//------------------------------------------------------------------------------

class Def : public AIRNode {
private:

    /// Do not copy-create a \p Def instance.
    Def(const Def&);
    /// Do not copy-assign a \p Def instance.
    Def& operator = (const Def&);

    void registerUse(Use* use);
    void unregisterUse(Use* use);

protected:

    Def(IndexKind index, const Type* type)
        : AIRNode(index) 
        , type_(type)
    {}

public:

    ~Def() { anydsl_assert(uses_.empty(), "there are still uses pointing to this def"); }

    const UseSet& uses() const { return uses_; }
    const Type* type() const { return type_; }
    World& world() const;
    const Ops& ops() const { return ops_; }

    Ops::iterator ops_append(Def* def)  { return ops_.insert(ops_.end(), this, def); }
    Ops::iterator ops_prepend(Def* def) { return ops_.insert(ops_.begin(), this, def); }

protected:

    void setType(const Type* type) { type_ = type; }

    Ops ops_;

private:

    const Type* type_;
    UseSet uses_;

    friend Use::Use(AIRNode*, Def*);
    friend Use::~Use();
};

//------------------------------------------------------------------------------

class Param : public Def {
private:

    Param(Lambda* parent, const Type* type)
        : Def(Index_Param, type)
        , parent_(parent)
    {}

    const Lambda* parent() const { return parent_; }

private:

    Lambda* parent_;

    friend class Lambda;
};

//------------------------------------------------------------------------------

class Value : public Def {
protected:

    Value(IndexKind index, const Type* type)
        : Def(index, type)
    {}
};

struct ValueNumber {
private:

    /// Do not copy-assign ValueNumber%s.
    ValueNumber& operator = (const ValueNumber& vn);

public:
    IndexKind index;
    uintptr_t op1;


    union {
        uintptr_t op2;
        size_t size;
    };

    union {
        uintptr_t op3;
        uintptr_t* more;
    };

    ValueNumber() {}
    explicit ValueNumber(IndexKind index)
        : index(index)
        , op1(0)
        , op2(0)
        , op3(0)
    {}
    ValueNumber(IndexKind index, uintptr_t p1)
        : index(index)
        , op1(p1)
        , op2(0)
        , op3(0)
    {}
    ValueNumber(IndexKind index, uintptr_t p1, uintptr_t p2) 
        : index(index)
        , op1(p1)
        , op2(p2)
        , op3(0)
    {}
    ValueNumber(IndexKind index, uintptr_t p1, uintptr_t p2, uintptr_t p3)
        : index(index)
        , op1(p1)
        , op2(p2)
        , op3(p3)
    {}
    ValueNumber(const ValueNumber& vn) {
        std::memcpy(this, &vn, sizeof(ValueNumber));
        if (hasMore(index)) {
            more = new uintptr_t[size];
            std::memcpy(more, vn.more, sizeof(uintptr_t) * size);
        }
    }
#if (__cplusplus >= 201103L)
    ValueNumber(ValueNumber&& vn) {
        std::memcpy(this, &vn, sizeof(ValueNumber));
        if (hasMore(index)) 
            vn.more = 0;
    }
#endif
    ~ValueNumber() {
        if (hasMore(index))
            delete[] more;
    }

    /**
     * Creates a ValueNumber where the number of built-in fields do not suffice.
     * Memory allocation and deallocation is handled by this class. 
     * However, the caller is responsible to fill the allocated fields
     * (pointed to by \p more) with correct data.
     */
    static ValueNumber createMore(IndexKind index, size_t size) {
        ValueNumber res(index);
        res.size = size;
        res.more = new uintptr_t[size];
        return res;
    }

    bool operator == (const ValueNumber& vn) const;

    static bool hasMore(IndexKind kind) {
        switch (kind) {
            case Index_Tuple:
            case Index_Pi:
            case Index_Sigma: 
                return true;
            default:
                return false;
        }
    }
};

size_t hash_value(const ValueNumber& vn);

//------------------------------------------------------------------------------

} // namespace anydsl

#endif // ANYDSL_AIR_DEF_H
