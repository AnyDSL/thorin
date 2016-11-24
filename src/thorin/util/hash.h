#ifndef THORIN_UTIL_HASH_H
#define THORIN_UTIL_HASH_H

#include <algorithm>
#include <memory>
#include <utility>
#include <cassert>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace thorin {

//------------------------------------------------------------------------------

// currently no better place to fit this
/// Determines whether \p i is a power of two.
inline size_t is_power_of_2(size_t i) { return ((i != 0) && !(i & (i - 1))); }

//------------------------------------------------------------------------------

/// Magic numbers from http://www.isthe.com/chongo/tech/comp/fnv/index.html#FNV-param .
struct FNV1 {
    static const uint64_t offset = 14695981039346656037ull;
    static const uint64_t prime  = 1099511628211ull;
};

#define THORIN_SUPPORTED_HASH_TYPES \
    static_assert(std::is_signed<T>::value || std::is_unsigned<T>::value, \
            "please provide your own hash function; use hash_combine to create one");

/// Returns a new hash by combining the hash @p seed with @p val.
template<class T>
uint64_t hash_combine(uint64_t seed, T val) {
    THORIN_SUPPORTED_HASH_TYPES
    if (std::is_signed<T>::value)
        return hash_combine(seed, typename std::make_unsigned<T>::type(val));
    assert(std::is_unsigned<T>::value);
    for (uint64_t i = 0; i < sizeof(T); ++i) {
        T octet = val & T(0xff); // extract lower 8 bits
        seed ^= octet;
        seed *= FNV1::prime;
        val = val >> uint64_t(8);
    }
    return seed;
}

template<class T>
uint64_t hash_combine(uint64_t seed, T* val) { return hash_combine(seed, uintptr_t(val)); }

template<class T, class... Args>
uint64_t hash_combine(uint64_t seed, T val, Args... args) { return hash_combine(hash_combine(seed, val), args...); }

template<class T>
uint64_t hash_begin(T val) { return hash_combine(FNV1::offset, val); }
inline uint64_t hash_begin() { return FNV1::offset; }

template<class T>
struct Hash {
    uint64_t operator()(T val) const {
        THORIN_SUPPORTED_HASH_TYPES
        if (std::is_signed<T>::value)
            return Hash<typename std::make_unsigned<T>::type>()(val);
        assert(std::is_unsigned<T>::value);
        if (sizeof(uint64_t) >= sizeof(T))
            return val;
        return hash_begin(val);
    }
};

template<class T>
uint64_t hash_value(T val) { return Hash<T>()(val); }

template<class T>
struct Hash<T*> {
    //uint64_t operator()(T* val) const { return Hash<uintptr_t>()(uintptr_t(val)); }
    uint64_t operator()(T* val) const { return hash_begin(uintptr_t(val)); }
};

template<class T>
struct PtrSentinel {
    static_assert(std::is_pointer<T>::value, "must be a pointer");
    T operator()() const { return (T)(1); }
};

template<class T>
struct MaxSentinel {
    T operator()() const { return std::numeric_limits<T>::max(); }
};

//------------------------------------------------------------------------------

/// Used internally for @p HashSet and @p HashMap.
template<class Key, class T, class Sentinel, class Hasher, class KeyEqual>
class HashTable {
private:
    class Node {
    private:
        template<class Key_, class T_>
        struct get_key { static Key_& get(std::pair<Key_, T_>& pair) { return pair.first; } };

        template<class Key_>
        struct get_key<Key_, void> { static Key_& get(Key_& key) { return key; } };

        template<class Key_, class T_>
        struct get_value { static T_& get(std::pair<Key_, T_>& pair) { return pair.second; } };

        template<class Key_>
        struct get_value<Key_, void> { static Key_& get(Key_& key) { return key; } };

    public:
        typedef Key key_type;
        typedef typename std::conditional<std::is_void<T>::value, Key, T>::type mapped_type;
        typedef typename std::conditional<std::is_void<T>::value, Key, std::pair<Key, T>>::type value_type;

        Node()
            : value_() {
            key() = Sentinel()();
        }
        template<class... Args>
        Node(Args&&... args)
            : value_(std::forward<Args>(args)...)
        {}

        key_type& key() { return get_key<Key, T>::get(value_); }
        mapped_type& mapped() { return get_value<Key, T>::get(value_); }

        friend void swap(Node& n1, Node& n2) {
            using std::swap;
            swap(n1.value_, n2.value_);
        }

        bool is_invalid() { return key() == Sentinel()(); }

    private:
        value_type value_;

        friend class HashTable;
    };

    template<bool is_const>
    class iterator_base {
    public:
        typedef std::ptrdiff_t difference_type;
        typedef typename Node::value_type value_type;
        typedef typename std::conditional<is_const, const value_type&, value_type&>::type reference;
        typedef typename std::conditional<is_const, const value_type*, value_type*>::type pointer;
        typedef std::forward_iterator_tag iterator_category;


        iterator_base(Node* node, const HashTable* table)
            : node_(node)
            , table_(table)
#ifndef NDEBUG
            , id_(table->id())
#endif
        {}

        iterator_base(const iterator_base<false>& i)
            : node_(i.node_)
            , table_(i.table_)
#ifndef NDEBUG
            , id_(i.id_)
#endif
        {}

        iterator_base& operator=(iterator_base other) { swap(*this, other); return *this; }
        iterator_base& operator++() { assert(this->table_->id_ == this->id_); *this = skip(node_+1, table_); return *this; }
        iterator_base operator++(int) { assert(this->table_->id_ == this->id_); iterator_base res = *this; ++(*this); return res; }
        reference operator*() const { assert(this->table_->id_ == this->id_); return node_->value_; }
        pointer operator->() const { assert(this->table_->id_ == this->id_); return &node_->value_; }
        bool operator==(const iterator_base& other) { assert(this->table_ == other.table_ && this->id_ == other.id_ && this->table_->id_ == this->id_); return this->node_ == other.node_; }
        bool operator!=(const iterator_base& other) { assert(this->table_ == other.table_ && this->id_ == other.id_ && this->table_->id_ == this->id_); return this->node_ != other.node_; }
        friend void swap(iterator_base& i1, iterator_base& i2) {
            using std::swap;
            swap(i1.node_,  i2.node_);
            swap(i1.table_, i2.table_);
#ifndef NDEBUG
            swap(i1.id_,    i2.id_);
#endif
        }

    private:
        static iterator_base skip(Node* node, const HashTable* table) {
            while (node != table->end_node() && node->is_invalid())
                ++node;
            return iterator_base(node, table);
        }

        Node* node_;
        const HashTable* table_;
#ifndef NDEBUG
        int id_;
#endif
        friend class HashTable;
    };

public:
    typedef typename Node::key_type key_type;
    typedef typename Node::mapped_type mapped_type;
    typedef typename Node::value_type value_type;
    typedef std::size_t size_type;
    typedef Hasher hasher;
    typedef KeyEqual key_equal;
    typedef iterator_base<false> iterator;
    typedef iterator_base<true> const_iterator;
    enum {
        min_capacity = 16
    };

    HashTable()
        : capacity_(0)
        , size_(0)
        , nodes_(dummy_)
#ifndef NDEBUG
        , id_(0)
#endif
    {}

    HashTable(HashTable&& other)
        : HashTable()
    {
        swap(*this, other);
    }

    HashTable(const HashTable& other)
        : capacity_(other.capacity_)
        , size_(0)
        , nodes_(alloc())
#ifndef NDEBUG
        , id_(0)
#endif
    {
        insert(other.begin(), other.end());
    }

    template<class InputIt>
    HashTable(InputIt first, InputIt last)
        : HashTable(capacity)
    {
        insert(first, last);
    }

    HashTable(std::initializer_list<value_type> ilist)
        : HashTable(capacity)
    {
        insert(ilist);
    }

    ~HashTable() { destroy(); }

    // iterators
    iterator begin() { return iterator::skip(nodes_, this); }
    iterator end() { return iterator(end_node(), this); }
    const_iterator begin() const { return const_iterator(const_cast<HashTable*>(this)->begin()); }
    const_iterator end() const { return const_iterator(const_cast<HashTable*>(this)->end()); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    // getters
    size_t capacity() const { return capacity_; }
    size_t size() const { return size_; }
    bool empty() const { return size() == 0; }

    // emplace/insert
    template<class... Args>
    std::pair<iterator,bool> emplace(Args&&... args) {
        using std::swap;
#ifndef NDEBUG
        ++id_;
#endif
        auto n = Node(std::forward<Args>(args)...);
        auto& key = n.key();

        if (capacity_ == 0) {
            capacity_ = min_capacity;
            nodes_ = alloc();
        } else if (size_ > capacity_/size_t(4) + capacity_/size_t(2))
            rehash(capacity_*size_t(2));

        auto result = end_node();
        for (size_t i = desired_pos(key), distance = 0; true; i = mod(i+1), ++distance) {
            if (nodes_[i].is_invalid()) {
                ++size_;
                swap(nodes_[i], n);
                result = result == end_node() ? nodes_+i : result;
                return std::make_pair(iterator(result, this), true);
            } else if (result == end_node() && KeyEqual()(nodes_[i].key(), key)) {
                return std::make_pair(iterator(nodes_+i, this), false);
            } else {
                size_t cur_distance = probe_distance(i);
                if (cur_distance < distance) {
                    result = result == end_node() ? nodes_+i : result;
                    distance = cur_distance;
                    swap(nodes_[i], n);
                }
            }
        }
    }

    std::pair<iterator, bool> insert(const value_type& value) { return emplace(value); }
    std::pair<iterator, bool> insert(value_type&& value) { return emplace(std::move(value)); }
    void insert(std::initializer_list<value_type> ilist) { insert(ilist.begin(), ilist.end()); }

    template<class R>
    bool insert_range(const R& range) { return insert(range.begin(), range.end()); }

    template<class I>
    bool insert(I begin, I end) {
        bool changed = false;
        for (auto i = begin; i != end; ++i)
            changed |= insert(*i).second;
        return changed;
    }

    void erase(const_iterator pos) {
        using std::swap;

        assert(pos.table_ == this && "iterator does not match to this table");
        assert(pos.id_ == id_ && "iterator used after emplace/insert");
        assert(!empty());
        assert(pos != end() && !pos.node_->is_invalid());
        --size_;
        Node empty;
        swap(*pos.node_, empty);

        if (capacity_ != min_capacity && size_ < capacity_/size_t(4))
            rehash(capacity_/size_t(2));
        else {
            size_t curr = pos.node_-nodes_;
            size_t next = mod(curr+1);
            while (true) {
                if (nodes_[next].is_invalid() || probe_distance(next) == 0)
                    break;

                swap(nodes_[curr], nodes_[next]);
                curr = next;
                next = mod(next+1);
            };
        }
#ifndef NDEBUG
        ++id_;
#endif
    }

    void erase(const_iterator first, const_iterator last) {
        for (auto i = first; i != last; ++i)
            erase(i);
    }

    size_t erase(const key_type& key) {
        auto i = find(key);
        if (i == end())
            return 0;
        erase(i);
        return 1;
    }

    void clear() {
        destroy();
        size_ = 0;
        capacity_ = min_capacity;
        nodes_ = alloc();
    }

    iterator find(const key_type& key) {
        if (empty())
            return end();

        for (size_t i = desired_pos(key); true; i = mod(i+1)) {
            if (nodes_[i].is_invalid())
                return end();
            if (key_equal()(nodes_[i].key(), key))
                return iterator(nodes_+i, this);
        }
    }

    const_iterator find(const key_type& key) const { return const_iterator(const_cast<HashTable*>(this)->find(key).node_, this); }
    size_t count(const key_type& key) const { return find(key) == end() ? 0 : 1; }
    bool contains(const key_type& key) const { return count(key) == 1; }

    void rehash(size_t new_capacity) {
        using std::swap;

        auto old_capacity = capacity_;
        capacity_ = new_capacity;
        auto old_nodes = alloc();
        swap(old_nodes, nodes_);

        for (size_t i = 0; i != old_capacity; ++i) {
            auto& old = old_nodes[i];
            if (!old.is_invalid()) {
                for (size_t i = desired_pos(old.key()), distance = 0; true; i = mod(i+1), ++distance) {
                    if (nodes_[i].is_invalid()) {
                        swap(nodes_[i], old);
                        break;
                    } else {
                        size_t cur_distance = probe_distance(i);
                        if (cur_distance < distance) {
                            distance = cur_distance;
                            swap(nodes_[i], old);
                        }
                    }
                }
            }
        }

        if (old_nodes != dummy_)
            delete[] old_nodes;
    }

    friend void swap(HashTable& t1, HashTable& t2) {
        using std::swap;
        swap(t1.capacity_, t2.capacity_);
        swap(t1.size_,     t2.size_);

        if (t1.nodes_ == t1.dummy_) {
            if (t2.nodes_ == t2.dummy_) {
                // do nothing
            } else {
                t1.nodes_ = t2.nodes_;
                t2.nodes_ = t2.dummy_;
            }
        } else if (t2.nodes_ == t2.dummy_) {
            t2.nodes_ = t1.nodes_;
            t1.nodes_ = t1.dummy_;
        } else {
            swap(t1.nodes_,    t2.nodes_);
        }
#ifndef NDEBUG
        swap(t1.id_,       t2.id_);
#endif
    }

    HashTable& operator=(HashTable other) { swap(*this, other); return *this; }

private:
#ifndef NDEBUG
    int id() const { return id_; }
#endif
    size_t mod(size_t i) const { return i & (capacity_-1); }
    size_t desired_pos(const key_type& key) const { return mod(hasher()(key)); }
    size_t probe_distance(size_t i) { return mod(i + capacity() - desired_pos(nodes_[i].key())); }
    Node* end_node() const { return nodes_ + capacity(); }

    Node* alloc() {
        assert(capacity_ == 0 || is_power_of_2(capacity_));
        return new Node[capacity_]();
    }

    void destroy() {
        if (nodes_ != dummy_)
            delete[] nodes_;
    }

    uint32_t capacity_;
    uint32_t size_;
    Node* nodes_;
    Node dummy_[0];
#ifndef NDEBUG
    int id_;
#endif
};

//------------------------------------------------------------------------------

/**
 * This container is for the most part compatible with <tt>std::unordered_set</tt>.
 * We use our own implementation in order to have a consistent and deterministic behavior across different platforms.
 */
template<class Key, class Sentinel = PtrSentinel<Key>, class Hasher = Hash<Key>, class KeyEqual = std::equal_to<Key>>
class HashSet : public HashTable<Key, void, Sentinel, Hasher, KeyEqual> {
public:
    typedef Hasher hasher;
    typedef KeyEqual key_equal;
    typedef HashTable<Key, void, Sentinel, Hasher, KeyEqual> Super;
    typedef typename Super::key_type key_type;
    typedef typename Super::mapped_type mapped_type;
    typedef typename Super::value_type value_type;
    typedef typename Super::size_type size_type;
    typedef typename Super::iterator iterator;
    typedef typename Super::const_iterator const_iterator;

    HashSet() {}
    template<class InputIt>
    HashSet(InputIt first, InputIt last)
        : Super(first, last)
    {}
    HashSet(std::initializer_list<value_type> ilist)
        : Super(ilist)
    {}

    friend void swap(HashSet& s1, HashSet& s2) { swap(static_cast<Super&>(s1), static_cast<Super&>(s2)); }
};

//------------------------------------------------------------------------------

/**
 * This container is for the most part compatible with <tt>std::unordered_map</tt>.
 * We use our own implementation in order to have a consistent and deterministic behavior across different platforms.
 */
template<class Key, class T, class Sentinel = PtrSentinel<Key>, class Hasher = Hash<Key>, class KeyEqual = std::equal_to<Key>>
class HashMap : public HashTable<Key, T, Sentinel, Hasher, KeyEqual> {
public:
    typedef Hasher hasher;
    typedef KeyEqual key_equal;
    typedef HashTable<Key, T, Sentinel, Hasher, KeyEqual> Super;
    typedef typename Super::key_type key_type;
    typedef typename Super::mapped_type mapped_type;
    typedef typename Super::value_type value_type;
    typedef typename Super::size_type size_type;
    typedef typename Super::iterator iterator;
    typedef typename Super::const_iterator const_iterator;

    HashMap()
        : Super()
    {}
    template<class InputIt>
    HashMap(InputIt first, InputIt last, size_t capacity = Super::min_capacity)
        : Super(first, last)
    {}
    HashMap(std::initializer_list<value_type> ilist, size_t capacity = Super::min_capacity)
        : Super(ilist, capacity)
    {}

    mapped_type& operator[](const key_type& key) { return Super::insert(value_type(key, T())).first->second; }
    mapped_type& operator[](key_type&& key) { return Super::insert(value_type(std::move(key), T())).first->second; }

    friend void swap(HashMap& m1, HashMap& m2) { swap(static_cast<Super&>(m1), static_cast<Super&>(m2)); }
};

//------------------------------------------------------------------------------

template<class Key, class T, class Sentinel, class Hasher, class KeyEqual>
T* find(const HashMap<Key, T*, Sentinel, Hasher, KeyEqual>& map, const typename HashMap<Key, T*, Sentinel, Hasher, KeyEqual>::key_type& key) {
    auto i = map.find(key);
    return i == map.end() ? nullptr : i->second;
}

template<class Key, class Sentinel, class Hasher, class KeyEqual, class Arg>
bool visit(HashSet<Key, Sentinel, Hasher, KeyEqual>& set, const Arg& key) {
    return !set.insert(key).second;
}

//------------------------------------------------------------------------------

}

#endif
