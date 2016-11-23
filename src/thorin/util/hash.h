#ifndef THORIN_UTIL_HASH_H
#define THORIN_UTIL_HASH_H

#include <algorithm>
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
    uint64_t operator()(T* val) const { return Hash<uintptr_t>()(uintptr_t(val)); }
    //uint64_t operator()(T* val) const { return hash_begin(uintptr_t(val)); }
};

//------------------------------------------------------------------------------

/// Used internally for @p HashSet and @p HashMap.
template<class Key, class T, class Hasher, class KeyEqual>
class HashTable {
private:
    class Node {
    private:
        template<class Key_, class T_>
        struct get_key { static const Key_& get(const std::pair<Key_, T_>& pair) { return pair.first; } };

        template<class Key_>
        struct get_key<Key_, void> { static const Key_& get(const Key_& key) { return key; } };

        template<class Key_, class T_>
        struct get_value { static const T_& get(const std::pair<Key_, T_>& pair) { return pair.second; } };

        template<class Key_>
        struct get_value<Key_, void> { static const Key_& get(const Key_& key) { return key; } };

    public:
        typedef Key key_type;
        typedef typename std::conditional<std::is_void<T>::value, Key, T>::type mapped_type;
        typedef typename std::conditional<std::is_void<T>::value, Key, std::pair<Key, T>>::type value_type;

        Node() {}
        template<class... Args>
        Node(Args&&... args)
            : value_(args...)
        {}

        const key_type& key() const { return get_key<Key, T>::get(value_); }
        const mapped_type& mapped() const { return get_value<Key, T>::get(value_); }

    private:
        value_type value_;

        friend class HashTable;
    };

    static bool is_valid(Node** p) { return *p != nullptr; }

    template<bool is_const>
    class iterator_base {
    public:
        typedef std::ptrdiff_t difference_type;
        typedef typename Node::value_type value_type;
        typedef typename std::conditional<is_const, const value_type&, value_type&>::type reference;
        typedef typename std::conditional<is_const, const value_type*, value_type*>::type pointer;
        typedef std::forward_iterator_tag iterator_category;


#ifndef NDEBUG
        iterator_base(Node** node, Node** end, const HashTable* table)
            : node_(node)
            , end_(end)
            , table_(table)
            , id_(table->id())
#else
        iterator_base(Node** node, Node** end, const HashTable*)
            : node_(node)
            , end_(end)
#endif
        {}

        iterator_base(const iterator_base<false>& i)
            : node_(i.node_)
            , end_(i.end_)
#ifndef NDEBUG
            , table_(i.table_)
            , id_(i.id_)
#endif
        {}

        iterator_base& operator=(iterator_base other) { swap(*this, other); return *this; }
        iterator_base& operator++() { assert(this->table_->id_ == this->id_); *this = skip(node_+1, end_, table_); return *this; }
        iterator_base operator++(int) { assert(this->table_->id_ == this->id_); iterator_base res = *this; ++(*this); return res; }
        reference operator*() const { assert(this->table_->id_ == this->id_); return (*node_)->value_; }
        pointer operator->() const { assert(this->table_->id_ == this->id_); return &(*node_)->value_; }
        bool operator==(const iterator_base& other) { assert(this->table_ == other.table_ && this->id_ == other.id_ && this->table_->id_ == this->id_); return this->node_ == other.node_; }
        bool operator!=(const iterator_base& other) { assert(this->table_ == other.table_ && this->id_ == other.id_ && this->table_->id_ == this->id_); return this->node_ != other.node_; }
        friend void swap(iterator_base& i1, iterator_base& i2) {
            using std::swap;
            swap(i1.node_,  i2.node_);
#ifndef NDEBUG
            swap(i1.table_, i2.table_);
            swap(i1.id_,    i2.id_);
#endif
        }

    private:
        static iterator_base skip(Node** node, Node** end, const HashTable* table) {
            while (node != end && !is_valid(node))
                ++node;
            return iterator_base(node, end, table);
        }

        Node** node_;
        Node** end_;
#ifndef NDEBUG
        const HashTable* table_;
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
        : capacity_(min_capacity)
        , size_(0)
        , nodes_(alloc())
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
    iterator begin() { return iterator::skip(nodes_, end_node(), this); }
    iterator end() { return iterator(end_node(), end_node(), this); }
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
#ifndef NDEBUG
        ++id_;
#endif
        auto n = new Node(args...);
        auto& key = n->key();

        rehash<true>();
        auto result = end();
        for (size_t i = desired_pos(key), distance = 0; true; i = mod(i+1), ++distance) {
            auto it = nodes_ + i;
            if (*it == nullptr) {
                ++size_;
                *it = n;
                if (result == end())
                    result.node_ = it;
                return std::make_pair(result, true);
            } else if (result == end() && KeyEqual()((*it)->key(), key)) {
                delete n;
                return std::make_pair(iterator(it, end_node(), this), false);
            } else {
                // if the existing elem has probed less than us, then swap places with existing elem, and keep going to find another slot for that elem
                size_t cur_distance = probe_distance(i);
                if (cur_distance < distance) {
                    if (result == end())
                        result.node_ = it;
                    distance = cur_distance;
                    std::swap(*it, n);
                }
            }
        }
    }
    std::pair<iterator, bool> insert(const value_type& value) { return emplace(value); }
    std::pair<iterator, bool> insert(value_type&& value) { return emplace(value); }
    template<class I>
    bool insert(I begin, I end) {
        bool changed = false;
        for (auto i = begin; i != end; ++i)
            changed |= insert(*i).second;
        return changed;
    }
    void insert(std::initializer_list<value_type> ilist) { insert(ilist.begin(), ilist.end()); }
    template<class R> bool insert_range(const R& range) { return insert(range.begin(), range.end()); }

    iterator erase(const_iterator pos) {
        assert(pos.table_ == this && "iterator does not match to this table");
        assert(pos.id_ == id_ && "iterator used after emplace/insert");
        assert(!empty());
        assert(pos != end() && is_valid(pos.node_));
        --size_;
        delete *pos.node_;
        *pos.node_ = nullptr;

        if (!rehash<false>()) {

            size_t curr = pos.node_-nodes_;
            size_t next = mod(curr+1);
            while (true) {
                if (nodes_[next] == nullptr || probe_distance(next) == 0)
                    break;

                std::swap(*(nodes_+curr), *(nodes_+next));
                curr = next;
                next = mod(next+1);
            };
        }
#ifndef NDEBUG
        ++id_;
#endif
        return iterator(pos.node_, end_node(), this);
    }
    iterator erase(const_iterator first, const_iterator last) {
        assert(false && "TODO: currently broken");
        for (auto i = first; i != last; ++i)
            erase(i);
        return last;
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
        for (size_t i = desired_pos(key); true; i = mod(i+1)) {
            auto it = nodes_ + i;
            if (*it == nullptr)
                return end();
            if (key_equal()((*it)->key(), key))
                return iterator(it, end_node(), this);
        }
    }
    const_iterator find(const key_type& key) const { return const_iterator(const_cast<HashTable*>(this)->find(key).node_, end_node(), this); }
    size_t count(const key_type& key) const { return find(key) == end() ? 0 : 1; }
    bool contains(const key_type& key) const { return count(key) == 1; }

    template<bool enlarge>
    bool rehash() {
        size_t c4 = capacity_/size_t(4);
        size_t c2 = capacity_/size_t(2);
        size_t old_capacity = capacity_;
        size_t new_capacity = capacity_;

        if (!enlarge) {
            if (size_ < c4) {
                new_capacity = std::max(size_t(min_capacity), c2);
                if (new_capacity != old_capacity)
                    goto do_rehash;
            }
            return false;
        }

        if (size_ > c4 + c2)
            new_capacity = capacity_*size_t(2);
        else
            return false;

do_rehash:
        capacity_ = new_capacity;
        auto nodes = alloc();

        auto mod = [&](size_t i) { return i & (new_capacity-1); };
        auto desired_pos = [&](const key_type& key) { return mod(hasher()(key)); };
        auto probe_distance = [&](size_t i) { return mod(i + new_capacity - desired_pos(nodes[i]->key())); };

        for (size_t i = 0; i != old_capacity; ++i) {
            if (is_valid(nodes_+i)) {
                auto old = nodes_[i];
                for (size_t i = desired_pos(old->key()), distance = 0; true; i = mod(i+1), ++distance) {
                    auto it = nodes + i;
                    if (*it == nullptr) {
                        nodes[i] = old;
                        break;
                    } else {
                        size_t cur_distance = probe_distance(i);
                        if (cur_distance < distance) {
                            distance = cur_distance;
                            std::swap(*it, old);
                        }
                    }
                }
            }
        }

        std::swap(nodes, nodes_);
        delete[] nodes;

        return true;
    }

    // copy/move stuff
    friend void swap(HashTable& table1, HashTable& table2) {
        using std::swap;
        swap(table1.capacity_,      table2.capacity_);
        swap(table1.size_,          table2.size_);
        swap(table1.nodes_,         table2.nodes_);
#ifndef NDEBUG
        swap(table1.id_,            table2.id_);
#endif
    }
    HashTable& operator=(HashTable other) { swap(*this, other); return *this; }

private:
#ifndef NDEBUG
    int id() const { return id_; }
#endif
    size_t mod(size_t i) const { return i & (capacity_-1); }
    size_t desired_pos(const key_type& key) const { return mod(hasher()(key)); }
    size_t probe_distance(size_t i) { return mod(i + capacity() - desired_pos(nodes_[i]->key())); }
    Node** end_node() const { return nodes_ + capacity(); }
    void destroy() {
        for (size_t i = 0, e = capacity_; i != e; ++i) {
            if (is_valid(nodes_+i))
                delete nodes_[i];
        }
        delete[] nodes_;
    }
    Node** alloc() {
        assert(is_power_of_2(capacity_));
        auto nodes = new Node*[capacity_]();
        return nodes;
    }

    size_t capacity_;
    size_t size_;
    Node** nodes_;
#ifndef NDEBUG
    int id_;
#endif
};

//------------------------------------------------------------------------------

/**
 * This container is for the most part compatible with <tt>std::unordered_set</tt>.
 * We use our own implementation in order to have a consistent and deterministic behavior across different platforms.
 */
template<class Key, class Hasher = Hash<Key>, class KeyEqual = std::equal_to<Key>>
class HashSet : public HashTable<Key, void, Hasher, KeyEqual> {
public:
    typedef Hasher hasher;
    typedef KeyEqual key_equal;
    typedef HashTable<Key, void, Hasher, KeyEqual> Super;
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
template<class Key, class T, class Hasher = Hash<Key>, class KeyEqual = std::equal_to<Key>>
class HashMap : public HashTable<Key, T, Hasher, KeyEqual> {
public:
    typedef Hasher hasher;
    typedef KeyEqual key_equal;
    typedef HashTable<Key, T, Hasher, KeyEqual> Super;
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

template<class Key, class T, class Hasher, class KeyEqual>
T* find(const HashMap<Key, T*, Hasher, KeyEqual>& map, const typename HashMap<Key, T*, Hasher, KeyEqual>::key_type& key) {
    auto i = map.find(key);
    return i == map.end() ? nullptr : i->second;
}

template<class Key, class T, class Hasher, class KeyEqual>
T*& retrieve(HashMap<Key, T*, Hasher, KeyEqual>& map, const typename HashMap<Key, T*, Hasher, KeyEqual>::key_type& key) {
    return map.emplace(key, nullptr).first->second;
}

template<class Key, class Hasher, class KeyEqual, class Arg>
bool visit(HashSet<Key, Hasher, KeyEqual>& set, const Arg& key) {
    return !set.insert(key).second;
}

//------------------------------------------------------------------------------

}

#endif
