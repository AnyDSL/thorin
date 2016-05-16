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

    static Node* tombstone()       { return (Node*) -1; }
    static Node* end_pointer()     { return (Node*)  1; }
    static bool is_end(Node** p)   { return *p == end_pointer(); }
    static bool is_valid(Node** p) { return *p != nullptr && *p != tombstone(); }

    template<bool is_const>
    class iterator_base {
    public:
        typedef std::ptrdiff_t difference_type;
        typedef typename Node::value_type value_type;
        typedef typename std::conditional<is_const, const value_type&, value_type&>::type reference;
        typedef typename std::conditional<is_const, const value_type*, value_type*>::type pointer;
        typedef std::forward_iterator_tag iterator_category;


#ifndef NDEBUG
        iterator_base(Node** node, const HashTable* table)
            : node_(node)
            , table_(table)
            , id_(table->id())
#else
        iterator_base(Node** node, const HashTable*)
            : node_(node)
#endif
        {}

        iterator_base(const iterator_base<false>& i)
            : node_(i.node_)
#ifndef NDEBUG
            , table_(i.table_)
            , id_(i.id_)
#endif
        {}

        iterator_base& operator=(iterator_base other) { swap(*this, other); return *this; }
        iterator_base& operator++() { assert(this->table_->id_ == this->id_); node_ = move_to_valid(++node_); return *this; }
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
        static Node** move_to_valid(Node** n) {
            while (!is_valid(n) && !is_end(n)) ++n;
            return n;
        }

        Node** node_;
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
    static const size_t min_capacity = 16;

    HashTable(size_type capacity = min_capacity, const hasher& hash_function = hasher(), const key_equal& key_eq = key_equal())
        : capacity_(std::max(size_type(min_capacity), capacity))
        , load_(0)
        , size_(0)
        , nodes_(alloc())
        , hash_function_(hash_function)
        , key_eq_(key_eq)
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
        , load_(0)
        , size_(0)
        , nodes_(alloc())
        , hash_function_(other.hash_function_)
        , key_eq_(other.key_eq_)
#ifndef NDEBUG
        , id_(0)
#endif
        {
            insert(other.begin(), other.end());
        }
    template<class InputIt>
    HashTable(InputIt first, InputIt last, size_type capacity = min_capacity, const hasher& hash_function = hasher(), const key_equal& key_eq = key_equal())
        : HashTable(capacity, hash_function, key_eq)
    {
        insert(first, last);
    }
    HashTable(std::initializer_list<value_type> ilist, size_type capacity = min_capacity, const hasher& hash_function = hasher(), const key_equal& key_eq = key_equal())
        : HashTable(capacity, hash_function, key_eq)
    {
        insert(ilist);
    }
    ~HashTable() { destroy(); }

    // iterators
    iterator begin() { return iterator(iterator::move_to_valid(nodes_), this); }
    iterator end() { auto n = nodes_ + capacity(); assert(is_end(n)); return iterator(n, this); }
    const_iterator begin() const { return const_iterator(const_cast<HashTable*>(this)->begin()); }
    const_iterator end() const { return const_iterator(const_cast<HashTable*>(this)->end()); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    // getters
    hasher hash_function() const { return hash_function_; }
    key_equal key_eq() const { return key_eq_; }
    size_type capacity() const { return capacity_; }
    size_type load() const { return load_; }
    size_type size() const { return size_; }
    bool empty() const { return size() == 0; }

    // emplace/insert
    template<class... Args>
    std::pair<iterator,bool> emplace(Args&&... args) {
#ifndef NDEBUG
        ++id_;
#endif
        auto n = new Node(args...);
        auto c4 = capacity_/size_t(4), c2 = capacity_/size_t(2);
        if (size_ < c4)
            rehash(c2);
        else if (size_ > c4 + c2)
            rehash(capacity_*size_t(2));
        else if (load_ > c4 + c2)
            rehash(capacity_);  // free garbage (remove all tombstones)

        Node** insert_pos = nullptr;
        auto& key = n->key();
        for (uint64_t i = hash_function_(key), step = 0; true; i += ++step) {
            size_t x = i & (capacity_-1);
            auto it = nodes_ + x;
            if (*it == nullptr) {
                if (insert_pos == nullptr) {
                    insert_pos = it;
                    ++load_;
                }
                ++size_;
                *insert_pos = n;
                return std::make_pair(iterator(insert_pos, this), true);
            } else if (*it == tombstone()) {
                if (insert_pos == nullptr)
                    insert_pos = it;
            } else if (key_eq_((*it)->key(), key)) {
                delete n;
                return std::make_pair(iterator(it, this), false);
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

    // erase
    iterator erase(const_iterator pos) {
        assert(pos.table_ == this && "iterator does not match to this table");
        assert(pos.id_ == id_ && "iterator used after emplace/insert");
        assert(!empty());
        assert(is_valid(pos.node_) && pos != end());
        --size_;
        delete *pos.node_;
        *pos.node_ = tombstone();
        return iterator(iterator::move_to_valid(pos.node_), this);
    }
    iterator erase(const_iterator first, const_iterator last) {
        for (auto i = first; i != last; ++i)
            erase(i);
        return last;
    }
    size_type erase(const key_type& key) {
        auto i = find(key);
        if (i == end())
            return 0;
        erase(i);
        return 1;
    }
    void clear() {
        destroy();
        size_ = 0;
        load_ = 0;
        capacity_ = min_capacity;
        nodes_ = alloc();
    }

    // find
    iterator find(const key_type& key) {
#ifndef NDEBUG
        int old_id = id_;
#endif
        for (uint64_t i = hash_function_(key), step = 0; true; i += ++step) {
            size_t x = i & (capacity_-1);
            auto it = nodes_ + x;
            if (*it == nullptr) {
                assert(old_id == id());
                return end();
            } else if (*it != tombstone() && key_eq_((*it)->key(), key)) {
                assert(old_id == id());
                return iterator(it, this);
            }
        }
    }
    const_iterator find(const key_type& key) const { return const_iterator(const_cast<HashTable*>(this)->find(key).node_, this); }
    size_type count(const key_type& key) const { return find(key) == end() ? 0 : 1; }
    bool contains(const key_type& key) const { return count(key) == 1; }

    void rehash(size_type new_capacity) {
        new_capacity = std::max(size_type(min_capacity), new_capacity);
        size_t old_capacity = capacity_;
        capacity_ = new_capacity;
        auto nodes = alloc();
        load_ = size_;

        for (size_t i = 0; i != old_capacity; ++i) {
            if (is_valid(nodes_+i)) {
                Node* old = nodes_[i];
                for (uint64_t i = hash_function_(old->key()), step = 0; true; i += ++step) {
                    size_t x = i & (capacity_-1);
                    if (nodes[x] == nullptr) {
                        nodes[x] = old;
                        break;
                    }
                }
            }
        }

        std::swap(nodes, nodes_);
        delete[] nodes;
    }

    // copy/move stuff
    friend void swap(HashTable& table1, HashTable& table2) {
        using std::swap;
        swap(table1.capacity_,      table2.capacity_);
        swap(table1.size_,          table2.size_);
        swap(table1.nodes_,         table2.nodes_);
        swap(table1.hash_function_, table2.hash_function_);
        swap(table1.key_eq_,        table2.key_eq_);
#ifndef NDEBUG
        swap(table1.id_,            table2.id_);
#endif
    }
    HashTable& operator=(HashTable other) { swap(*this, other); return *this; }

private:
#ifndef NDEBUG
    int id() const { return id_; }
#endif
    void destroy() {
        for (size_t i = 0, e = capacity_; i != e; ++i) {
            if (is_valid(nodes_+i))
                delete nodes_[i];
        }
        delete[] nodes_;
    }
    Node** alloc() {
        assert(is_power_of_2(capacity_));
        auto nodes = new Node*[capacity_+1](); // the last node serves as end
        nodes[capacity_] = end_pointer();      // mark end as occupied
        return nodes;
    }

    size_type capacity_;
    size_type load_;
    size_type size_;
    Node** nodes_;
    hasher hash_function_;
    key_equal key_eq_;
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

    HashSet(size_type capacity = Super::min_capacity, const hasher& hash_function = hasher(), const key_equal& key_eq = key_equal())
        : Super(capacity, hash_function, key_eq)
    {}
    template<class InputIt>
    HashSet(InputIt first, InputIt last, size_type capacity = Super::min_capacity, const hasher& hash_function = hasher(), const key_equal& key_eq = key_equal())
        : Super(first, last, capacity, hash_function, key_eq)
    {}
    HashSet(std::initializer_list<value_type> ilist, size_type capacity = Super::min_capacity, const hasher& hash_function = hasher(), const key_equal& key_eq = key_equal())
        : Super(ilist, capacity, hash_function, key_eq)
    {}
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

    HashMap(size_type capacity = Super::min_capacity, const hasher& hash_function = hasher(), const key_equal& key_eq = key_equal())
        : Super(capacity, hash_function, key_eq)
    {}
    template<class InputIt>
    HashMap(InputIt first, InputIt last, size_type capacity = Super::min_capacity, const hasher& hash_function = hasher(), const key_equal& key_eq = key_equal())
        : Super(first, last, capacity, hash_function, key_eq)
    {}
    HashMap(std::initializer_list<value_type> ilist, size_type capacity = Super::min_capacity, const hasher& hash_function = hasher(), const key_equal& key_eq = key_equal())
        : Super(ilist, capacity, hash_function, key_eq)
    {}

    mapped_type& operator[](const key_type& key) { return Super::insert(value_type(key, T())).first->second; }
    mapped_type& operator[](key_type&& key) { return Super::insert(value_type(std::move(key), T())).first->second; }
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
