#ifndef THORIN_UTIL_HASH_H
#define THORIN_UTIL_HASH_H

#include <cassert>
#include <cstdint>
#include <iostream>
#include <functional>
#include <limits>
#include <type_traits>

namespace thorin {

template<class T> 
inline size_t hash_value(const T& t) { return std::hash<T>()(t); }
template<class T>
inline size_t hash_combine(size_t seed, const T& val) { return seed ^ (hash_value(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2)); }

inline size_t is_power_of_2(size_t i) { return ((i != 0) && !(i & (i - 1))); }

//------------------------------------------------------------------------------

#if 0
// magic numbers from http://www.isthe.com/chongo/tech/comp/fnv/index.html#FNV-param
template<class T> struct FNV1 {};
template<> struct FNV1<uint32_t> {
    static const uint32_t offset = 2166136261u;
    static const uint32_t prime  = 16777619u;
    static const int size = 32;
};

template<> struct FNV1<uint64_t> {
    static const uint64_t offset = 14695981039346656037ull;
    static const uint64_t prime  = 1099511628211ull;
    static const int size = 64;
};

template<class T>
size_t hash_combine(T data) {
    static_assert(std::is_unsigned<T>::value, "must be unsigned integer type");
    static_assert(std::numeric_limits<T>::digits == sizeof(T)*8, "something weird goes on here... please review");
    size_t hash = FNV1<size_t>::offset;
    for (int i = 0; i < std::numeric_limits<T>::digits/8; ++i) {
        T octet = data & T(0xff); // extract lower 8 bits
        hash ^= octet;
        hash *= FNV1<size_t>::prime;
        data = data >> size_t(8);
    }

    return hash;
}

inline size_t hash_combine( int8_t seed) { return hash_combine( uint8_t(seed)); }
inline size_t hash_combine(int16_t seed) { return hash_combine(uint16_t(seed)); }
inline size_t hash_combine(int32_t seed) { return hash_combine(uint32_t(seed)); }
inline size_t hash_combine(int64_t seed) { return hash_combine(uint64_t(seed)); }
#endif

template <class T>
inline std::size_t hash_value_signed(T val) {
    const int size_t_bits = std::numeric_limits<std::size_t>::digits;
    // ceiling(std::numeric_limits<T>::digits / size_t_bits) - 1
    const int length = (std::numeric_limits<T>::digits - 1) / size_t_bits;
    std::size_t seed = 0;
    T positive = val < 0 ? -1 - val : val;

    // hopefully, this loop can be unrolled.
    for(unsigned int i = length * size_t_bits; i > 0; i -= size_t_bits)
        seed ^= (std::size_t) (positive >> i) + (seed<<6) + (seed>>2);
    seed ^= (std::size_t) val + (seed<<6) + (seed>>2);
    return seed;
}

template <class T>
inline std::size_t hash_value_unsigned(T val) {
    const int size_t_bits = std::numeric_limits<std::size_t>::digits;
    // ceiling(std::numeric_limits<T>::digits / size_t_bits) - 1
    const int length = (std::numeric_limits<T>::digits - 1) / size_t_bits;
    std::size_t seed = 0;

    // Hopefully, this loop can be unrolled.
    for(unsigned int i = length * size_t_bits; i > 0; i -= size_t_bits)
        seed ^= (std::size_t) (val >> i) + (seed<<6) + (seed>>2);
    seed ^= (std::size_t) val + (seed<<6) + (seed>>2);
    return seed;
}

template<class Key> struct Hash;
template<> struct Hash<  int8_t> { size_t operator () (  int8_t i) const { return hash_value_signed(i); } };
template<> struct Hash< int16_t> { size_t operator () ( int16_t i) const { return hash_value_signed(i); } };
template<> struct Hash< int32_t> { size_t operator () ( int32_t i) const { return hash_value_signed(i); } };
template<> struct Hash< int64_t> { size_t operator () ( int64_t i) const { return hash_value_signed(i); } };
template<> struct Hash< uint8_t> { size_t operator () ( uint8_t i) const { return hash_value_unsigned(i); } };
template<> struct Hash<uint16_t> { size_t operator () (uint16_t i) const { return hash_value_unsigned(i); } };
template<> struct Hash<uint32_t> { size_t operator () (uint32_t i) const { return hash_value_unsigned(i); } };
template<> struct Hash<uint64_t> { size_t operator () (uint64_t i) const { return hash_value_unsigned(i); } };
template<class T> struct Hash<T*> { size_t operator () (T* p) const { return hash_value_signed(std::intptr_t(p)); } };

//------------------------------------------------------------------------------

#define IS_VALID(p) (*(p) != nullptr && *(p) != ((HashNode*)-1))
#define IS_END(p)   (*(p) == ((HashNode*)1))

template<class Key, class T, class Hasher, class KeyEqual>
class HashTable {
private:
    class HashNode {
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

        HashNode() {}
        template<class... Args>
        HashNode(Args&&... args)
            : value_(args...)
        {}

        const key_type& key() const { return get_key<Key, T>::get(value_); }
        const mapped_type& mapped() const { return get_value<Key, T>::get(value_); }

    private:
        value_type value_;

        friend class HashTable;
    };

    template<bool is_const>
    class iterator_base {
    public:
        typedef std::ptrdiff_t difference_type;
        typedef typename HashNode::value_type value_type;
        typedef typename std::conditional<is_const, const value_type&, value_type&>::type reference;
        typedef typename std::conditional<is_const, const value_type*, value_type*>::type pointer;
        typedef std::bidirectional_iterator_tag iterator_category;

        iterator_base(HashNode** node, const HashTable* table) 
            : node_(node)
#ifndef NDEBUG
            , table_(table) 
            , id_(table->id())
#endif
        {}
        iterator_base(const iterator_base<false>& i) 
            : node_(i.node_)
#ifndef NDEBUG
            , table_(i.table_) 
            , id_(i.id_)
#endif
        {}

        iterator_base& operator=(const iterator_base& i) { node_ = i.node_; table_ = i.table_; return *this; }
        iterator_base& operator++() { node_ = move_to_valid(++node_); return *this; }
        iterator_base operator++(int) { iterator_base res = *this; ++(*this); return res; }
        reference operator*() const { return (*node_)->value_; }
        pointer operator->() const { return &(*node_)->value_; }
        bool operator == (const iterator_base& other) { assert(this->table_ == other.table_ && this->id_ == other.id_); return this->node_ == other.node_; }
        bool operator != (const iterator_base& other) { assert(this->table_ == other.table_ && this->id_ == other.id_); return this->node_ != other.node_; }

    private:
        static HashNode** move_to_valid(HashNode** n) {
            while (!IS_VALID(n) && !IS_END(n)) ++n;
            return n; 
        }
        HashNode** node_;
#ifndef NDEBUG
        const HashTable* table_;
        int id_;
#endif

        friend class HashTable;
    };

public:
    static const size_t min_capacity = 16;
    typedef typename HashNode::key_type key_type;
    typedef typename HashNode::mapped_type mapped_type;
    typedef typename HashNode::value_type value_type;
    typedef std::size_t size_type;
    typedef Hasher hasher;
    typedef KeyEqual key_equal;
    typedef iterator_base<false> iterator;
    typedef iterator_base<true> const_iterator;

    HashTable(size_type capacity = min_capacity, const hasher& hash_function = hasher(), const key_equal& key_eq = key_equal())
        : capacity_(std::max(size_type(min_capacity), capacity))
        , size_(0)
        , nodes_(alloc())
        , hash_function_(hash_function)
        , key_eq_(key_eq)
#ifndef NDEBUG
        , id_(0)
#endif
    {}
    HashTable(const HashTable&) = delete;
    HashTable(HashTable&& other)
        : capacity_(std::move(other.capacity_))
        , size_(std::move(other.size_))
        , nodes_(std::move(other.nodes_))
        , hash_function_(std::move(other.hash_function_))
        , key_eq_(std::move(other.key_eq_))
#ifndef NDEBUG
        , id_(std::move(other.id_))
#endif
    {
        swap(*this, other);
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
    iterator end() { auto n = nodes_ + capacity(); assert(IS_END(n)); return iterator(n, this); }
    const_iterator begin() const { return const_iterator(const_cast<HashTable*>(this)->begin()); }
    const_iterator end() const { return const_iterator(const_cast<HashTable*>(this)->end()); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    // getters
    hasher hash_function() const { return hash_function_; }
    key_equal key_eq() const { return key_eq_; }
    size_type size() const { return size_; }
    size_type capacity() const { return capacity_; }
    bool empty() const { return size() == 0; }

    // emplace/insert
    template<class... Args>
    std::pair<iterator,bool> emplace(Args&&... args) { 
#ifndef NDEBUG
        ++id_;
#endif
        auto n = new HashNode(args...);
        if (size_ > capacity_/size_t(4) + capacity_/size_t(2))
            rehash(capacity_*size_t(2));
        else if (capacity_ > min_capacity && size_ < capacity_/size_t(4))
            rehash(capacity_/size_t(4));

        auto& key = n->key();
        for (size_t i = hash_function_(key), step = 0; true; i = (i + step++)) {
            size_t x = i & (capacity_-1);
            auto it = nodes_ + x;
            if (*it == nullptr || *it == (HashNode*)-1) {
                ++size_;
                *it = n;
                return std::make_pair(iterator(it, this), true);
            } else if (key_eq_(key, (*it)->key())) {
                delete n;
                return std::make_pair(iterator(it, this), false);
            }
        }
    }
    std::pair<iterator, bool> insert(const value_type& value) { return emplace(value); }
    std::pair<iterator, bool> insert(value_type&& value) { return emplace(value); }
    template<class InputIt>
    void insert(InputIt first, InputIt last) { 
        for (auto i = first; i != last; ++i) 
            insert(*i); 
    }
    void insert(std::initializer_list<value_type> ilist) { insert(ilist.begin(), ilist.end()); }

    // erase
    iterator erase(const_iterator pos) {
        assert(pos.table_ == this && "iterator does not match to this table");
        assert(pos.id_ == id_ && "iterator used after emplace/insert");
        assert(!empty());
        assert(IS_VALID(pos.node_) && pos != end());
        --size_;
        delete *pos.node_;
        *pos.node_ = (HashNode*)-1;
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
        capacity_ = min_capacity;
        nodes_ = alloc();
    }

    // find
    iterator find(const key_type& key) {
        for (size_t i = hash_function_(key), step = 0; true; i = (i + step++)) {
            size_t x = i & (capacity_-1);
            auto it = nodes_ + x;
            if (*it == nullptr)
                return end();
            else if (*it != (HashNode*)-1 && key_eq_(key, (*it)->key()))
                return iterator(it, this);
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

        for (size_t i = 0; i != old_capacity; ++i) {
            if (IS_VALID(nodes_+i)) {
                HashNode* old = nodes_[i];
                for (size_t i = hash_function_(old->key()), step = 0; true; i = (i + step++)) {
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
    HashTable& operator= (HashTable other) { std::swap(*this, other); return *this; }

protected:
#ifndef NDEBUG
    int id() const { return id_; }
#else
    int id() const { return 0; }
#endif
    void destroy() {
        for (size_t i = 0, e = capacity_; i != e; ++i) {
            if (IS_VALID(nodes_+i))
                delete nodes_[i];
        }
        delete[] nodes_;
    }
    HashNode** alloc() {
        assert(is_power_of_2(capacity_));
        auto nodes = new HashNode*[capacity_+1](); // the last node servers as end
        nodes[capacity_] = (HashNode*)1;           // mark end as occupied
        return nodes;
    }

    size_type capacity_;
    size_type size_;
    HashNode** nodes_;
    hasher hash_function_;
    key_equal key_eq_;
#ifndef NDEBUG
    int id_;
#endif
};

//------------------------------------------------------------------------------

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

    mapped_type& operator[] (const key_type& key) { return Super::insert(value_type(key, T())).first->second; }
    mapped_type& operator[] (key_type&& key) { return Super::insert(value_type(std::move(key), T())).first->second; }
};

template<class Key, class T, class Hasher, class KeyEqual>
T* find(const HashMap<Key, T*, Hasher, KeyEqual>& map, const typename HashMap<Key, T*, Hasher, KeyEqual>::key_type& key) {
    auto i = map.find(key);
    return i == map.end() ? nullptr : i->second;
}

template<class Key, class Hasher, class KeyEqual, class Arg> 
bool visit(HashSet<Key, Hasher, KeyEqual>& set, const Arg& key) { 
    return !set.insert(key).second; 
}

template<class Key, class Hasher, class KeyEqual, class Arg> 
void visit_first(HashSet<Key, Hasher, KeyEqual>& set, const Arg& key) { 
    assert(!set.contains(key)); 
    visit(set, key); 
}

//------------------------------------------------------------------------------

}

#endif
