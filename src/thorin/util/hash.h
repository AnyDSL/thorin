#ifndef THORIN_UTIL_HASH_H
#define THORIN_UTIL_HASH_H

#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "thorin/util/utility.h"

namespace thorin {

uint16_t fetch_gid();

//------------------------------------------------------------------------------

/// Magic numbers from http://www.isthe.com/chongo/tech/comp/fnv/index.html#FNV-param .
struct FNV1 {
    static const uint64_t offset = 14695981039346656037ull;
    static const uint64_t prime  = 1099511628211ull;
};

/// Returns a new hash by combining the hash @p seed with @p val.
template<class T>
uint64_t hash_combine(uint64_t seed, T val) {
    static_assert(std::is_signed<T>::value || std::is_unsigned<T>::value,
                  "please provide your own hash function");

    if (std::is_signed<T>::value)
        return hash_combine(seed, typename std::make_unsigned<T>::type(val));

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
uint64_t hash_combine(uint64_t seed, T val, Args&&... args) {
    return hash_combine(hash_combine(seed, val), std::forward<Args>(args)...);
}

template<class T>
uint64_t hash_begin(T val) { return hash_combine(FNV1::offset, val); }
inline uint64_t hash_begin() { return FNV1::offset; }

template<class T>
struct Hash {};

template<class T>
struct Hash<T*> {
    static uint64_t hash(T* ptr) { return uintptr_t(ptr) >> uintptr_t(log2(alignof(T))); }
    static bool eq(T* a, T* b) { return a == b; }
    static T* sentinel() { return (T*)(1); }
};

//------------------------------------------------------------------------------

/// Used internally for @p HashSet and @p HashMap.
template<class Key, class T, class H = Hash<Key>>
class HashTable {
public:
    typedef Key key_type;
    typedef typename std::conditional<std::is_void<T>::value, Key, T>::type mapped_type;
    typedef typename std::conditional<std::is_void<T>::value, Key, std::pair<Key, T>>::type value_type;

private:
    template<class K, class V>
    struct get_key { static K& get(std::pair<K, V>& pair) { return pair.first; } };

    template<class K>
    struct get_key<K, void> { static K& get(K& key) { return key; } };

    static key_type& key(value_type* ptr) { return get_key<Key, T>::get(*ptr); }
    static bool is_invalid(value_type* ptr) { return key(ptr) == H::sentinel(); }
    bool is_invalid(size_t i) { return is_invalid(nodes_+i); }

public:
    template<bool is_const>
    class iterator_base {
    public:
        typedef typename HashTable<Key, T, H>::value_type value_type;
        typedef std::ptrdiff_t difference_type;
        typedef typename std::conditional<is_const, const value_type&, value_type&>::type reference;
        typedef typename std::conditional<is_const, const value_type*, value_type*>::type pointer;
        typedef std::forward_iterator_tag iterator_category;

        iterator_base(value_type* ptr, const HashTable* table)
            : ptr_(ptr)
            , table_(table)
#ifndef NDEBUG
            , id_(table->id())
#endif
        {}

        iterator_base(const iterator_base<false>& i)
            : ptr_(i.ptr_)
            , table_(i.table_)
#ifndef NDEBUG
            , id_(i.id_)
#endif
        {}

        inline void verify() const { assert(table_->id_ == id_); }
        inline void verify(iterator_base i) const {
            assert(table_ == i.table_ && id_ == i.id_);
            verify();
        }

        iterator_base& operator=(const iterator_base& other) = default;
        iterator_base& operator++() { verify(); *this = skip(ptr_+1, table_); return *this; }
        iterator_base operator++(int) { verify(); iterator_base res = *this; ++(*this); return res; }
        reference operator*() const { verify(); return *ptr_; }
        pointer operator->() const { verify(); return ptr_; }
        bool operator==(const iterator_base& other) { verify(other); return this->ptr_ == other.ptr_; }
        bool operator!=(const iterator_base& other) { verify(other); return this->ptr_ != other.ptr_; }

    private:
        static iterator_base skip(value_type* ptr, const HashTable* table) {
            while (ptr != table->end_ptr() && is_invalid(ptr))
                ++ptr;
            return iterator_base(ptr, table);
        }

        value_type* ptr_;
        const HashTable* table_;
#ifndef NDEBUG
        int id_;
#endif
        friend class HashTable;
    };

    typedef std::size_t size_type;
    typedef iterator_base<false> iterator;
    typedef iterator_base<true> const_iterator;
    enum {
        StackCapacity = 8,
        MinCapacity = 16,
    };

    HashTable()
        : capacity_(StackCapacity)
        , size_(0)
        , gid_(fetch_gid())
        , nodes_(array_.data())
#ifndef NDEBUG
        , id_(0)
#endif
    {
        fill(nodes_);
    }

    HashTable(HashTable&& other)
        : HashTable()
    {
        swap(*this, other);
    }

    HashTable(const HashTable& other)
        : capacity_(other.capacity_)
        , size_(other.size_)
        , gid_(fetch_gid())
#ifndef NDEBUG
        , id_(0)
#endif
    {
        if (other.on_heap()) {
            nodes_ = alloc();
            std::copy(other.nodes_, other.nodes_+capacity_, nodes_);
        } else {
            nodes_ = array_.data();
            array_ = other.array_;
        }
    }

    template<class InputIt>
    HashTable(InputIt first, InputIt last)
        : HashTable()
    {
        insert(first, last);
    }

    HashTable(std::initializer_list<value_type> ilist)
        : HashTable()
    {
        insert(ilist);
    }

    ~HashTable() {
        if (on_heap())
            delete[] nodes_;
#ifndef NDEBUG
        assert(num_misses_ <= num_operations_ * 10 && "your hash function is garbage");
#endif
    }

    // iterators
    iterator begin() { return iterator::skip(nodes_, this); }
    iterator end() { return iterator(end_ptr(), this); }
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
        if (size_ > capacity_/size_t(4) + capacity_/size_t(2))
            rehash(capacity_*size_t(2));

        return emplace_no_rehash(std::forward<Args>(args)...);
    }

    // emplace/insert
    template<class... Args>
    std::pair<iterator,bool> emplace_no_rehash(Args&&... args) {
        using std::swap;
#ifndef NDEBUG
        ++id_;
        ++num_operations_;
#endif
        auto n = value_type(std::forward<Args>(args)...);
        auto& k = key(&n);

        auto result = end_ptr();
        for (size_t i = desired_pos(k), distance = 0; true; i = mod(i+1), ++distance) {
            if (is_invalid(i)) {
                ++size_;
                swap(nodes_[i], n);
                result = result == end_ptr() ? nodes_+i : result;
                return std::make_pair(iterator(result, this), true);
            } else if (result == end_ptr() && H::eq(key(nodes_+i), k)) {
                return std::make_pair(iterator(nodes_+i, this), false);
            } else {
#ifndef NDEBUG
                ++num_misses_;
#endif
                size_t cur_distance = probe_distance(i);
                if (cur_distance < distance) {
                    result = result == end_ptr() ? nodes_+i : result;
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
        size_t s = size() + std::distance(begin, end);
        size_t c = round_to_power_of_2(s);

        if (s > c/size_t(4) + c/size_t(2))
            c *= size_t(2);

        if (c != capacity_)
            rehash(c);

        bool changed = false;
        for (auto i = begin; i != end; ++i)
            changed |= emplace_no_rehash(*i).second;
        return changed;
    }

    void erase(const_iterator pos) {
        using std::swap;

        pos.verify();
        assert(pos.table_ == this && "iterator does not match to this table");
        assert(!empty());
        assert(pos != end() && !is_invalid(pos.ptr_));
        --size_;
        value_type empty;
        key(&empty) = H::sentinel();
        swap(*pos.ptr_, empty);

        if (capacity_ > MinCapacity && size_ < capacity_/size_t(4))
            rehash(capacity_/size_t(2));
        else {
            for (size_t curr = pos.ptr_-nodes_, next = mod(curr+1);
                !is_invalid(next) && probe_distance(next) != 0; curr = next, next = mod(next+1)) {
                swap(nodes_[curr], nodes_[next]);
#ifndef NDEBUG
                ++num_misses_;
#endif
            }
        }
#ifndef NDEBUG
        ++id_;
        ++num_operations_;
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
        size_ = 0;

        if (on_heap()) {
            delete[] nodes_;
            nodes_ = array_.data();
            capacity_ = StackCapacity;
        }

        fill(nodes_);
    }

    iterator find(const key_type& k) {
#ifndef NDEBUG
        ++num_operations_;
#endif
        if (empty())
            return end();

        for (size_t i = desired_pos(k); true; i = mod(i+1)) {
            if (is_invalid(i))
                return end();
            if (H::eq(key(nodes_+i), k))
                return iterator(nodes_+i, this);
#ifndef NDEBUG
            ++num_misses_;
#endif
        }
    }

    const_iterator find(const key_type& key) const {
        return const_iterator(const_cast<HashTable*>(this)->find(key).ptr_, this);
    }

    size_t count(const key_type& key) const { return find(key) == end() ? 0 : 1; }
    bool contains(const key_type& key) const { return count(key) == 1; }

    void rehash(size_t new_capacity) {
        using std::swap;

        assert(is_power_of_2(new_capacity));

        auto old_capacity = capacity_;
        capacity_ = new_capacity;
        auto old_nodes = alloc();
        swap(old_nodes, nodes_);

        for (size_t i = 0; i != old_capacity; ++i) {
            auto& old = old_nodes[i];
            if (!is_invalid(&old)) {
#ifndef NDEBUG
                ++num_operations_;
#endif
                for (size_t i = desired_pos(key(&old)), distance = 0; true; i = mod(i+1), ++distance) {
                    if (is_invalid(i)) {
                        swap(nodes_[i], old);
                        break;
                    } else {
#ifndef NDEBUG
                        ++num_misses_;
#endif
                        size_t cur_distance = probe_distance(i);
                        if (cur_distance < distance) {
                            distance = cur_distance;
                            swap(nodes_[i], old);
                        }
                    }
                }
            }
        }

        if (old_capacity != StackCapacity)
            delete[] old_nodes;
    }

    friend void swap(HashTable& t1, HashTable& t2) {
        using std::swap;

        if (t1.on_heap()) {
            if (t2.on_heap())
                swap(t1.nodes_, t2.nodes_);
            else {
                std::move(t2.array_.begin(), t2.array_.end(), t1.array_.begin());
                t2.nodes_ = t1.nodes_;
                t1.nodes_ = t1.array_.data();
            }
        } else {
            if (t2.on_heap()) {
                std::move(t1.array_.begin(), t1.array_.end(), t2.array_.begin());
                t1.nodes_ = t2.nodes_;
                t2.nodes_ = t2.array_.data();
            } else
                t1.array_.swap(t2.array_);
        }

        swap(t1.capacity_, t2.capacity_);
        swap(t1.size_,     t2.size_);
        swap(t1.gid_,      t2.gid_);
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
    size_t desired_pos(const key_type& key) const { return mod(hash_combine(H::hash(key), gid_)); }
    size_t probe_distance(size_t i) { return mod(i + capacity() - desired_pos(key(nodes_+i))); }
    value_type* end_ptr() const { return nodes_ + capacity(); }
    bool on_heap() const { return capacity_ != StackCapacity; }

    value_type* alloc() {
        assert(is_power_of_2(capacity_));
        auto nodes = new value_type[capacity_];
        return fill(nodes);
    }

    value_type* fill(value_type* nodes) {
        for (size_t i = 0, e = capacity_; i != e; ++i)
            key(nodes+i) = H::sentinel();
        return nodes;
    }

    size_t capacity_;
    size_t size_;
    uint16_t gid_;
    std::array<value_type, StackCapacity> array_;
    value_type* nodes_;
#ifndef NDEBUG
    int num_operations_ = 0;
    int num_misses_ = 0;
    int id_;
#endif
};

//------------------------------------------------------------------------------

/**
 * This container is for the most part compatible with <tt>std::unordered_set</tt>.
 * We use our own implementation in order to have a consistent and deterministic behavior across different platforms.
 */
template<class Key, class H = Hash<Key>>
class HashSet : public HashTable<Key, void, H> {
public:
    typedef HashTable<Key, void, H> Super;
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
template<class Key, class T, class H = Hash<Key>>
class HashMap : public HashTable<Key, T, H> {
public:
    typedef HashTable<Key, T, H> Super;
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
    HashMap(InputIt first, InputIt last)
        : Super(first, last)
    {}

    HashMap(std::initializer_list<value_type> ilist)
        : Super(ilist)
    {}

    mapped_type& operator[](const key_type& key) { return Super::insert(value_type(key, T())).first->second; }
    mapped_type& operator[](key_type&& key) {
        return Super::insert(value_type(std::move(key), T())).first->second;
    }

    friend void swap(HashMap& m1, HashMap& m2) { swap(static_cast<Super&>(m1), static_cast<Super&>(m2)); }
};

//------------------------------------------------------------------------------

template<class Key, class T, class H>
T* find(const HashMap<Key, T*, H>& map, const typename HashMap<Key, T*, H>::key_type& key) {
    auto i = map.find(key);
    return i == map.end() ? nullptr : i->second;
}

template<class Key, class H, class Arg>
bool visit(HashSet<Key, H>& set, const Arg& key) {
    return !set.emplace(key).second;
}

//------------------------------------------------------------------------------

}

#endif
