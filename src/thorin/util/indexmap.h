#ifndef THORIN_UTIL_INDEXMAP_H
#define THORIN_UTIL_INDEXMAP_H

#include "thorin/util/array.h"
#include "thorin/util/iterator.h"

namespace thorin {

template<class Indexer, class Key, class Value>
class IndexMap {
private:
    template<class T>
    struct IsValidPred {
        static bool is_valid(T) { return true; }
    };

    template<class T>
    struct IsValidPred<T*> {
        static bool is_valid(T* value) { return value != nullptr; }
    };

public:
    IndexMap(IndexMap&& other)
        : indexer_(std::move(other.indexer_))
        , array_(std::move(other.array_))
    {}
    IndexMap(const IndexMap& other)
        : indexer_(other.indexer_)
        , array_(other.array_)
    {}
    IndexMap(const Indexer& indexer, const Value& value = Value())
        : indexer_(indexer)
        , array_(indexer.size(), value)
    {}
    IndexMap(const Indexer& indexer, ArrayRef<Value> array)
        : indexer_(indexer)
        , array_(array)
    {}
    template<class I>
    IndexMap(const Indexer& indexer, const I begin, const I end)
        : indexer_(indexer)
        , array_(begin, end)
    {}

    const Indexer& indexer() const { return indexer_; }
    size_t capacity() const { return array_.size(); }
    Value& operator[](Key key) { auto i = indexer().index(key); assert(i != size_t(-1)); return array_[i]; }
    const Value& operator[](Key key) const { return const_cast<IndexMap*>(this)->operator[](key); }
    Array<Value>& array() { return array_; }
    const Array<Value>& array() const { return array_; }
    Value& array(size_t i) { return array_[i]; }
    const Value& array(size_t i) const { return array_[i]; }

    typedef filter_iterator<typename Array<Value>::const_iterator, bool (*)(Value)> const_iterator;
    const_iterator begin() const { return make_filter(array_.begin(), array_.end(), IsValidPred<Value>::is_valid); }
    const_iterator end() const { return make_filter(array_.end(), array_.end(), IsValidPred<Value>::is_valid); }

    friend void swap(IndexMap& map1, IndexMap& map2) {
        using std::swap;
        swap(map1.indexer_, map2.indexer_);
        swap(map1.array_, map2.array_);
    }

private:
    const Indexer& indexer_;
    Array<Value> array_;
};

template<class Indexer, class Key, class Value>
inline Value* find(IndexMap<Indexer, Key, Value*>& map, Key key) {
    auto i = map.indexer().index(key);
    return i != size_t(-1) ? map.array()[i] : nullptr;
}

template<class Indexer, class Key, class Value>
inline const Value* find(const IndexMap<Indexer, Key, Value*>& map, Key key) {
    return find(const_cast<IndexMap<Indexer, Key, Value*>&>(map), key);
}

}

#endif
