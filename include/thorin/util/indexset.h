#ifndef THORIN_UTIL_INDEXSET_H
#define THORIN_UTIL_INDEXSET_H

#include "thorin/util/array.h"

namespace thorin {

template<class Indexer, class Key>
class IndexSet {
public:
    class reference {
    private:
        reference(uint64_t& word, uint64_t pos)
            : word_(word)
            , pos_(pos)
        {}

    public:
        reference operator=(bool b) {
            if (b)
                word_ |= uint64_t(1) << pos_;
            else
                word_ &= ~(uint64_t(1) << pos_);
            return *this;
        }
        operator bool() const { return word_ & (uint64_t(1) << pos_); }

    private:
        uint64_t word() const { return word_; }

        uint64_t& word_;
        uint64_t pos_;

        friend class IndexSet;
    };

    // TODO write iterators
    // TODO add size

    IndexSet(const Indexer& indexer)
        : indexer_(indexer)
        , bits_((capacity()+63u) / 64u)
    {}
    IndexSet(IndexSet&& other)
        : IndexSet(indexer)
    {
        swap(*this, other);
    }
    IndexSet(const IndexSet& other)
        : indexer_(other.indexer())
        , bits_(other.bits_)
    {}

    const Indexer& indexer() const { return indexer_; }
    size_t capacity() const { return indexer().size(); }
    size_t next(size_t pos = 0) {
        for (size_t i = pos, e = capacity(); i != e; ++i) {
            if (bits_[i])
                return i;
        }
        return pos;
    }
    reference operator[](Key key) {
        auto i = indexer().index(key);
        assert(i != size_t(-1));
        return reference(bits_[i / 64u], i % 64u);
    }
    bool operator[](Key key) const { return (*const_cast<IndexSet<Indexer, Key>*>(this))[key]; }

    /// Depending on @p flag this method either inserts (true) or removes (false) @p key and returns true if successful.
    template<bool flag>
    bool set(Key key) {
        auto ref = (*this)[key];
        auto old = ref.word();
        ref = flag;
        return old != ref.word();
    }
    bool insert(Key key) { return set<true>(key); } ///< Inserts \p key and returns true if successful.
    bool erase(Key key) { return set<false>(key); } ///< Erase \p key and returns true if successful.
    bool contains(Key key) const { return (*this)[key]; }
    void clear() { std::fill(bits_.begin(), bits_.end(), 0u); }

    template<class Op>
    IndexSet& transform(const IndexSet& other, Op op) {
        assert(this->size() == other.size());
        for (size_t i = 0, e = capacity(); i != e; ++i)
            this->bits_[i] = op(this->bits_[i], other.bits_[i]);
        return *this;
    }
    IndexSet& operator&=(const IndexSet& other) { return transform(other, std::bit_and<uint64_t>()); }
    IndexSet& operator|=(const IndexSet& other) { return transform(other, std::bit_or <uint64_t>()); }
    IndexSet& operator^=(const IndexSet& other) { return transform(other, std::bit_xor<uint64_t>()); }
    IndexSet& operator =(IndexSet other) { swap(*this, other); return *this; }
    friend void swap(IndexSet& set1, IndexSet& set2) {
        using std::swap;
        assert(&set1.indexer() == &set2.indexer());
        swap(set1.bits_, set2.bits_);
    }

private:
    const Indexer& indexer_;
    Array<uint64_t> bits_;
};

template<class Indexer, class Key>
bool visit(IndexSet<Indexer, Key>& set, const Key& key) {
    return !set.insert(key);
}

template<class Indexer, class Key>
void visit_first(IndexSet<Indexer, Key>& set, const Key& key) {
    assert(!set.contains(key));
    visit(set, key);
}

}

#endif
