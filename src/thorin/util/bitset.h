#ifndef THORIN_UTIL_BITSET_H
#define THORIN_UTIL_BITSET_H

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace thorin {

class BitSet {
public:
    static const size_t npos = -1;

    class reference {
    public:
        reference(uint64_t& word, size_t pos)
            : word_(word)
            , pos_(pos)
        {}

        reference operator=(bool b) {
            if (b) 
                word_ |= 1 << pos_;
            else   
                word_ &= ~(1 << pos_);
            return *this;
        }
        operator bool() const { return word_ & ( 1 << pos_); }

    private:
        uint64_t& word_;
        size_t pos_;
    };

    BitSet(size_t size = 0)
        : size_(size)
        , bits_((size+63u) / 64u)
    {}
    BitSet(BitSet&& other)
        : BitSet()
    {
        swap(*this, other);
    }
    BitSet(const BitSet& other)
        : size_(other.size_)
        , bits_(other.bits_)
    {}

    size_t size() const { return size_; }
    size_t next(size_t pos = 0) {
        for (size_t i = pos, e = size(); i != e; ++i) {
            if (bits_[i])
                return i;
        }
        return pos;
    }
    reference operator[] (size_t i) {
        assert(i < size_ && "out of bounds access");
        return reference(bits_[i / 64u], i % 64u);
    }
    BitSet& operator |= (const BitSet& other) {
        assert(this->size() == other.size());
        for (size_t i = 0, e = size(); i != e; ++i)
            this->bits_[i] |= other.bits_[i];
        return *this;
    }
    BitSet& operator &= (const BitSet& other) {
        assert(this->size() == other.size());
        for (size_t i = 0, e = size(); i != e; ++i)
            this->bits_[i] &= other.bits_[i];
        return *this;
    }
    BitSet& operator ^= (const BitSet& other) {
        assert(this->size() == other.size());
        for (size_t i = 0, e = size(); i != e; ++i)
            this->bits_[i] |= other.bits_[i];
        return *this;
    }
    BitSet& operator = (BitSet other) { swap(*this, other); return *this; }
    friend void swap(BitSet set1, BitSet set2) {
        using std::swap;
        swap(set1.size_, set2.size_);
        swap(set1.bits_, set2.bits_);
    }

private:
    size_t size_;
    std::vector<uint64_t> bits_;
};

}

#endif
