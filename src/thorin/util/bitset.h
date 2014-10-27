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
            if (b) word_ |= 1 << pos_;
            else   word_ &= ~(1 << pos_);
            return *this;
        }
        operator bool() const { return word_ & ( 1 << pos_); }

    private:
        uint64_t& word_;
        size_t pos_;
    };

    BitSet(size_t size)
        : size_(size)
        , bits_(size / 64u + 1u)
    {}

    size_t size() const { return size_; }
    reference operator[] (size_t i) {
        assert(i < size_ && "out of bounds access");
        return reference(bits_[i / 64u], i % 64u);
    }
    size_t next(size_t pos = 0) {
        for (size_t i = pos, e = size(); i != e; ++i) {
            if (bits_[i])
                return i;
        }
        return pos;
    }

private:
    size_t size_;
    std::vector<uint64_t> bits_;
};

}

#endif
