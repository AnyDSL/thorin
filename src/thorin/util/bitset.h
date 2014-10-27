#ifndef THORIN_UTIL_BITSET_H
#define THORIN_UTIL_BITSET_H

#include <cassert>
#include <cstdint>
#include <vector>

namespace thorin {

class BitSet {
private:
    size_t size_;
    std::vector<uint64_t> set_;

public:
    struct reference {
    private:
        uint64_t& word_;
        size_t pos_;
    public:
        reference(uint64_t& word, size_t pos) : word_(word), pos_(pos) {}

        reference operator=(bool b) {
            if (b) word_ |= 1 << pos_;
            else   word_ &= ~(1 << pos_);
            return *this;
        }
        operator bool() const { return word_ & ( 1 << pos_); }
    };

    BitSet(size_t size) : size_(size), set_(size / 64u + 1u) {}

    reference operator[](size_t i) {
        assert(i < size_ && "out of bounds access");
        return reference(set_[i / 64u], i % 64u);
    }
};

}

#endif
