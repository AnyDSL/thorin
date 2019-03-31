#include "thorin/util/bitset.h"
#include "thorin/util/stream.h"

namespace thorin {

void BitSet::dealloc() const {
    if (num_words_ != 1)
        delete[] words_;
}

size_t BitSet::count() const {
    size_t result = 0;
    auto w = words();
    for (size_t i = 0, e = num_words(); i != e; ++i)
        result += bitcount(w[i]);
    return result;
}

inline static uint64_t begin_mask(uint64_t i) { return -1_u64 << (i % 64_u64); }
inline static uint64_t   end_mask(uint64_t i) { return ~begin_mask(i); }

bool BitSet::any_range(const size_t begin, size_t end) const {
    if (begin >= end)
        return false;

    end = std::min(end, num_bits());
    size_t i = begin / 64_s;
    size_t e =   end / 64_s;
    auto bmask = begin_mask(begin);
    auto emask =   end_mask(  end);

    // if i and e are within the same word
    if (i == e) return words()[i] & bmask & emask;

    // use bmask for the first word
    bool result = (words()[i++] & bmask);

    // all other words except the last one
    for (; !result && i != e; ++i)
        result |= words()[i];

    // only use emask if there actually *is* an emask - otherwise we are getting out of bounds
    return result || (emask && (words_[i] & emask));
}

BitSet& BitSet::operator>>=(uint64_t shift) {
    uint64_t div = shift/64_u64;
    uint64_t rem = shift%64_u64;
    auto w = words();

    if (div >= num_words())
        clear();
    else {
        for (size_t i = 0, e = num_words()-div; i != e; ++i)
            w[i] = w[i+div];
        std::fill(w+num_words()-div, w+num_words(), 0);

        uint64_t carry = 0;
        for (size_t i = num_words()-div; i-- != 0;) {
            uint64_t new_carry = w[i] << (64_u64-rem);
            w[i] = (w[i] >> rem) | carry;
            carry = new_carry;
        }
    }

    return *this;
}

void BitSet::ensure_capacity(size_t i) const {
    size_t num_new_words = (i+64_s) / 64_s;
    if (num_new_words > num_words_) {
        num_new_words = round_to_power_of_2(num_new_words);
        assert(num_new_words >= num_words_ * 2_s
                && "num_new_words must be a power of two at least twice of num_words_");
        uint64_t* new_words = new uint64_t[num_new_words];

        // copy over and fill rest with zero
        std::fill(std::copy_n(words(), num_words_, new_words), new_words + num_new_words, 0);

        // record new num_words and words_ pointer
        dealloc();
        num_words_ = num_new_words;
        words_ = new_words;
    }
}

}
