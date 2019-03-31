#ifndef THORIN_UTIL_BITSET_H
#define THORIN_UTIL_BITSET_H

#include <algorithm>

#include "thorin/util/utility.h"

namespace thorin {

class BitSet {
public:
    class reference {
    private:
        reference(uint64_t* word, uint16_t index)
            : tagged_ptr_(word, index)
        {}

    public:
        reference operator=(bool b) {
            uint64_t mask = 1_u64 << index();
            if (b)
                word() |= mask;
            else
                word() &= ~mask;
            return *this;
        }
        operator bool() const { return word() & (1_u64 << index()); }

    private:
        const uint64_t& word() const { return *tagged_ptr_.ptr(); }
        uint64_t& word() { return *tagged_ptr_.ptr(); }
        uint64_t index() const { return tagged_ptr_.index(); }

        TaggedPtr<uint64_t> tagged_ptr_;

        friend class BitSet;
    };

    BitSet()
        : word_(0)
        , num_words_(1)
    {}
    BitSet(const BitSet& other)
        : BitSet()
    {
        ensure_capacity(other.num_bits()-1);
        std::copy_n(other.words(), other.num_words(), words());
    }
    BitSet(BitSet&& other)
        : words_(std::move(other.words_))
        , num_words_(std::move(other.num_words_))
    {
        other.words_ = nullptr;
    }
    ~BitSet() { dealloc(); }
    /// @name get, set, clear, toggle, and test bits
    //@{
    bool test(size_t i) const {
        if ((i/64_s) >= num_words())
            return false;
        return *(words() + i/64_s) & (1_u64 << i%64_u64);
    }

    BitSet& set   (size_t i) { ensure_capacity(i); *(words() + i/64_s) |=  (1_u64 << i%64_u64); return *this; }
    BitSet& toggle(size_t i) { ensure_capacity(i); *(words() + i/64_s) ^=  (1_u64 << i%64_u64); return *this; }
    BitSet& clear (size_t i) { ensure_capacity(i); *(words() + i/64_s) &= ~(1_u64 << i%64_u64); return *this; }
    /// clears all bits
    void clear() { dealloc(); num_words_ = 1; word_ = 0; }

    reference operator[](size_t i) { ensure_capacity(i); return reference(words() + i/64_s, i%64_u64); }
    bool operator[](size_t i) const { return (*const_cast<BitSet*>(this))[i]; }
    //@}

    /// @name any
    /// Is any bit range set?
    //@{
    /// Is any bit in @c [begin,end[ set?
    bool any_range(const size_t begin, const size_t end) const;
    /// Is any bit in @c [0,end[ set?
    bool any_end(const size_t end) const { return any_range(0, end); }
    /// Is any bit in @c [begin,infinity[ set?
    bool any_begin(const size_t begin) const { return any_range(begin, num_bits()); }
    bool any() const { return any_range(0, num_bits()); }
    //@}

    /// @name none
    /// Is no bit in range set?
    //@{
    /// Is no bit in @c [begin,end[ set?
    bool none_range(const size_t begin, const size_t end) const { return !any_range(begin, end); }
    /// Is no bit in @c [0,end[ set?
    bool none_end(const size_t end) const { return none_range(0, end); }
    /// Is no bit in @c [begin,infinity[ set?
    bool none_begin(const size_t begin) const { return none_range(begin, num_bits()); }
    bool none() const { return none_range(0, num_bits()); }
    //@}

    /// @name shift
    //@{
    BitSet& operator>>=(uint64_t shift);
    BitSet operator>>(uint64_t shift) const { BitSet res(*this); res >>= shift; return res; }
    //@}

    /// @name boolean operators
    //@{
    BitSet& operator&=(const BitSet& other) { return op_assign<std::bit_and<uint64_t>>(other); }
    BitSet& operator|=(const BitSet& other) { return op_assign<std::bit_or <uint64_t>>(other); }
    BitSet& operator^=(const BitSet& other) { return op_assign<std::bit_xor<uint64_t>>(other); }
    BitSet operator&(BitSet b) const { BitSet res(*this); res &= b; return res; }
    BitSet operator|(BitSet b) const { BitSet res(*this); res |= b; return res; }
    BitSet operator^(BitSet b) const { BitSet res(*this); res ^= b; return res; }
    //@}

    /// number of bits set
    size_t count() const;

    void friend swap(BitSet& b1, BitSet& b2) {
        using std::swap;
        swap(b1.num_words_, b2.num_words_);
        swap(b1.words_,     b2.words_);
    }

    BitSet& operator=(BitSet other) { swap(*this, other); return *this; }

private:
    void ensure_capacity(size_t num_bits) const;
    template<class F>
    BitSet& op_assign(const BitSet& other) {
        if (this->num_words() < other.num_words())
            this->ensure_capacity(other.num_bits()-1);
        auto  this_words = this->words();
        auto other_words = other.words();
        for (size_t i = 0, e = other.num_words(); i != e; ++i)
            this_words[i] = F()(this_words[i], other_words[i]);
        return *this;
    }

    void dealloc() const;
    const uint64_t* words() const { return num_words_ == 1 ? &word_ : words_; }
    uint64_t* words() { return num_words_ == 1 ? &word_ : words_; }
    size_t num_words() const { return num_words_; }
    size_t num_bits() const { return num_words_*64_s; }

    union {
        mutable uint64_t* words_;
        uint64_t word_;
    };
    mutable uint32_t num_words_;

public:
    uint32_t padding = 0;
};

static_assert(sizeof(BitSet) == 16);

}

#endif
