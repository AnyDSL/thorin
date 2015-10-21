#ifndef THORIN_UTIL_ITERATOR_H
#define THORIN_UTIL_ITERATOR_H

#include <iterator>
#include <functional>

namespace thorin {

//------------------------------------------------------------------------------

template<class I, class P, class V = typename std::iterator_traits<I>::value_type>
class filter_iterator {
public:
    typedef typename std::iterator_traits<I>::difference_type difference_type;
    typedef V value_type;
    typedef V& reference;
    typedef V* pointer;
    typedef typename std::iterator_traits<I>::iterator_category iterator_category;

    filter_iterator(I iterator, I end, P predicate)
        : iterator_(iterator)
        , end_(end)
        , predicate_(predicate)
    {
        skip();
    }
    filter_iterator(const filter_iterator& other)
        : iterator_(other.iterator())
        , end_(other.end())
        , predicate_(other.predicate())
    {
        skip();
    }
    filter_iterator(filter_iterator&& other)
        : iterator_(std::move(other.iterator_))
        , end_(std::move(other.end_))
        , predicate_(std::move(other.predicate_))
    {
        skip();
    }

    I iterator() const { return iterator_; }
    I end() const { return end_; }
    P predicate() const { return predicate_; }

    filter_iterator& operator= (filter_iterator other) { swap(*this, other); return *this; }
    filter_iterator& operator++ () {
        assert(iterator_ != end_);
        ++iterator_;
        skip();
        return *this;
    }
    filter_iterator operator++ (int) { filter_iterator res = *this; ++(*this); return res; }
    reference operator* () const { return (reference) *iterator_; }
    pointer operator-> () const { return (pointer) &*iterator_; }
    bool operator== (const filter_iterator& other) { return this->iterator_ == other.iterator_; }
    bool operator!= (const filter_iterator& other) { return this->iterator_ != other.iterator_; }
    friend void swap(filter_iterator& i1, filter_iterator& i2) {
        using std::swap;
        swap(i1, i2);
    }

private:
    void skip() {
        while (iterator_ != end() && !predicate_(*iterator_))
            ++iterator_;
    }

    I iterator_;
    I end_;
    P predicate_;
};

//------------------------------------------------------------------------------

template<class I, class P>
filter_iterator<I, P> make_filter(I begin, I end, P pred) { return filter_iterator<I, P>(begin, end, pred); }

//------------------------------------------------------------------------------

template<class I>
struct Range {
    Range(I begin, I end)
        : begin_(begin)
        , end_(end)
    {}

    I begin() const { return begin_; }
    I end() const { return end_; }

private:
    I begin_;
    I end_;
};

//------------------------------------------------------------------------------

template<class I>
Range<I> range(I begin, I end) { return Range<I>(begin, end); }

template<class I, class P>
Range<filter_iterator<I, P>> range(I begin, I end, P predicate) {
    typedef filter_iterator<I, P> Filter;
    return range(Filter(begin, end, predicate), Filter(end, end, predicate));
}

template<class V, class I, class P>
Range<filter_iterator<I, P, V>> range(I begin, I end, P predicate) {
    typedef filter_iterator<I, P, V> Filter;
    return range(Filter(begin, end, predicate), Filter(end, end, predicate));
}

//------------------------------------------------------------------------------

}

#endif
