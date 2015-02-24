#ifndef THORIN_UTIL_FILTER_H
#define THORIN_UTIL_FILTER_H

#include <iterator>
#include <functional>

namespace thorin {

template<class I, class P>
class filter_iterator {
public:
    typedef typename std::iterator_traits<I>::difference_type difference_type;
    typedef typename std::iterator_traits<I>::value_type value_type;
    typedef typename std::iterator_traits<I>::reference reference;
    typedef typename std::iterator_traits<I>::pointer pointer;
    typedef typename std::iterator_traits<I>::iterator_category iterator_category;

    filter_iterator(I iterator, I end, P predicate)
        : iterator_(iterator)
        , end_(end)
        , predicate_(predicate)
    {}
    filter_iterator(const filter_iterator& other)
        : iterator_(other.iterator())
        , end_(other.end())
        , predicate_(other.predicate())
    {}
    filter_iterator(filter_iterator&& other) 
        : iterator_(std::move(other.iterator_))
        , end_(std::move(other.end_))
        , predicate_(std::move(other.predicate_))
    {}

    I iterator() const { return iterator_; }
    I end() const { return end_; }
    P predicate() const { return predicate_; }

    filter_iterator& operator= (filter_iterator other) { swap(*this, other); return *this; }
    filter_iterator& operator++ () { 
        do {
            ++iterator_;
        } while (predicate_(*iterator_) && iterator_ != end());
        return *this; 
    }
    filter_iterator operator++ (int) { filter_iterator res = *this; ++(*this); return res; }
    reference operator* () const { return *iterator_; }
    pointer operator-> () const { return &*iterator_; }
    bool operator== (const filter_iterator& other) { return this->iterator_ == other.iterator_; }
    bool operator!= (const filter_iterator& other) { return this->iterator_ != other.iterator_; }
    friend void swap(filter_iterator& i1, filter_iterator& i2) {
        using std::swap;
        swap(i1, i2);
    }

private:
    I iterator_;
    I end_;
    P predicate_;
};

template<class I, class P>
filter_iterator<I, P> make_filter(I begin, I end, P pred) { return filter_iterator<I, P>(begin, end, pred); }

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

template<class I, class P>
Range<filter_iterator<I, P>> range(I begin, I end, P predicate) { 
    typedef filter_iterator<I, P> Filter;
    return Range<Filter>(Filter(begin, end, predicate), Filter(end, end, predicate));
}

}

#endif
