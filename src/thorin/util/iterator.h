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
    typedef std::forward_iterator_tag iterator_category;

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

    filter_iterator& operator++() {
        assert(iterator_ != end_);
        ++iterator_;
        skip();
        return *this;
    }
    filter_iterator operator++(int) { filter_iterator res = *this; ++(*this); return res; }
    reference operator*() const { return (reference) *iterator_; }
    pointer operator->() const { return (pointer) &*iterator_; }
    bool operator==(const filter_iterator& other) { return this->iterator_ == other.iterator_; }
    bool operator!=(const filter_iterator& other) { return this->iterator_ != other.iterator_; }
    filter_iterator& operator=(filter_iterator other) { swap(*this, other); return *this; }

    friend void swap(filter_iterator& i1, filter_iterator& i2) { using std::swap; swap(i1, i2); }

private:
    void skip() {
        while (iterator_ != end() && !predicate_(*iterator_))
            ++iterator_;
    }

    I iterator_;
    I end_;
    P predicate_;
};

template<class I, class P>
filter_iterator<I, P> make_filter(I begin, I end, P pred) { return filter_iterator<I, P>(begin, end, pred); }

//------------------------------------------------------------------------------

template<class I, class OutT, class F>
class map_iterator {
public:
    typedef typename std::iterator_traits<I>::difference_type difference_type;
    typedef OutT value_type;
    typedef OutT& reference;
    typedef OutT* pointer;
    typedef std::forward_iterator_tag iterator_category;

    map_iterator(I iterator, I end, F function)
        : iterator_(iterator)
        , end_(end)
        , function_(function)
    {}
    map_iterator(const map_iterator& other)
        : iterator_(other.iterator())
        , end_(other.end())
        , function_(other.function())
    {}
    map_iterator(map_iterator&& other)
        : iterator_(std::move(other.iterator_))
        , end_(std::move(other.end_))
        , function_(std::move(other.function_))
    {}

    I iterator() const { return iterator_; }
    I end() const { return end_; }
    F function() const { return function_; }

    map_iterator& operator++() {
        assert(iterator_ != end_);
        ++iterator_;
        return *this;
    }
    map_iterator operator++(int) { map_iterator res = *this; ++(*this); return res; }
    value_type operator*() const { return function_(*iterator_); }
    pointer operator->() const { return (pointer) &function_(*iterator_); }
    bool operator==(const map_iterator& other) { return this->iterator_ == other.iterator_; }
    bool operator!=(const map_iterator& other) { return this->iterator_ != other.iterator_; }
    map_iterator& operator=(map_iterator other) { swap(*this, other); return *this; }

    friend void swap(map_iterator& i1, map_iterator& i2) { using std::swap; swap(i1, i2); }

private:

    I iterator_;
    I end_;
    F function_;
};

//------------------------------------------------------------------------------

template<class I>
struct Range {
    Range(I begin, I end)
        : begin_(begin)
        , end_(end)
    {}

    I begin() const { return begin_; }
    I end() const { return end_; }
    size_t distance() const { return std::distance(begin(), end()); }

private:
    I begin_;
    I end_;
};

template<class I>
auto range(I begin, I end) -> Range<I> { return Range<I>(begin, end); }

template<class T>
auto range(const T& t) -> Range<decltype(t.begin())> { return range(t.begin(), t.end()); }

template<class T>
auto reverse_range(const T& t) -> Range<decltype(t.rbegin())> { return range(t.rbegin(), t.rend()); }

template<class I, class P>
auto  range(I begin, I end, P predicate) -> Range<filter_iterator<I, P>> {
    typedef filter_iterator<I, P> Filter;
    return range(Filter(begin, end, predicate), Filter(end, end, predicate));
}

template<class V, class I, class P>
auto range(I begin, I end, P predicate) -> Range<filter_iterator<I, P, V>> {
    typedef filter_iterator<I, P, V> Filter;
    return range(Filter(begin, end, predicate), Filter(end, end, predicate));
}

template<class I, class F>
auto map_range(I begin, I end, F function) -> Range<map_iterator<I, decltype(function(*begin)), F>> {
    typedef map_iterator<I, decltype(function(*begin)), F> Map;
    return range(Map(begin, end, function), Map(end, end, function));
}

//------------------------------------------------------------------------------

}

#endif
