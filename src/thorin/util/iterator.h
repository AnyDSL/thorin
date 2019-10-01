#ifndef THORIN_UTIL_ITERATOR_H
#define THORIN_UTIL_ITERATOR_H

#include <iterator>
#include <functional>

namespace thorin {

//------------------------------------------------------------------------------

template<class I>
struct Range {
    using iterator          = I;
    using difference_type   = typename std::iterator_traits<I>::difference_type;
    using value_type        = typename std::iterator_traits<I>::value_type;
    using reference         = typename std::iterator_traits<I>::reference;
    using pointer           = typename std::iterator_traits<I>::pointer;
    using iterator_category = typename std::iterator_traits<I>::iterator_category;

    Range(I begin, I end)
        : begin_(begin)
        , end_(end)
    {}
    Range(const Range& other)
        : begin_(other.begin())
        , end_  (other.end())
    {}
    Range(Range&& other)
        : begin_(std::move(other.begin_))
        , end_  (std::move(other.end_))
    {}

    I begin() const { return begin_; }
    I end() const { return end_; }
    size_t distance() const { return std::distance(begin(), end()); }

    value_type operator*() const { return *begin_; }
    pointer operator->() const { return (pointer) &*begin_; }
    bool operator==(const Range& other) { assert(this->end_ == other.end_); return this->begin_ == other.begin_; }
    bool operator!=(const Range& other) { assert(this->end_ == other.end_); return this->begin_ != other.begin_; }
    Range& operator=(Range other) { swap(*this, other); return *this; }

    friend void swap(Range& r1, Range& r2) {
        using std::swap;
        swap(r1.begin_, r2.begin_);
        swap(r1.end_,   r2.end_);
    }

protected:
    I begin_;
    I end_;
};

template<class I>
auto range(I begin, I end) { return Range(begin, end); }

template<class C>
auto range(const C& container) { return Range(container.begin(), container.end()); }

template<class C>
auto reverse_range(const C& container) { return range(container.rbegin(), container.rend()); }

template<class C>
auto reverse_range(C& container) { return range(container.rbegin(), container.rend()); }

//------------------------------------------------------------------------------

template<class I, class P, class V = typename std::iterator_traits<I>::value_type>
class filter_iterator : public Range<I> {
public:
    using super             = Range<I>;
    using value_type        = typename super::value_type;
    using reference         = typename super::reference;
    using pointer           = typename super::pointer;
    using iterator_category = typename super::iterator_category;

    filter_iterator(I begin, I end, P predicate)
        : Range<I>(begin, end)
        , predicate_(predicate)
    {
        skip();
    }
    filter_iterator(const filter_iterator& other)
        : Range<I>(other.begin(), other.end())
        , predicate_(other.predicate())
    {
        skip();
    }
    filter_iterator(filter_iterator&& other)
        : Range<I>(std::move(other.begin_), std::move(other.end_))
        , predicate_(std::move(other.predicate_))
    {
        skip();
    }

    P predicate() const { return predicate_; }

    filter_iterator& operator++() {
        assert(super::begin_ != super::end_);
        ++super::begin_;
        skip();
        return *this;
    }
    filter_iterator operator++(int) { filter_iterator res = *this; ++(*this); return res; }
    filter_iterator& operator=(filter_iterator other) { swap(*this, other); return *this; }

    friend void swap(filter_iterator& i1, filter_iterator& i2) {
        using std::swap;
        swap(static_cast<super&>(i1), static_cast<super&>(i2));
        swap(i1.predicate_, i2.predicate_);
    }

private:
    void skip() {
        while (super::begin_ != super::end() && !predicate_(*super::begin_))
            ++super::begin_;
    }

    P predicate_;
};

template<class I, class P>
auto filter(I begin, I end, P predicate) { return filter_iterator(begin, end, predicate); }

template<class R, class P>
auto filter(const R& range, P predicate) { return filter_iterator(range.begin(), range.end(), predicate); }

//------------------------------------------------------------------------------

template<class I, class F>
class map_iterator {
public:
    using iterator          = I;
    using difference_type   = typename std::iterator_traits<iterator>::difference_type;
    using value_type        = typename std::result_of<F(typename std::iterator_traits<I>::value_type)>::type;
    using reference         = value_type&;
    using pointer           = value_type*;
    using iterator_category = typename std::iterator_traits<I>::iterator_category;

    map_iterator(iterator iter, F map)
        : iter_(iter)
        , map_(map)
    {}
    map_iterator(const map_iterator& other)
        : iter_(other.iter_)
        , map_ (other.map_)
    {}
    map_iterator(map_iterator&& other)
        : iter_(std::move(other.iter_))
        , map_(std::move(other.map_))
    {}

    F map() const { return map_; }

    map_iterator& operator++() { ++iter_; return *this; }
    map_iterator operator++(int) { map_iterator res = *this; ++(*this); return res; }
    value_type operator*() const { return map_(*iter_); }
    pointer operator->() const { return (pointer) &map_(*iter_); }
    bool operator==(const map_iterator& other) { return this->iter_ == other.iter_; }
    bool operator!=(const map_iterator& other) { return this->iter_ != other.iter_; }
    map_iterator& operator=(map_iterator other) { swap(*this, other); return *this; }

    friend void swap(map_iterator& i1, map_iterator& i2) {
        using std::swap;
        swap(i1.iter_, i2.iter_);
        swap(i1.map_,  i2.map_ );
    }

private:
    iterator iter_;
    F map_;
};

template<class I, class F>
auto map(I begin, I end, F f) { return range(map_iterator(begin, f), map_iterator(end, f)); }

template<class R, class F>
auto map(const R& r, F f) { return range(map_iterator(r.begin(), f), map_iterator(r.end(), f)); }

//------------------------------------------------------------------------------

template<class T, class = void>
struct is_range : std::false_type {};
template<class T>
struct is_range<T, std::void_t<decltype(std::declval<T>().begin()), decltype(std::declval<T>().end())>> : std::true_type {};

//------------------------------------------------------------------------------

}

#endif
