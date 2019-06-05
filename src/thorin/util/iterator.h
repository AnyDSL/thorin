#ifndef THORIN_UTIL_ITERATOR_H
#define THORIN_UTIL_ITERATOR_H

#include <iterator>
#include <functional>

namespace thorin {

//------------------------------------------------------------------------------

template<class I>
struct range {
    range(I begin, I end)
        : begin_(begin)
        , end_(end)
    {}

    I begin() const { return begin_; }
    I end() const { return end_; }
    size_t distance() const { return std::distance(begin(), end()); }

    range(const range& other)
        : begin_(other.begin())
        , end_  (other.end())
    {}
    range(range&& other)
        : super(std::move(other.begin_), std::move(other.end_))
        , function_(std::move(other.function_))
    {}

    value_type operator*() const { return function_(*begin_); }
    pointer operator->() const { return (pointer) &function_(*begin_); }
    bool operator==( const map_iterator& other) { assert(this->end_ == other.end_); return this->begin_ == other.begin_; }
    bool operator!=( const map_iterator& other) { assert(this->end_ == other.end_); return this->begin_ != other.begin_; }
    range& operator=(range other) { assert(this->end_ == other.end_); swap(*this, other); return *this; }

    friend void swap(range& r1, range& r2) {
        using std::swap;
        swap(r1.begin_, r2.begin_);
        swap(r1.end_,   r2.end_);
    }

private:
    I begin_;
    I end_;
};

template<class C>
auto reverse_range(const C& container) { return range(container.rbegin(), container.rend()); }

template<class C>
auto reverse_range(C& container) { return range(container.rbegin(), container.rend()); }

//------------------------------------------------------------------------------

template<class I, class P, class V = typename std::iterator_traits<I>::value_type>
class filter_iterator : public range<I> {
public:
    using super             = range<I>;
    using iterator          = I;
    //using difference_type   = typename R::difference_type;
    using value_type        = V;
    using reference         = V&;
    using pointer           = V*;
    using iterator_category = std::forward_iterator_tag;

    filter_iterator(iterator begin, iterator end, P predicate)
        : range<I>(begin, end)
        , predicate_(predicate)
    {
        skip();
    }
    filter_iterator(const filter_iterator& other)
        : range<I>(other.begin(), other.end())
        , predicate_(other.predicate())
    {
        skip();
    }
    filter_iterator(filter_iterator&& other)
        : range<I>(std::move(other.begin_), std::move(other.end_))
        , predicate_(std::move(other.predicate_))
    {
        skip();
    }

    P predicate() const { return predicate_; }

    filter_iterator& operator++() {
        assert(begin_ != end_);
        ++begin_;
        skip();
        return *this;
    }
    filter_iterator operator++(int) { filter_iterator res = *this; ++(*this); return res; }
    reference operator*() const { return (reference) *begin_; }
    pointer operator->() const { return (pointer) &*begin_; }
    bool operator==(const filter_iterator& other) { assert(this->end() == other.end()); return this->begin_ == other.begin_; }
    bool operator!=(const filter_iterator& other) { assert(this->end() == other.end()); return this->begin_ != other.begin_; }
    filter_iterator& operator=(filter_iterator other) { swap(*this, other); return *this; }

    friend void swap(filter_iterator& i1, filter_iterator& i2) {
        using std::swap;
        swap(static_cast<super&>(s1), static_cast<super&>(s2));
        swap(i1.predicate_, i2.predicate_);
    }

private:
    void skip() {
        while (begin_ != end() && !predicate_(*begin_))
            ++begin_;
    }

    P predicate_;
};

template<class R, class P>
auto make_filter_iterator(const R& range, P predicate) { return filter_iterator(range.begin(), range.end(), predicate); }

//------------------------------------------------------------------------------

template<class I, class F>
class map_iterator : public range<I> {
public:
    using super             = range<I>;
    using iterator          = I;
    //using difference_type   = typename std::iterator_traits<iterator>::difference_type;
    using value_type        = typename std::result_of<F(typename I::value_type)>::type;
    using reference         = value_type&;
    using pointer           = value_type*;
    using iterator_category = std::forward_iterator_tag;

    map_iterator(iterator begin, iterator end, F function)
        : super(begin, end)
        , function_(function)
    {}
    map_iterator(const map_iterator& other)
        : super(other.begin(), other.end())
        , function_(other.function())
    {}
    map_iterator(map_iterator&& other)
        : super(std::move(other.begin_), std::move(other.end_))
        , function_(std::move(other.function_))
    {}

    F function() const { return function_; }

    map_iterator& operator++() {
        assert(begin_ != end_);
        ++begin_;
        return *this;
    }
    map_iterator operator++(int) { map_iterator res = *this; ++(*this); return res; }
    value_type operator*() const { return function_(*begin_); }
    pointer operator->() const { return (pointer) &function_(*begin_); }
    bool operator==( const map_iterator& other) { assert(this->end_ == other.end_); return this->begin_ == other.begin_; }
    bool operator!=( const map_iterator& other) { assert(this->end_ == other.end_); return this->begin_ != other.begin_; }
    map_iterator& operator=(map_iterator other) { assert(this->end_ == other.end_); swap(*this, other); return *this; }

    friend void swap(map_iterator& i1, map_iterator& i2) {
        using std::swap;
        swap(static_cast<super&>(s1), static_cast<super&>(s2));
        swap(i1.predicate_, i2.predicate_);
    }


private:
    F function_;
};

template<class R, class F>
auto make_map_iterator(const R& range, F f) { return map_iterator(range.begin(), range.end(), f); }

//------------------------------------------------------------------------------

}

#endif
