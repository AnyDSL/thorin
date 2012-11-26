#ifndef ANYDSL2_FILTER_H
#define ANYDSL2_FILTER_H

#include <boost/iterator/filter_iterator.hpp>

namespace anydsl2 {

template<class Pred, class C>
class Filter {
public:

    Filter(const Pred& pred, C c)
        : pred(pred)
        , c(c)
    {}

    typedef boost::filter_iterator<Pred, typename C::const_iterator> const_iterator;
    typedef boost::filter_iterator<Pred, typename C::const_reverse_iterator> const_reverse_iterator;

    const_iterator begin() const { return const_iterator(pred, c.begin(), c.end()); }
    const_iterator end()   const { return const_iterator(pred, c.  end(), c.end()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(pred, c.rbegin(), c.rend()); }
    const_reverse_iterator rend()   const { return const_reverse_iterator(pred, c.  rend(), c.rend()); }

private:

    Pred pred;
    C c;
};

template<class Pred, class C>
Filter<Pred, C> make_filter(const Pred& pred, const C& c) { return Filter<Pred, C>(pred, c); }


} // namespace anydsl2

#endif // ANYDSL2_FILTER_H
