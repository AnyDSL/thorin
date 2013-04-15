#ifndef ANYDSL2_PRINTER_H
#define ANYDSL2_PRINTER_H

#include <iostream>

namespace anydsl2 {

class Def;

class Printer {
public:

    Printer(std::ostream& o, bool fancy)
        : o(o)
        , indent(0)
        , fancy_(fancy)
    {}

    bool fancy() const { return fancy_; }

    Printer& print_name(const Def* def);

    Printer& newline();
    Printer& up() { ++indent; return newline(); }
    Printer& down() { --indent; return newline(); }

    template<class T>
    Printer& operator << (const T& data) {
        o << data;
        return *this;
    }

    std::ostream& o;
    int indent;

private:

    bool fancy_;
};

template<class P, class T>
P& checked_print(P& p, const T* t) {
    if (t)
        t->print(p);
    else
        p << "<NULL>";

    return p;
}

#define ANYDSL2_DUMP_COMMA_LIST(p, list) \
    const BOOST_TYPEOF((list))& l = (list); \
    if (!l.empty()) { \
        boost::remove_const<BOOST_TYPEOF(l)>::type::const_iterator i = l.begin(), e = l.end() - 1; \
        for (; i != e; ++i) { \
            checked_print((p), (*i)); \
            (p) << ", "; \
        } \
        checked_print((p), (*i)); \
    }

} // namespace anydsl2

#endif
