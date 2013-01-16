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

    template<class T>
    Printer& dump(const T* t) {
        if (t)
            t->vdump(*this);
        else
            o << "<NULL>";

        return *this;
    }

    Printer& dump_name(const Def* def);

    Printer& newline();
    Printer& up();
    Printer& down();

    template<class T>
    Printer& operator << (const T& data) {
        o << data;
        return *this;
    }

    std::ostream& o;
    int indent;

    static void dump_name(std::ostream& o, const Def* def, bool fancy);

private:

    bool fancy_;
};

#define ANYDSL2_DUMP_COMMA_LIST(p, list) \
    const BOOST_TYPEOF((list))& l = (list); \
    if (!l.empty()) { \
        boost::remove_const<BOOST_TYPEOF(l)>::type::const_iterator i = l.begin(), e = l.end() - 1; \
        for (; i != e; ++i) { \
            (p).dump(*i); \
            (p) << ", "; \
        } \
        (p).dump(*i); \
    }

} // namespace anydsl2

#endif
