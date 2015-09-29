#include "log.h"

/*
 *       Custom Class
 */
class Counter : Printable {
public:
    Counter() : c(0) { };

    void inc() {
        c++;
    }

    const int get() const {
        return c;
    }

    const void print(std::ostream &out) const {
      out << "[" << get() << "]";
    }

private:
    int c;
};
