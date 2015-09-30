#include <iostream>

#include "log.h"

class Counter : public thorin::Streamable {
public:
    Counter() : c(0) { };

    void inc() {
        c++;
    }

    const int get() const {
        return c;
    }

    const void stream(std::ostream &out) const {
      out << "[" << get() << "]";
    }

private:
    int c;
};

int main(int argc, const char* argv[]) {
    if (argc != 3)
        exit(-1);

    thorin::Log::set_level((thorin::Log::Level) atoi(argv[1]));
    thorin::Log::set_stream(std::cerr);
    int runs = atoi(argv[2]);

    ILOG("Starting %i runs!", runs);
    for (Counter c; c.get() < runs; c.inc()) {
        DLOG("-> %Y / %i", &c, runs);
    }
    ILOG("Finished!");
}
