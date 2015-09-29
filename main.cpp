#include <stdio.h>
#include <stdlib.h>

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
    if (argc < 3)
        exit(-1);

    thorin::Log::set_level((thorin::Log::Level) atoi(argv[1]));
    int runs = atoi(argv[2]);

    ILOG("Starting %i runs!\n", runs);
    for (Counter c; c.get() < runs; c.inc()) {
        DLOG("-> %Y / %i\n", &c, runs);
    }
    ILOG("Finished!\n");
}
