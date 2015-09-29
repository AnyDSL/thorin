#include <stdio.h>
#include <stdlib.h>

#include "log.h"

LogLevel Logging::level = LogLevel::Info;

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

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        exit(-1);
    }

    if (atoi(argv[1]) == 0) {
        Logging::level = LogLevel::Debug;
    }

    int runs = atoi(argv[2]);

    ILOG("Starting %i runs!\n", runs);
    for (Counter c; c.get() < runs; c.inc()) {
        DLOG("-> %Y / %i\n", &c, runs);
    }
    ILOG("Finished!\n");
}
