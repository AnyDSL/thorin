#include <stdlib.h>
#include <time.h>

#include <iostream>

#include "thorin_int_runtime.h"
#include "thorin_ext_runtime.h"

// common implementations of internal runtime functions
static long global_time = 0;
void getMicroTime() {
    struct timespec now;
    #ifdef __APPLE__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    now.tv_sec = mts.tv_sec;
    now.tv_nsec = mts.tv_nsec;
    #else
    clock_gettime(CLOCK_MONOTONIC, &now);
    #endif

    if (global_time==0) {
        global_time = now.tv_sec*1000000LL + now.tv_nsec / 1000LL;
    } else {
        global_time = (now.tv_sec*1000000LL + now.tv_nsec / 1000LL) - global_time;
        std::cerr << "   timing: " << global_time * 1.0e-3f << "(ms)" << std::endl;
        global_time = 0;
    }
}
float random_val(int max) {
    return ((float)random() / RAND_MAX) * max;
}


#ifdef PROVIDE_MAIN
extern "C" void main_impala(void);

int main(int argc, const char **argv) {
    // initialize AnyDSL runtime
    thorin_init();
    // run main
    main_impala();
    return 0;
}
#endif
