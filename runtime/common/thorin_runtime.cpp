#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include <iostream>

#include "thorin_int_runtime.h"
#include "thorin_ext_runtime.h"

// common implementations of internal runtime functions
static long global_time = 0;
long get_micro_time() {
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

    long time = now.tv_sec*1000000LL + now.tv_nsec / 1000LL;
    if (global_time==0) {
        global_time = time;
    } else {
        global_time = time - global_time;
        std::cerr << "   timing: " << global_time * 1.0e-3f << "(ms)" << std::endl;
        global_time = 0;
    }
    return time;
}
float random_val(int max) {
    return ((float)random() / RAND_MAX) * max;
}


#ifdef PROVIDE_MAIN
extern "C" int main_impala(void);

int main(int argc, const char **argv) {
    // initialize AnyDSL runtime
    thorin_init();
    // run main
    int ret = main_impala();
    // print total timing
    thorin_print_total_timing();
    return ret;
}
#endif
