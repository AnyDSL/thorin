#include <cstdlib>
#include <time.h>

#ifdef __APPLE__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include <iostream>

#include "thorin_utils.h"

// common implementations of runtime utility functions
long long thorin_get_micro_time() {
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

    long long time = now.tv_sec*1000000LL + now.tv_nsec / 1000LL;
    return time;
}
void thorin_print_micro_time(long long time) {
    std::cerr << "   timing: " << time * 1.0e-3f << "(ms)" << std::endl;
}
void thorin_print_gflops(float f) { printf("GFLOPS: %f\n", f); }
float thorin_random_val(int max) {
    return ((float)random() / RAND_MAX) * max;
}
