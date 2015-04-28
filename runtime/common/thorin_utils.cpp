#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>

#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "thorin_utils.h"

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
void* thorin_aligned_malloc(size_t size, size_t alignment) { return ::aligned_alloc(alignment, size); }
void thorin_aligned_free(void* ptr) { ::free(ptr); }
#elif _POSIX_VERSION >= 200112L || _XOPEN_SOURCE >= 600
void* thorin_aligned_malloc(size_t size, size_t alignment) {
    void* p;
    posix_memalign(&p, alignment, size);
    return p;
}
void thorin_aligned_free(void* ptr) { free(ptr); }
#elif defined(_WIN32) || defined(__CYGWIN__)
#include <malloc.h>

void* thorin_aligned_malloc(size_t size, size_t alignment) { return ::_aligned_malloc(size, alignment); }
void thorin_aligned_free(void* ptr) { ::_aligned_free(ptr); }
#else
#error "don't know how to retrieve aligned memory"
#endif

long long thorin_get_micro_time() {
#if defined(_WIN32) || defined(__CYGWIN__) // Use QueryPerformanceCounter on Windows
    LARGE_INTEGER counter, freq;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&freq);
    return counter.QuadPart * 1000000LL / freq.QuadPart;
#else
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
    long long time = now.tv_sec * 1000000LL + now.tv_nsec / 1000LL;
    return time;
#endif
}

void thorin_print_micro_time(long long time) {
    std::cerr << "   timing: " << time / 1000 << "(ms)" << std::endl;
}

void thorin_print_gflops(float f) { printf("GFLOPS: %f\n", f); }

float thorin_random_val(int max) {
    static thread_local std::mt19937 std_gen;
    static std::uniform_real_distribution<float> std_dist(0.0f, 1.0f);
    return std_dist(std_gen) * max;
}
