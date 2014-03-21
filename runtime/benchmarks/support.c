#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main_impala();

void getMicroTime() {
    static long global_time = 0;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    if (global_time==0) {
        global_time = now.tv_sec*1000000LL + now.tv_nsec / 1000LL;
    } else {
        global_time = (now.tv_sec*1000000LL + now.tv_nsec / 1000LL) - global_time;
        printf("\ttiming: %f(ms)\n", global_time * 1.0e-3);
        global_time = 0;
    }
}

void* array(int elem_size, int width, int height) {
    void *mem;
    posix_memalign(&mem, 32, elem_size * width * height);
    return mem;
}

void free_array(void *host) {
    free(host);
}

float random_val(int max) {
    return ((float)random() / RAND_MAX) * max;
}

int main() {
    return main_impala();
}
