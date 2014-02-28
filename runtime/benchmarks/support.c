#include <stdlib.h>
#include <stdint.h>

inline uint64_t rdtsc() {
    uint32_t low, high;
    __asm__ __volatile__ (
        "xorl %%eax,%%eax \n    cpuid"
        ::: "%rax", "%rbx", "%rcx", "%rdx" );
    __asm__ __volatile__ (
                            "rdtsc" : "=a" (low), "=d" (high));
    return (uint64_t)high << 32 | low;
}
            
static uint64_t start, end;
void reset_and_start_timer() { start = rdtsc(); }
void print_time() {
    end = rdtsc();
    printf("elapsed time: %d\n", (end-start) / (1024. * 1024.));
}

void* array(int size) {
    void* data;
    posix_memalign(&data, 16, size*sizeof(float));
    return data;
}

int main() {
    return main_impala();
}
