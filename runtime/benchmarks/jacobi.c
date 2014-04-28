#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define SIZE 2014

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
int main() {
    float* in   = malloc(sizeof(float)*SIZE*SIZE);
    float* out  = malloc(sizeof(float)*SIZE*SIZE);
    float a     = 0.2f;
    float b     = 1.0f - 4.0f * a;
    float stencil[3][3] = {{0.0f, b, 0.0f},
                           {   b, a, b   },
                           {0.0f, b, 0.0f}};
    getMicroTime();
    for (int y = 0+1; y < SIZE-1; ++y) {
        for (int x = 0+1; x < SIZE-1; ++x) {
            float sum = 0.f;
#if 1
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 3; ++i) {
                    float stencil_val = stencil[i][j];
                    size_t idx_x = x+i-1;
                    size_t idx_y = y+j-1;
                    sum += in[idx_y * SIZE + idx_x] * stencil_val;
                }
            }
#else
            sum += in[(y+1) * SIZE + (x  )] * stencil[0][1];
            sum += in[(y  ) * SIZE + (x-1)] * stencil[1][0];
            sum += in[(y  ) * SIZE + (x  )] * stencil[1][1];
            sum += in[(y  ) * SIZE + (x+1)] * stencil[1][2];
            sum += in[(y-1) * SIZE + (x  )] * stencil[2][1];

#endif
            out[y*SIZE + x] = sum;
        }
    }
    getMicroTime();

    free(in);
    free(out);
}
