#include "thorin_runtime.h"

#ifdef PROVIDE_MAIN
extern "C" int main_impala(void);

int main(int argc, const char** argv) {
    // initialize AnyDSL runtime
    thorin_init();
    // run main
    int ret = main_impala();
    // print total timing
    thorin_print_total_timing();
    return ret;
}
#endif
