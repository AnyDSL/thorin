#include "thorin_runtime.h"

#ifdef PROVIDE_MAIN
extern "C" int main_impala(void);

int main(int argc, const char** argv) {
    // initialize AnyDSL runtime
    thorin_init();
    // run main
    return main_impala();
}
#endif
