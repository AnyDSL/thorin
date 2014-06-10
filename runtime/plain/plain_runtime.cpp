#include <stdlib.h>
#include <iostream>

extern "C"
{
	// runtime functions
	void thorin_init();
	void *thorin_malloc(size_t size);
	void thorin_free(void *ptr);
	extern int main_impala();
}


// helper functions
void thorin_init() {  }
void *thorin_malloc(size_t size) {
    void *mem;
    posix_memalign(&mem, 64, size);
    std::cerr << " * malloc() -> " << mem << std::endl;
    return mem;
}
void thorin_free(void *ptr) {
    free(ptr);
}

#ifdef PROVIDE_MAIN
int main(int argc, char *argv[]) {
    return main_impala();
}
#endif
