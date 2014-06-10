#ifndef _THORIN_EXT_RUNTIME_H
#define _THORIN_EXT_RUNTIME_H

extern "C" {
	// runtime functions, used externally from C++ interface
	void thorin_init();
	void *thorin_malloc(size_t size);
	void thorin_free(void *ptr);
} 

#endif
