#ifndef _THORIN_INT_RUNTIME_H
#define _THORIN_INT_RUNTIME_H

extern "C" {
	// runtime functions, used internally (by Impala etc.) [TODO: better namespacing]
	float random_val(int max);
	void getMicroTime();
} 

#endif
