int ave(int a, int b) {
   return (a+b)/2;
}

__kernel void simple(__global int *out) {
   int tid = get_global_id(0);
   out[tid] = ave(tid, tid);
}

__kernel void simple_cmem(__global int *out, __constant int *cmem) {
   int tid = get_global_id(0);
   out[tid] = cmem[get_local_id(0)];
}
