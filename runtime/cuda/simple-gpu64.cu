
__device__ int ave(int a, int b)
{
   return (a+b)/2;
}

extern "C" __global__ void simple(int *data)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   data[tid] = ave(tid, tid);
}

texture<int, cudaTextureType1D, cudaReadModeElementType> tex;
extern "C" __global__ void simple_tex(int *data)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   data[tid] = tex1Dfetch(tex, tid);
}

