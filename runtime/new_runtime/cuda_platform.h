#ifndef CUDA_PLATFORM_H
#define CUDA_PLATFORM_H

#include "platform.h"
#include "runtime.h"

#include <cstring>
#include <vector>
#include <unordered_map>

#include <cuda.h>
#include <nvvm.h>

#if CUDA_VERSION < 6050
    #error "CUDA 6.5 or higher required!"
#endif

#if CUDA_VERSION >= 7000
#include <nvrtc.h>
#endif

/// CUDA platform. Has the same number of devices as that of the CUDA implementation.
class CudaPlatform : public Platform {
public:
    CudaPlatform(Runtime* runtime);

protected:
    struct dim3 {
        unsigned int x, y, z;
        dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
    };

    void* alloc(device_id dev, int64_t size) override;
    void release(device_id dev, void* ptr, int64_t size) override;

    void* map(void* ptr, int64_t offset, int64_t size);
    void unmap(void* view) override;

    void set_block_size(device_id dev, uint32_t x, uint32_t y, uint32_t z) override;
    void set_grid_size(device_id dev, uint32_t x, uint32_t y, uint32_t z) override;
    void set_arg(device_id dev, uint32_t arg, void* ptr, uint32_t size) override;
    void load_kernel(device_id dev, const char* file, const char* name) override;
    void launch_kernel(device_id dev) override;

    void copy(const void* src, void* dst) override;
    void copy_from_host(const void* src, void* dst) override;
    void copy_to_host(const void* src, void* dst) override;

    int dev_count() override;

    std::string name() override { return "CUDA"; }

    typedef std::unordered_map<std::string, CUfunction> FunctionMap;

    struct DeviceData {
        CUdevice dev;
        CUcontext ctx;
        int compute_minor;
        int compute_major;

        dim3 grid, block;
        CUfunction kernel;
        std::vector<void*> kernel_args;

        std::unordered_map<std::string, CUmodule> modules;
        std::unordered_map<CUmodule, FunctionMap> functions;
    };

    std::vector<DeviceData> devices_;

    void checkCudaErrors(CUresult err, const char*, const char*, const int);
    void checkNvvmErrors(nvvmResult err, const char*, const char*, const int);
#if CUDA_VERSION >= 7000
    void checkNvrtcErrors(nvrtcResult err, const char*, const char*, const int);
#endif

    void compile_nvvm(device_id dev, const char* file_name, CUjit_target target_cc);
    void compile_cuda(device_id dev, const char* file_name, CUjit_target target_cc);
    void create_module(device_id dev, const char* file_name, CUjit_target target_cc, const void* ptx);
};

#endif
