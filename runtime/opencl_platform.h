#ifndef OPENCL_PLATFORM_H
#define OPENCL_PLATFORM_H

#include "platform.h"
#include "runtime.h"

#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

/// OpenCL platform. Has the same number of devices as that of the OpenCL implementation.
class OpenCLPlatform : public Platform {
public:
    OpenCLPlatform(Runtime* runtime);
    ~OpenCLPlatform();

protected:
    struct dim3 {
        int x, y, z;
        dim3(int x = 1, int y = 1, int z = 1) : x(x), y(y), z(z) {}
    };

    void* alloc(device_id dev, int64_t size) override;
    void* alloc_unified(device_id dev, int64_t size) override;
    void release(device_id dev, void* ptr) override;

    void set_block_size(device_id dev, int32_t x, int32_t y, int32_t z) override;
    void set_grid_size(device_id dev, int32_t x, int32_t y, int32_t z) override;
    void set_kernel_arg(device_id dev, int32_t arg, void* ptr, int32_t size) override;
    void set_kernel_arg_ptr(device_id dev, int32_t arg, void* ptr) override;
    void set_kernel_arg_struct(device_id dev, int32_t arg, void* ptr, int32_t size) override;
    void load_kernel(device_id dev, const char* file, const char* name) override;
    void launch_kernel(device_id dev) override;
    void synchronize(device_id dev) override;

    void copy(device_id dev_src, const void* src, int64_t offset_src, device_id dev_dst, void* dst, int64_t offset_dst, int64_t size) override;
    void copy_from_host(const void* src, int64_t offset_src, device_id dev_dst, void* dst, int64_t offset_dst, int64_t size) override;
    void copy_to_host(device_id dev_src, const void* src, int64_t offset_src, void* dst, int64_t offset_dst, int64_t size) override;

    int dev_count() override;

    std::string name() override { return "OpenCL"; }

    typedef std::unordered_map<std::string, cl_kernel> KernelMap;

    struct DeviceData {
        cl_platform_id platform;
        cl_device_id dev;
        cl_command_queue queue;
        cl_context ctx;
        cl_kernel kernel;

        size_t local_work_size[3], global_work_size[3];
        cl_ulong start_kernel, end_kernel;
        std::vector<void*> kernel_args;
        std::vector<void*> kernel_vals;
        std::vector<size_t> kernel_arg_sizes;
        std::list<cl_mem> kernel_structs;

        std::unordered_map<std::string, cl_program> programs;
        std::unordered_map<cl_program, KernelMap> kernels;
    };

    std::vector<DeviceData> devices_;

    void checkOpenCLErrors(cl_int err, const char*, const char*, const int);
};

#endif
