#include "opencl_platform.h"
#include "runtime.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#ifndef KERNEL_DIR
#define KERNEL_DIR ""
#endif

std::string get_opencl_error_code_str(int error) {
    #define CL_ERROR_CODE(CODE) case CODE: return #CODE;
    switch (error) {
        CL_ERROR_CODE(CL_SUCCESS)
        CL_ERROR_CODE(CL_DEVICE_NOT_FOUND)
        CL_ERROR_CODE(CL_DEVICE_NOT_AVAILABLE)
        CL_ERROR_CODE(CL_COMPILER_NOT_AVAILABLE)
        CL_ERROR_CODE(CL_MEM_OBJECT_ALLOCATION_FAILURE)
        CL_ERROR_CODE(CL_OUT_OF_RESOURCES)
        CL_ERROR_CODE(CL_OUT_OF_HOST_MEMORY)
        CL_ERROR_CODE(CL_PROFILING_INFO_NOT_AVAILABLE)
        CL_ERROR_CODE(CL_MEM_COPY_OVERLAP)
        CL_ERROR_CODE(CL_IMAGE_FORMAT_MISMATCH)
        CL_ERROR_CODE(CL_IMAGE_FORMAT_NOT_SUPPORTED)
        CL_ERROR_CODE(CL_BUILD_PROGRAM_FAILURE)
        CL_ERROR_CODE(CL_MAP_FAILURE)
        #ifdef CL_VERSION_1_1
        CL_ERROR_CODE(CL_MISALIGNED_SUB_BUFFER_OFFSET)
        CL_ERROR_CODE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
        #endif
        #ifdef CL_VERSION_1_2
        CL_ERROR_CODE(CL_COMPILE_PROGRAM_FAILURE)
        CL_ERROR_CODE(CL_LINKER_NOT_AVAILABLE)
        CL_ERROR_CODE(CL_LINK_PROGRAM_FAILURE)
        CL_ERROR_CODE(CL_DEVICE_PARTITION_FAILED)
        CL_ERROR_CODE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
        #endif
        CL_ERROR_CODE(CL_INVALID_VALUE)
        CL_ERROR_CODE(CL_INVALID_DEVICE_TYPE)
        CL_ERROR_CODE(CL_INVALID_PLATFORM)
        CL_ERROR_CODE(CL_INVALID_DEVICE)
        CL_ERROR_CODE(CL_INVALID_CONTEXT)
        CL_ERROR_CODE(CL_INVALID_QUEUE_PROPERTIES)
        CL_ERROR_CODE(CL_INVALID_COMMAND_QUEUE)
        CL_ERROR_CODE(CL_INVALID_HOST_PTR)
        CL_ERROR_CODE(CL_INVALID_MEM_OBJECT)
        CL_ERROR_CODE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CL_ERROR_CODE(CL_INVALID_IMAGE_SIZE)
        CL_ERROR_CODE(CL_INVALID_SAMPLER)
        CL_ERROR_CODE(CL_INVALID_BINARY)
        CL_ERROR_CODE(CL_INVALID_BUILD_OPTIONS)
        CL_ERROR_CODE(CL_INVALID_PROGRAM)
        CL_ERROR_CODE(CL_INVALID_PROGRAM_EXECUTABLE)
        CL_ERROR_CODE(CL_INVALID_KERNEL_NAME)
        CL_ERROR_CODE(CL_INVALID_KERNEL_DEFINITION)
        CL_ERROR_CODE(CL_INVALID_KERNEL)
        CL_ERROR_CODE(CL_INVALID_ARG_INDEX)
        CL_ERROR_CODE(CL_INVALID_ARG_VALUE)
        CL_ERROR_CODE(CL_INVALID_ARG_SIZE)
        CL_ERROR_CODE(CL_INVALID_KERNEL_ARGS)
        CL_ERROR_CODE(CL_INVALID_WORK_DIMENSION)
        CL_ERROR_CODE(CL_INVALID_WORK_GROUP_SIZE)
        CL_ERROR_CODE(CL_INVALID_WORK_ITEM_SIZE)
        CL_ERROR_CODE(CL_INVALID_GLOBAL_OFFSET)
        CL_ERROR_CODE(CL_INVALID_EVENT_WAIT_LIST)
        CL_ERROR_CODE(CL_INVALID_EVENT)
        CL_ERROR_CODE(CL_INVALID_OPERATION)
        CL_ERROR_CODE(CL_INVALID_GL_OBJECT)
        CL_ERROR_CODE(CL_INVALID_BUFFER_SIZE)
        CL_ERROR_CODE(CL_INVALID_MIP_LEVEL)
        CL_ERROR_CODE(CL_INVALID_GLOBAL_WORK_SIZE)
        #ifdef CL_VERSION_1_1
        CL_ERROR_CODE(CL_INVALID_PROPERTY)
        #endif
        #ifdef CL_VERSION_1_2
        CL_ERROR_CODE(CL_INVALID_IMAGE_DESCRIPTOR)
        CL_ERROR_CODE(CL_INVALID_COMPILER_OPTIONS)
        CL_ERROR_CODE(CL_INVALID_LINKER_OPTIONS)
        CL_ERROR_CODE(CL_INVALID_DEVICE_PARTITION_COUNT)
        #endif
        #ifdef CL_VERSION_2_0
        CL_ERROR_CODE(CL_INVALID_PIPE_SIZE)
        CL_ERROR_CODE(CL_INVALID_DEVICE_QUEUE)
        #endif
        default: return "unknown error code";
    }
    #undef CL_ERROR_CODE
}

#define checkErr(err, name)  checkOpenCLErrors (err, name, __FILE__, __LINE__)

void OpenCLPlatform::checkOpenCLErrors(cl_int err, const char* name, const char* file, const int line) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")"
                  << " [file " << file << ", line " << line << "]: "
                  << get_opencl_error_code_str(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

OpenCLPlatform::OpenCLPlatform(Runtime* runtime)
    : Platform(runtime)
{
}

OpenCLPlatform::~OpenCLPlatform() {
    for (size_t i = 0; i < devices_.size(); i++) {
        // TODO
    }
}

void* OpenCLPlatform::alloc(device_id dev, int64_t size) {
    if (!size) return 0;

    cl_int err = CL_SUCCESS;
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl_mem mem = clCreateBuffer(devices_[dev].ctx, flags, size, NULL, &err);
    checkErr(err, "clCreateBuffer()");

    return (void*)mem;
}

void* OpenCLPlatform::alloc_unified(device_id dev, int64_t size) {
    if (!size) return 0;

    cl_int err = CL_SUCCESS;
    // CL_MEM_ALLOC_HOST_PTR -> OpenCL allocates memory that can be shared - preferred on AMD hardware ?
    // CL_MEM_USE_HOST_PTR   -> use existing, properly aligned and sized memory
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
    cl_mem mem = clCreateBuffer(devices_[dev].ctx, flags, size, NULL, &err);
    checkErr(err, "clCreateBuffer()");

    return (void*)mem;
}

void OpenCLPlatform::release(device_id, void* ptr) {
    cl_int err = clReleaseMemObject((cl_mem)ptr);
    checkErr(err, "clReleaseMemObject()");
}

void OpenCLPlatform::set_block_size(device_id dev, int32_t x, int32_t y, int32_t z) {
    auto& local_work_size = devices_[dev].local_work_size;
    local_work_size[0] = x;
    local_work_size[1] = y;
    local_work_size[2] = z;
}

void OpenCLPlatform::set_grid_size(device_id dev, int32_t x, int32_t y, int32_t z) {
    auto& global_work_size = devices_[dev].global_work_size;
    global_work_size[0] = x;
    global_work_size[1] = y;
    global_work_size[2] = z;
}

void OpenCLPlatform::set_kernel_arg(device_id dev, int32_t arg, void* ptr, int32_t size) {
    auto& args = devices_[dev].kernel_args;
    args.resize(std::max(arg + 1, (int32_t)args.size()));
    args[arg] = ptr;
}

void OpenCLPlatform::set_kernel_arg_ptr(device_id dev, int32_t arg, void* ptr) {
    auto& vals = devices_[dev].kernel_vals;
    auto& args = devices_[dev].kernel_args;
    vals.resize(std::max(arg + 1, (int32_t)vals.size()));
    args.resize(std::max(arg + 1, (int32_t)args.size()));
    vals[arg] = ptr;
    // The argument will be set at kernel launch (since the vals array may grow)
    args[arg] = nullptr;
}

void OpenCLPlatform::set_kernel_arg_struct(device_id dev, int32_t arg, void* ptr, int32_t size) {
    set_kernel_arg(dev, arg, ptr, size);
}

void OpenCLPlatform::load_kernel(device_id dev, const char* file, const char* name) {
}

void OpenCLPlatform::launch_kernel(device_id dev) {
}

extern std::atomic_llong thorin_kernel_time;

void OpenCLPlatform::synchronize(device_id dev) {
    cl_int err = clFinish(devices_[dev].queue);
    checkErr(err, "clFinish()");
    // TODO
    //float time;
    //CUresult err = cuEventSynchronize(cuda_dev.end_kernel);
    //checkErrDrv(err, "cuEventSynchronize()");

    //cuEventElapsedTime(&time, cuda_dev.start_kernel, cuda_dev.end_kernel);
    //thorin_kernel_time.fetch_add(time * 1000);
}

void OpenCLPlatform::copy(device_id dev_src, const void* src, int64_t offset_src, device_id dev_dst, void* dst, int64_t offset_dst, int64_t size) {
}

void OpenCLPlatform::copy_from_host(const void* src, int64_t offset_src, device_id dev_dst, void* dst, int64_t offset_dst, int64_t size) {
}

void OpenCLPlatform::copy_to_host(device_id dev_src, const void* src, int64_t offset_src, void* dst, int64_t offset_dst, int64_t size) {
}

int OpenCLPlatform::dev_count() {
    return devices_.size();
}
