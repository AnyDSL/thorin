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
    // Get OpenCL platform count
    cl_uint num_platforms, num_devices;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    checkErr(err, "clGetPlatformIDs()");

    runtime_->log("Number of available Platforms: ", num_platforms);

    cl_platform_id* platforms = new cl_platform_id[num_platforms];

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    checkErr(err, "clGetPlatformIDs()");

    // get platform info for each platform
    for (size_t i=0; i<num_platforms; ++i) {
        auto platform = platforms[i];

        char buffer[1024];
        err  = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buffer), &buffer, NULL);
        runtime_->log("  Platform Name: ", buffer);
        err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buffer), &buffer, NULL);
        runtime_->log("  Platform Vendor: ", buffer);
        err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(buffer), &buffer, NULL);
        runtime_->log("  Platform Version: ", buffer);
        checkErr(err, "clGetPlatformInfo()");

        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        checkErr(err, "clGetDeviceIDs()");

        cl_device_id* devices = new cl_device_id[num_devices];
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, &num_devices);
        checkErr(err, "clGetDeviceIDs()");

        // get device info for each device
        for (size_t j=0; j<num_devices; ++j) {
            auto device = devices[j];
            cl_device_type dev_type;
            cl_uint device_vendor_id;

            err  = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer), &buffer, NULL);
            err |= clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
            std::string type_str;
            if (dev_type & CL_DEVICE_TYPE_CPU)         type_str  = "CL_DEVICE_TYPE_CPU";
            if (dev_type & CL_DEVICE_TYPE_GPU)         type_str  = "CL_DEVICE_TYPE_GPU";
            if (dev_type & CL_DEVICE_TYPE_ACCELERATOR) type_str  = "CL_DEVICE_TYPE_ACCELERATOR";
            #ifdef CL_VERSION_1_2
            if (dev_type & CL_DEVICE_TYPE_CUSTOM)      type_str  = "CL_DEVICE_TYPE_CUSTOM";
            #endif
            if (dev_type & CL_DEVICE_TYPE_DEFAULT)     type_str += "|CL_DEVICE_TYPE_DEFAULT";
            runtime_->log("  (", devices_.size(), ") Device Name: ", buffer, " (", type_str, ")");
            err |= clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(buffer), &buffer, NULL);
            err |= clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR_ID, sizeof(device_vendor_id), &device_vendor_id, NULL);
            runtime_->log("      Device Vendor: ", buffer, " (ID: ", device_vendor_id, ")");
            err |= clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(buffer), &buffer, NULL);
            runtime_->log("      Device OpenCL Version: ", buffer);
            err |= clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(buffer), &buffer, NULL);
            runtime_->log("      Device Driver Version: ", buffer);
            err |= clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, sizeof(buffer), &buffer, NULL);
            //runtime_->log("      Device Extensions: ", buffer);
            std::string extensions(buffer);
            size_t found = extensions.find("cl_khr_spir");
            bool has_spir = found!=std::string::npos;
            runtime_->log("      Device SPIR Support: ", has_spir);
            #ifdef CL_DEVICE_SPIR_VERSIONS
            if (has_spir) {
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_SPIR_VERSIONS , sizeof(buffer), &buffer, NULL);
                runtime_->log(" (Version: ", buffer, ")");
            }
            #endif

            cl_bool has_unified = false;
            #ifdef CL_VERSION_2_0
            cl_device_svm_capabilities caps;
            err |= clGetDeviceInfo(devices[j], CL_DEVICE_SVM_CAPABILITIES, sizeof(caps), &caps, NULL);
            runtime_->log("      Device SVM capabilities:");
            if (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) runtime_->log(" CL_DEVICE_SVM_COARSE_GRAIN_BUFFER");
            else                                          runtime_->log(" n/a");
            if (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)   runtime_->log(" CL_DEVICE_SVM_FINE_GRAIN_BUFFER");
            if (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)   runtime_->log(" CL_DEVICE_SVM_FINE_GRAIN_SYSTEM");
            if (caps & CL_DEVICE_SVM_ATOMICS)             runtime_->log(" CL_DEVICE_SVM_ATOMICS");
            // TODO: SVM is inconsistent with unified memory in OpenCL 1.2
            has_unified = (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) || (dev_type & CL_DEVICE_TYPE_CPU);
            #else
            err |= clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(has_unified), &has_unified, NULL);
            runtime_->log("      Device Host Unified Memory: ", has_unified);
            #endif
            checkErr(err, "clGetDeviceInfo()");

            auto dev = devices_.size();
            devices_.resize(dev + 1);
            devices_[dev].platform = platform;
            devices_[dev].dev = device;

            // create context
            cl_int err = CL_SUCCESS;
            cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
            devices_[dev].ctx = clCreateContext(cprops, 1, &devices_[dev].dev, NULL, NULL, &err);
            checkErr(err, "clCreateContext()");

            // create command queue
            #ifdef CL_VERSION_2_0
            cl_queue_properties cprops[3] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
            devices_[dev].queue = clCreateCommandQueueWithProperties(devices_[dev].ctx, devices_[dev].dev, cprops, &err);
            checkErr(err, "clCreateCommandQueueWithProperties()");
            #else
            devices_[dev].queue = clCreateCommandQueue(devices_[dev].ctx, devices_[dev].dev, CL_QUEUE_PROFILING_ENABLE, &err);
            checkErr(err, "clCreateCommandQueue()");
            #endif
        }
        delete[] devices;
    }
    delete[] platforms;
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
