#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_platform.h"
#include "runtime.h"

#ifndef LIBDEVICE_DIR
#define LIBDEVICE_DIR ""
#endif
#ifndef KERNEL_DIR
#define KERNEL_DIR ""
#endif

#define check_dev(dev) __check_device(dev)
#define checkErrNvvm(err, name)  checkNvvmErrors  (err, name, __FILE__, __LINE__)
#define checkErrNvrtc(err, name) checkNvrtcErrors (err, name, __FILE__, __LINE__)
#define checkErrDrv(err, name)   checkCudaErrors  (err, name, __FILE__, __LINE__)

inline std::string cudaErrorString(CUresult errorCode) {
    const char* error_name;
    const char* error_string;
    cuGetErrorName(errorCode, &error_name);
    cuGetErrorString(errorCode, &error_string);
    return std::string(error_name) + ": " + std::string(error_string);
}

void CudaPlatform::checkCudaErrors(CUresult err, const char* name, const char* file, const int line) {
    if (CUDA_SUCCESS != err) {
        runtime_->error("Driver API function ", name, " (", err, ")",
                        " [file ", file, ", line ", line ,"]: ",
                        cudaErrorString(err));
    }
}

void CudaPlatform::checkNvvmErrors(nvvmResult err, const char* name, const char* file, const int line) {
    if (NVVM_SUCCESS != err) {
        runtime_->error("NVVM API function ", name, " (", err, ")",
                        " [file ", file, ", line ", line ,"]: ",
                        nvvmGetErrorString(err));
    }
}

#if CUDA_VERSION >= 7000
void CudaPlatform::checkNvrtcErrors(nvrtcResult err, const char* name, const char* file, const int line) {
    if (NVRTC_SUCCESS != err) {
        runtime_->error("NVRTC API function ", name, " (", err, ")",
                        " [file ", file, ", line ", line ,"]: ",
                        nvrtcGetErrorString(err));
    }
}
#endif

CudaPlatform::CudaPlatform(Runtime* runtime)
    : Platform(runtime)
{
    int device_count = 0, driver_version = 0, nvvm_major = 0, nvvm_minor = 0;

    setenv("CUDA_CACHE_DISABLE", "1", 1);

    CUresult err = cuInit(0);
    checkErrDrv(err, "cuInit()");

    err = cuDeviceGetCount(&device_count);
    checkErrDrv(err, "cuDeviceGetCount()");

    err = cuDriverGetVersion(&driver_version);
    checkErrDrv(err, "cuDriverGetVersion()");

    nvvmResult errNvvm = nvvmVersion(&nvvm_major, &nvvm_minor);
    checkErrNvvm(errNvvm, "nvvmVersion()");

    runtime_->log("CUDA Driver Version ", driver_version/1000, ".", (driver_version%100)/10);
    #if CUDA_VERSION >= 7000
    int nvrtc_major = 0, nvrtc_minor = 0;
    nvrtcResult errNvrtc = nvrtcVersion(&nvrtc_major, &nvrtc_minor);
    checkErrNvrtc(errNvrtc, "nvrtcVersion()");
    runtime_->log("NVRTC Version ", nvrtc_major, ".", nvrtc_minor);
    #endif
    runtime_->log("NVVM Version ", nvvm_major, ".", nvvm_minor);

    devices_.resize(device_count);

    // Initialize devices
    for (int i = 0; i < device_count; ++i) {
        char name[128];

        err = cuDeviceGet(&devices_[i].dev, i);
        checkErrDrv(err, "cuDeviceGet()");
        err = cuDeviceGetName(name, 128, devices_[i].dev);
        checkErrDrv(err, "cuDeviceGetName()");
        err = cuDeviceGetAttribute(&devices_[i].compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devices_[i].dev);
        checkErrDrv(err, "cuDeviceGetAttribute()");
        err = cuDeviceGetAttribute(&devices_[i].compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devices_[i].dev);
        checkErrDrv(err, "cuDeviceGetAttribute()");

        runtime_->log("Device ", i, ": ", name, ", ",
                      "Compute capability: ", devices_[i].compute_major,
                      ".", devices_[i].compute_minor);

        err = cuCtxCreate(&devices_[i].ctx, 0, devices_[i].dev);
        checkErrDrv(err, "cuCtxCreate()");
    }
}

void* CudaPlatform::alloc(device_id dev, int64_t size) {
    const int index = runtime_->device(dev).index;

    cuCtxPushCurrent(devices_[index].ctx);
    CUdeviceptr mem;

    CUresult err = cuMemAlloc(&mem, size);
    checkErrDrv(err, "cuMemAlloc()");

    cuCtxPopCurrent(NULL);
    return (void*)mem;
}

void CudaPlatform::release(void* ptr) {
    auto mem = runtime_->memory_info(ptr);
    const int index = runtime_->device(mem.dev).index;
    cuCtxPushCurrent(devices_[index].ctx);
    CUresult err = cuMemFree((CUdeviceptr)ptr);
    checkErrDrv(err, "cuMemFree()");
    cuCtxPopCurrent(NULL);
}

void* CudaPlatform::map(void* ptr, int64_t offset, int64_t size) {
    assert(0 && "Not implemented");
}

void CudaPlatform::unmap(void* view) {
    assert(0 && "Not implemented");
}

void CudaPlatform::set_block_size(device_id dev, unsigned x, unsigned y, unsigned z) {
    const int index = runtime_->device(dev).index;
    auto& block = devices_[index].block;
    block.x = x;
    block.y = y;
    block.z = z;
}

void CudaPlatform::set_grid_size(device_id dev, unsigned x, unsigned y, unsigned z) {
    const int index = runtime_->device(dev).index;
    auto& grid = devices_[index].block;
    grid.x = x;
    grid.y = y;
    grid.z = z;
}

void CudaPlatform::set_arg(device_id dev, int i, void* ptr) {
    const int index = runtime_->device(dev).index;
    auto& args = devices_[index].kernel_args;
    args.resize(i + 1);
    args[i] = ptr;
}

void CudaPlatform::launch_kernel(device_id dev) {
    const int index = runtime_->device(dev).index;
    auto& cuda_dev = devices_[index];
    cuLaunchKernel(cuda_dev.kernel,
                   cuda_dev.grid.x, cuda_dev.grid.y, cuda_dev.grid.z,
                   cuda_dev.block.x, cuda_dev.block.y, cuda_dev.block.z,
                   0, nullptr, cuda_dev.kernel_args.data(), nullptr);
}

void CudaPlatform::load_kernel(device_id dev, const char* file, const char* name) {
    const int index = runtime_->device(dev).index;
    auto& mod_cache = devices_[index].modules;
    auto mod = mod_cache.find(file);
    if (mod == mod_cache.end()) {
        CUjit_target target_cc =
            (CUjit_target)(devices_[index].compute_major * 10 +
                           devices_[index].compute_minor);

        // Compile the given file
        auto ext = strrchr(file, '.');
        if (ext && !strcmp(ext + 1, "nvvm")) {
            //compile_nvvm(dev, file, target_cc);
        } else if (ext && !strcmp(ext + 1, "cu")) {
            //compile_cuda(dev, file, target_cc);
        } else {
            runtime_->error("Invalid kernel file extension");
        }
    }

    // Checks that the function exists
    auto& func_cache = devices_[index].functions;
    auto& func_map = func_cache[mod->second];
    auto func = func_map.find(name);
    if (func == func_map.end())
        runtime_->error("Function ", name, " is not present in ", file);

    devices_[index].kernel = func->second;
}

void CudaPlatform::copy(const void* src, void* dst) {
    CUdeviceptr src_mem = (CUdeviceptr)src;
    CUdeviceptr dst_mem = (CUdeviceptr)dst;
    auto src_size = runtime_->memory_info(src).size;
    CUresult err = cuMemcpyDtoD(dst_mem, src_mem, src_size);
    checkErrDrv(err, "cuMemcpyDtoD()");
}

void CudaPlatform::copy_from_host(const void* src, void* dst) {
    CUdeviceptr dst_mem = (CUdeviceptr)dst;
    auto src_size = runtime_->memory_info(src).size;
    CUresult err = cuMemcpyHtoD(dst_mem, src, src_size);
    checkErrDrv(err, "cuMemcpyHtoD()");
}

void CudaPlatform::copy_to_host(const void* src, void* dst) {
    CUdeviceptr src_mem = (CUdeviceptr)src;
    auto src_size = runtime_->memory_info(src).size;
    CUresult err = cuMemcpyDtoH(dst, src_mem, src_size);
    checkErrDrv(err, "cuMemcpyDtoH()");
}

int CudaPlatform::dev_count() {
    return devices_.size();
}
