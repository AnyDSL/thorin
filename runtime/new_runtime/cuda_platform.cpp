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
#define LIBDEVICE_DIR "/usr/local/cuda/nvvm/libdevice/"
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
    cuCtxPushCurrent(devices_[dev].ctx);
    CUdeviceptr mem;

    CUresult err = cuMemAlloc(&mem, size);
    checkErrDrv(err, "cuMemAlloc()");

    cuCtxPopCurrent(NULL);
    return (void*)mem;
}

void CudaPlatform::release(device_id dev, void* ptr, int64_t size) {
    cuCtxPushCurrent(devices_[dev].ctx);
    CUresult err = cuMemFree((CUdeviceptr)ptr);
    checkErrDrv(err, "cuMemFree()");
    cuCtxPopCurrent(NULL);
}

void* CudaPlatform::map(void* ptr, int64_t offset, int64_t size) {
    return (void*)((CUdeviceptr)ptr + offset);
}

void CudaPlatform::unmap(void* view) {}

void CudaPlatform::set_block_size(device_id dev, int32_t x, int32_t y, int32_t z) {
    auto& block = devices_[dev].block;
    block.x = x;
    block.y = y;
    block.z = z;
}

void CudaPlatform::set_grid_size(device_id dev, int32_t x, int32_t y, int32_t z) {
    auto& grid = devices_[dev].block;
    grid.x = x;
    grid.y = y;
    grid.z = z;
}

void CudaPlatform::set_kernel_arg(device_id dev, int32_t arg, void* ptr, int32_t size) {
    auto& args = devices_[dev].kernel_args;
    auto& vals = devices_[dev].kernel_vals;
    args.resize(arg + 1);
    vals.resize(arg + 1);
    vals[arg] = ptr;
    args[arg] = (void*)&vals[arg];
}

void CudaPlatform::load_kernel(device_id dev, const char* file, const char* name) {
    auto& mod_cache = devices_[dev].modules;
    auto mod_it = mod_cache.find(file);
    CUmodule mod;
    if (mod_it == mod_cache.end()) {
        CUjit_target target_cc =
            (CUjit_target)(devices_[dev].compute_major * 10 +
                           devices_[dev].compute_minor);

        // Compile the given file
        auto ext = strrchr(file, '.');
        if (ext && !strcmp(ext + 1, "nvvm")) {
            compile_nvvm(dev, file, target_cc);
        } else if (ext && !strcmp(ext + 1, "cu")) {
            compile_cuda(dev, file, target_cc);
        } else {
            runtime_->error("Invalid kernel file extension");
        }

        mod = mod_cache[file];
    } else {
        mod = mod_it->second;
    }

    // Checks that the function exists
    auto& func_cache = devices_[dev].functions;
    auto& func_map = func_cache[mod];
    auto func_it = func_map.find(name);
    if (func_it == func_map.end()) {
        CUfunction func;
        CUresult err = cuModuleGetFunction(&func, mod, name);
        if (err != CUDA_SUCCESS)
            runtime_->error("Function '", name, "' is not present in '", file, "'");
        func_map.emplace(name, func);
        devices_[dev].kernel = func;
    } else {
        devices_[dev].kernel = func_it->second;
    }
}

void CudaPlatform::launch_kernel(device_id dev) {
    auto& cuda_dev = devices_[dev];
    cuLaunchKernel(cuda_dev.kernel,
                   cuda_dev.grid.x, cuda_dev.grid.y, cuda_dev.grid.z,
                   cuda_dev.block.x, cuda_dev.block.y, cuda_dev.block.z,
                   0, nullptr, cuda_dev.kernel_args.data(), nullptr);
}

void CudaPlatform::synchronize(device_id dev) {
    cuCtxPushCurrent(devices_[dev].ctx);
    CUresult err = cuCtxSynchronize();
    checkErrDrv(err, "cuCtxSynchronize()");
    cuCtxPopCurrent(NULL);
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

void CudaPlatform::compile_nvvm(device_id dev, const char* file_name, CUjit_target target_cc) {
    // Select libdevice module according to documentation
    std::string libdevice_file_name;
    switch (target_cc) {
        #if CUDA_VERSION == 6050
        case CU_TARGET_COMPUTE_37:
        #endif
        default:
            assert(false && "Unsupported compute capability");
        case CU_TARGET_COMPUTE_20:
        case CU_TARGET_COMPUTE_21:
        case CU_TARGET_COMPUTE_32:
            libdevice_file_name = "libdevice.compute_20.10.bc"; break;
        case CU_TARGET_COMPUTE_30:
        case CU_TARGET_COMPUTE_50:
        #if CUDA_VERSION >= 7000
        case CU_TARGET_COMPUTE_52:
        #endif
            libdevice_file_name = "libdevice.compute_30.10.bc"; break;
        case CU_TARGET_COMPUTE_35:
        #if CUDA_VERSION >= 7000
        case CU_TARGET_COMPUTE_37:
        #endif
            libdevice_file_name = "libdevice.compute_35.10.bc"; break;
    }

    std::ifstream libdevice_file(std::string(LIBDEVICE_DIR) + libdevice_file_name);
    if (!libdevice_file.is_open())
        runtime_->error("Can't open libdevice source file '", libdevice_file_name, "'");

    std::string libdevice_string = std::string(std::istreambuf_iterator<char>(libdevice_file), (std::istreambuf_iterator<char>()));

    std::ifstream src_file(std::string(KERNEL_DIR) + file_name);
    if (!src_file.is_open())
        runtime_->error("Can't open NVVM source file '", KERNEL_DIR, file_name, "'");

    std::string src_string = std::string(std::istreambuf_iterator<char>(src_file), (std::istreambuf_iterator<char>()));

    nvvmProgram program;
    nvvmResult err = nvvmCreateProgram(&program);
    checkErrNvvm(err, "nvvmCreateProgram()");

    err = nvvmAddModuleToProgram(program, libdevice_string.c_str(), libdevice_string.length(), libdevice_file_name.c_str());
    checkErrNvvm(err, "nvvmAddModuleToProgram()");

    err = nvvmAddModuleToProgram(program, src_string.c_str(), src_string.length(), file_name);
    checkErrNvvm(err, "nvvmAddModuleToProgram()");

    std::string compute_arch("-arch=compute_" + std::to_string(target_cc));
    int num_options = 2;
    const char* options[3];
    options[0] = compute_arch.c_str();
    options[1] = "-opt=3";
    options[2] = "-g";

    err = nvvmCompileProgram(program, num_options, options);
    if (err != NVVM_SUCCESS) {
        size_t log_size;
        nvvmGetProgramLogSize(program, &log_size);
        std::string error_log(log_size, '\0');
        nvvmGetProgramLog(program, &error_log[0]);
        runtime_->error("Compilation error: ", error_log);
    }

    checkErrNvvm(err, "nvvmCompileProgram()");

    size_t ptx_size;
    err = nvvmGetCompiledResultSize(program, &ptx_size);
    checkErrNvvm(err, "nvvmGetCompiledResultSize()");

    std::string ptx(ptx_size, '\0');
    err = nvvmGetCompiledResult(program, &ptx[0]);
    checkErrNvvm(err, "nvvmGetCompiledResult()");

    err = nvvmDestroyProgram(&program);
    checkErrNvvm(err, "nvvmDestroyProgram()");

    create_module(dev, file_name, target_cc, ptx.c_str());
}

#if CUDA_VERSION >= 7000
void CudaPlatform::compile_cuda(device_id dev, const char* file_name, CUjit_target target_cc) {
    std::ifstream src_file(std::string(KERNEL_DIR) + file_name);
    if (!src_file.is_open())
        runtime_->error("Can't open CUDA source file '", KERNEL_DIR, file_name, "'");

    std::string src_string = std::string(std::istreambuf_iterator<char>(src_file), (std::istreambuf_iterator<char>()));

    nvrtcProgram program;
    nvrtcResult err = nvrtcCreateProgram(&program, src_string.c_str(), file_name, 0, NULL, NULL);
    checkErrNvrtc(err, "nvrtcCreateProgram()");

    std::string compute_arch("-arch=compute_" + std::to_string(target_cc));
    int num_options = 1;
    const char* options[3];
    options[0] = compute_arch.c_str();
    options[1] = "-G";
    options[2] = "-lineinfo";

    err = nvrtcCompileProgram(program, num_options, options);
    if (err != NVRTC_SUCCESS) {
        size_t log_size;
        nvrtcGetProgramLogSize(program, &log_size);
        std::string error_log(log_size, '\0');
        nvrtcGetProgramLog(program, &error_log[0]);
        runtime_->error("Compilation error: ", error_log);
    }
    checkErrNvrtc(err, "nvrtcCompileProgram()");

    size_t ptx_size;
    err = nvrtcGetPTXSize(program, &ptx_size);
    checkErrNvrtc(err, "nvrtcGetPTXSize()");

    std::string ptx(ptx_size, '\0');
    err = nvrtcGetPTX(program, &ptx[0]);
    checkErrNvrtc(err, "nvrtcGetPTX()");

    err = nvrtcDestroyProgram(&program);
    checkErrNvrtc(err, "nvrtcDestroyProgram()");

    create_module(dev, file_name, target_cc, ptx.c_str());
}
#else
#ifndef NVCC_BIN
#define NVCC_BIN "nvcc"
#endif
void CudaPlatform::compile_cuda(device_id dev, const char* file_name, CUjit_target target_cc) {
    target_cc = target_cc == CU_TARGET_COMPUTE_21 ? CU_TARGET_COMPUTE_20 : target_cc; // compute_21 does not exist for nvcc
    std::string ptx_filename = std::string(file_name) + ".ptx";
    std::string command = (NVCC_BIN " -O4 -ptx -arch=compute_") + std::to_string(target_cc) + " ";
    command += std::string(KERNEL_DIR) + file_name + " -o " + ptx_filename + " 2>&1";

    if (auto stream = popen(command.c_str(), "r")) {
        std::string log;
        char buffer[256];

        while (fgets(buffer, 256, stream))
            log += buffer;

        int exit_status = pclose(stream);
        if (!WEXITSTATUS(exit_status)) {
            runtime_->log(log);
        } else {
            runtime_->error("Compilation error: ", log);
        }
    } else {
        runtime_->error("Cannot run NVCC");
    }

    std::ifstream src_file(ptx_filename);
    if (!src_file.is_open())
        runtime_->error("Cannot open PTX source file '", ptx_filename, "'");

    std::string src_string(std::istreambuf_iterator<char>(src_file), (std::istreambuf_iterator<char>()));
    create_module(dev, file_name, target_cc, src_string.c_str());
}
#endif

void CudaPlatform::create_module(device_id dev, const char* file_name, CUjit_target target_cc, const void* ptx) {
    const unsigned opt_level = 3;
    const int error_log_size = 10240;
    const int num_options = 4;
    char error_log_buffer[error_log_size] = { 0 };

    CUjit_option options[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_TARGET, CU_JIT_OPTIMIZATION_LEVEL };
    void* option_values[]  = { (void*)error_log_buffer, (void*)(size_t)error_log_size, (void*)target_cc, (void*)(size_t)opt_level };

    // load ptx source
    runtime_->log("Compiling '", file_name, "' on CUDA device ", dev);
    CUmodule mod;
    CUresult err = cuModuleLoadDataEx(&mod, ptx, num_options, options, option_values);
    checkErrDrv(err, "cuModuleLoadDataEx()");

    devices_[dev].modules[file_name] = mod;
    
}
