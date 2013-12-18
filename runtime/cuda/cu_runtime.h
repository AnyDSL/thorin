#ifndef __CUDA_RT_HPP__
#define __CUDA_RT_HPP__

#include <cuda.h>
#include <cuda_runtime.h>
#include <drvapi_error_string.h>
#include <nvvm.h>

#include <stdlib.h>

#include <fstream>
#include <iostream>

extern "C"
{

// global variables ...
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction cuFunction;
void **cuArgs;
int cuArgIdx, cuArgIdxMax;
dim3 cuDimProblem;


#define checkErrNvvm(err, name) __checkNvvmErrors (err, name, __FILE__, __LINE__)
#define checkErrDrv(err, name)  __checkCudaErrors (err, name, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char *name, const char *file, const int line) {
    if (CUDA_SUCCESS != err) {
        fprintf(stderr, "checkErrDrv(%s) Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                name, err, getCudaDrvErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
inline void __checkNvvmErrors(nvvmResult err, const char *name, const char *file, const int line) {
    if (NVVM_SUCCESS != err) {
        fprintf(stderr, "checkErrNvvm(%s) NVVM API error = %04d \"%s\" from file <%s>, line %i.\n",
                name, err, nvvmGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}


// initialize CUDA device
void init_cuda() {
    CUresult err = CUDA_SUCCESS;
    int device_count, driver_version = 0, nvvm_major = 0, nvvm_minor = 0;

    setenv("CUDA_CACHE_DISABLE", "1", 1);

    err = cuInit(0);
    checkErrDrv(err, "cuInit()");

    err = cuDeviceGetCount(&device_count);
    checkErrDrv(err, "cuDeviceGetCount()");

    err = cuDriverGetVersion(&driver_version);
    checkErrDrv(err, "cuDriverGetVersion()");

    nvvmResult errNvvm = nvvmVersion(&nvvm_major, &nvvm_minor);
    checkErrNvvm(errNvvm, "nvvmVersion()");

    std::cerr << "CUDA Driver Version " << driver_version/1000 << "." << (driver_version%100)/10 << std::endl;
    std::cerr << "NVVM Version " << nvvm_major << "." << nvvm_minor << std::endl;

    for (int i=0; i<device_count; i++) {
        int major, minor;
        char name[100];

        err = cuDeviceGet(&cuDevice, i);
        checkErrDrv(err, "cuDeviceGet()");
        err = cuDeviceGetName(name, 100, cuDevice);
        checkErrDrv(err, "cuDeviceGetName()");
        err = cuDeviceComputeCapability(&major, &minor, cuDevice);
        checkErrDrv(err, "cuDeviceComputeCapability()");

        if (i==0) std::cerr << "  [*] ";
        else std::cerr << "  [ ] ";
        std::cerr << "Name: " << name << std::endl;
        std::cerr << "      Compute capability: " << major << "." << minor << std::endl;
    }
    err = cuDeviceGet(&cuDevice, 0);
    checkErrDrv(err, "cuDeviceGet()");

    // create context
    err = cuCtxCreate(&cuContext, 0, cuDevice);
    checkErrDrv(err, "cuCtxCreate()");

    // initialize cuArgs
    cuArgs = (void **)malloc(sizeof(void *));
    cuArgIdx = 0;
    cuArgIdxMax = 1;
}


// load ptx assembly, create a module and kernel
void create_module_kernel(const char *ptx, const char *kernel_name) {
    CUresult err = CUDA_SUCCESS;
    //CUjit_target_enum target_cc;

    const int errorLogSize = 10240;
    char errorLogBuffer[errorLogSize] = {0};

    CUjit_option options[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES}; //{, CU_JIT_TARGET };
    void *optionValues[] = { (void *)errorLogBuffer, (void *)errorLogSize}; //{, (void *)target_cc };

    // load ptx source
    err = cuModuleLoadDataEx(&cuModule, ptx, 2, options, optionValues);
    if (err != CUDA_SUCCESS) {
        std::cerr << "Error log: " << errorLogBuffer << std::endl;
    }
    checkErrDrv(err, "cuModuleLoadDataEx()");

    // get function entry point
    err = cuModuleGetFunction(&cuFunction, cuModule, kernel_name);
    checkErrDrv(err, "cuModuleGetFunction()");
}


// load ll intermediate and compile to ptx
void load_kernel(const char *file_name, const char *kernel_name) {
    nvvmResult err;
    nvvmProgram program;
    size_t PTXSize;
    char *PTX = NULL;

    std::ifstream srcFile(file_name);
    if (!srcFile.is_open()) {
        std::cerr << "ERROR: Can't open LL source file '" << file_name << "'!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string llString(std::istreambuf_iterator<char>(srcFile),
            (std::istreambuf_iterator<char>()));

    err = nvvmCreateProgram(&program);
    checkErrNvvm(err, "nvvmCreateProgram()");

    err = nvvmAddModuleToProgram(program, llString.c_str(), llString.length(), file_name);
    checkErrNvvm(err, "nvvmAddModuleToProgram()");

    int num_options = 0;
    const char *options[2];
    options[0] = "-arch=compute_30";
    options[1] = "-ftz=1";

    err = nvvmCompileProgram(program, num_options, options);
    if (err != NVVM_SUCCESS) {
        size_t log_size;
        nvvmGetProgramLogSize(program, &log_size);
        char *error_log = (char*)malloc(log_size);
        nvvmGetProgramLog(program, error_log);
        fprintf(stderr, "Error log: %s\n", error_log);
        free(error_log);
    }
    checkErrNvvm(err, "nvvmAddModuleToProgram()");

    err = nvvmGetCompiledResultSize(program, &PTXSize);
    checkErrNvvm(err, "nvvmGetCompiledResultSize()");

    PTX = (char*)malloc(PTXSize);
    err = nvvmGetCompiledResult(program, PTX);
    if (err != NVVM_SUCCESS) free(PTX);
    checkErrNvvm(err, "nvvmGetCompiledResult()");

    err = nvvmDestroyProgram(&program);
    if (err != NVVM_SUCCESS) free(PTX);
    checkErrNvvm(err, "nvvmDestroyProgram()");

    // compile ptx
    create_module_kernel(PTX, kernel_name);
}

CUdeviceptr malloc_memory(size_t size) {
    CUresult err = CUDA_SUCCESS;
    CUdeviceptr mem;

    err = cuMemAlloc(&mem, size * sizeof(float));
    checkErrDrv(err, "cuMemAlloc()");

    return mem;
}

void free_memory(CUdeviceptr mem) {
    CUresult err = CUDA_SUCCESS;

    err = cuMemFree(mem);
    checkErrDrv(err, "cuMemFree()");
}

void write_memory(CUdeviceptr dev, void *host, size_t size) {
    CUresult err = CUDA_SUCCESS;

    err = cuMemcpyHtoD(dev, host, size * sizeof(float));
    checkErrDrv(err, "cuMemcpyHtoD()");
}

void read_memory(CUdeviceptr dev, void *host, size_t size) {
    CUresult err = CUDA_SUCCESS;

    err = cuMemcpyDtoH(host, dev, size * sizeof(float));
    checkErrDrv(err, "cuMemcpyDtoH()");
}

void synchronize() {
    CUresult err = CUDA_SUCCESS;

    err = cuCtxSynchronize();
    checkErrDrv(err, "cuCtxSynchronize()");
}


// set problem size
void set_problem_size(size_t size_x, size_t size_y, size_t size_z) {
    cuDimProblem.x = size_x;
    cuDimProblem.y = size_y;
    cuDimProblem.z = size_z;
}


// set kernel argument
void set_kernel_arg(void *host) {
    cuArgIdx++;
    if (cuArgIdx > cuArgIdxMax) {
        cuArgs = (void **)realloc(cuArgs, sizeof(void *)*cuArgIdx);
        cuArgIdxMax = cuArgIdx;
    }
    cuArgs[cuArgIdx-1] = (void *)malloc(sizeof(void *));
    cuArgs[cuArgIdx-1] = host;
}


// launch kernel
void launch_kernel(const char *kernel_name) {
    CUresult err = CUDA_SUCCESS;
    CUevent start, end;
    unsigned int event_flags = CU_EVENT_DEFAULT;
    float time;
    std::string error_string = "cuLaunchKernel(";
    error_string += kernel_name;
    error_string += ")";

    // compute launch configuration
    dim3 block(128, 1, 1);
    dim3 grid;
    grid.x = cuDimProblem.x / block.x;
    grid.y = cuDimProblem.y / block.y;
    grid.z = cuDimProblem.z / block.z;

    cuEventCreate(&start, event_flags);
    cuEventCreate(&end, event_flags);
    cuEventRecord(start, 0);

    // launch the kernel
    err = cuLaunchKernel(cuFunction, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, cuArgs, NULL);
    checkErrDrv(err, error_string.c_str());
    err = cuCtxSynchronize();
    checkErrDrv(err, error_string.c_str());

    cuEventRecord(end, 0);
    cuEventSynchronize(end);
    cuEventElapsedTime(&time, start, end);

    cuEventDestroy(start);
    cuEventDestroy(end);

    std::cerr << "Kernel timing for '" << kernel_name << "' (" << block.x*block.y << ": " << block.x << "x" << block.y << "): " << time << "(ms)" << std::endl;

    // reset argument index
    cuArgIdx = 0;
}

float *array(size_t num_elems) {
    return (float *)malloc(sizeof(float)*num_elems);
}
float random_val(int max) {
    return ((float)random() / RAND_MAX) * max;
}
CUdeviceptr nvvm_malloc_memory(size_t size) { return malloc_memory(size); }
void nvvm_free_memory(CUdeviceptr mem) { free_memory(mem); }

void nvvm_write_memory(CUdeviceptr dev, void *host, size_t size) { write_memory(dev, host, size); }
void nvvm_read_memory(CUdeviceptr dev, void *host, size_t size) { read_memory(dev, host, size); }

void nvvm_load_kernel(const char *file_name, const char *kernel_name) { load_kernel(file_name, kernel_name); }
void nvvm_set_kernel_arg(void *host) { set_kernel_arg(host); }
void nvvm_set_problem_size(size_t size_x, size_t size_y, size_t size_z) { set_problem_size(size_x, size_y, size_z); }

void nvvm_launch_kernel(const char *kernel_name) { launch_kernel(kernel_name); }
void nvvm_synchronize() { synchronize(); }

extern int main_impala();
int main(int argc, char *argv[]) {
    init_cuda();

    return main_impala();
}
}


#endif  // __CUDA_RT_HPP__

