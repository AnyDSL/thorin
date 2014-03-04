#include "cu_runtime.h"

#include <stdlib.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <unordered_map>


// define dim3
struct dim3 {
    unsigned int x, y, z;
    #if defined(__cplusplus)
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    #endif /* __cplusplus */
};

typedef struct dim3 dim3;
// define dim3

// global variables ...
typedef struct mem_ {
    size_t elem;
    size_t width;
    size_t height;
} mem_;
std::unordered_map<void*, mem_> host_mems_;
std::unordered_map<CUdeviceptr, void*> dev_mems_;
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction cuFunction;
CUtexref cuTexture;
void **cuArgs;
int cuArgIdx, cuArgIdxMax;
dim3 cuDimProblem, cuDimBlock;


#define checkErrNvvm(err, name) __checkNvvmErrors (err, name, __FILE__, __LINE__)
#define checkErrDrv(err, name)  __checkCudaErrors (err, name, __FILE__, __LINE__)

std::string getCUDAErrorCodeStrDrv(CUresult errorCode) {
    const char *errorName;
    const char *errorString;
    cuGetErrorName(errorCode, &errorName);
    cuGetErrorString(errorCode, &errorString);
    return std::string(errorName) + ": " + std::string(errorString);
}

inline void __checkCudaErrors(CUresult err, const char *name, const char *file, const int line) {
    if (CUDA_SUCCESS != err) {
        fprintf(stderr, "checkErrDrv(%s) Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                name, err, getCUDAErrorCodeStrDrv(err).c_str(), file, line);
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
    CUjit_target_enum target_cc = CU_TARGET_COMPUTE_20;

    const int errorLogSize = 10240;
    char errorLogBuffer[errorLogSize] = {0};

    int num_options = 2;
    CUjit_option options[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_TARGET };
    void *optionValues[] = { (void *)errorLogBuffer, (void *)errorLogSize, (void *)target_cc };

    // load ptx source
    err = cuModuleLoadDataEx(&cuModule, ptx, num_options, options, optionValues);
    if (err != CUDA_SUCCESS) {
        std::cerr << "Error log: " << errorLogBuffer << std::endl;
    }
    checkErrDrv(err, "cuModuleLoadDataEx()");

    // get function entry point
    err = cuModuleGetFunction(&cuFunction, cuModule, kernel_name);
    checkErrDrv(err, "cuModuleGetFunction()");
}


// load ll intermediate and compile to ptx
void load_kernel(size_t dev, const char *file_name, const char *kernel_name) {
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

    int num_options = 1;
    const char *options[2];
    options[0] = "-arch=compute_20";
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

void get_tex_ref(size_t dev, const char *name) {
    CUresult err = CUDA_SUCCESS;

    err = cuModuleGetTexRef(&cuTexture, cuModule, name);
    checkErrDrv(err, "cuModuleGetTexRef()");
}

void bind_tex(size_t dev, CUdeviceptr mem, CUarray_format format) {
    void *host = dev_mems_[mem];
    mem_ info = host_mems_[host];
    checkErrDrv(cuTexRefSetFormat(cuTexture, format, 1), "cuTexRefSetFormat()");
    checkErrDrv(cuTexRefSetFlags(cuTexture, CU_TRSF_READ_AS_INTEGER), "cuTexRefSetFlags()");
    checkErrDrv(cuTexRefSetAddress(0, cuTexture, mem, info.elem * info.width * info.height), "cuTexRefSetAddress()");
}

CUdeviceptr malloc_memory(size_t dev, void *host) {
    CUresult err = CUDA_SUCCESS;
    CUdeviceptr mem;
    mem_ info = host_mems_[host];

    err = cuMemAlloc(&mem, info.elem * info.width * info.height);
    checkErrDrv(err, "cuMemAlloc()");
    dev_mems_[mem] = host;

    return mem;
}

void free_memory(size_t dev, CUdeviceptr mem) {
    CUresult err = CUDA_SUCCESS;

    err = cuMemFree(mem);
    checkErrDrv(err, "cuMemFree()");
    dev_mems_.erase(mem);
}

void write_memory(size_t dev, CUdeviceptr mem, void *host) {
    CUresult err = CUDA_SUCCESS;
    mem_ info = host_mems_[host];

    err = cuMemcpyHtoD(mem, host, info.elem * info.width * info.height);
    checkErrDrv(err, "cuMemcpyHtoD()");
}

void read_memory(size_t dev, CUdeviceptr mem, void *host) {
    CUresult err = CUDA_SUCCESS;
    mem_ info = host_mems_[host];

    err = cuMemcpyDtoH(host, mem, info.elem * info.width * info.height);
    checkErrDrv(err, "cuMemcpyDtoH()");
}

void synchronize(size_t dev) {
    CUresult err = CUDA_SUCCESS;

    err = cuCtxSynchronize();
    checkErrDrv(err, "cuCtxSynchronize()");
}


void set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) {
    cuDimProblem.x = size_x;
    cuDimProblem.y = size_y;
    cuDimProblem.z = size_z;
}


void set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) {
    cuDimBlock.x = size_x;
    cuDimBlock.y = size_y;
    cuDimBlock.z = size_z;
}


void set_kernel_arg(size_t dev, void *host) {
    cuArgIdx++;
    if (cuArgIdx > cuArgIdxMax) {
        cuArgs = (void **)realloc(cuArgs, sizeof(void *)*cuArgIdx);
        cuArgIdxMax = cuArgIdx;
    }
    cuArgs[cuArgIdx-1] = (void *)malloc(sizeof(void *));
    cuArgs[cuArgIdx-1] = host;
}


void launch_kernel(size_t dev, const char *kernel_name) {
    CUresult err = CUDA_SUCCESS;
    CUevent start, end;
    unsigned int event_flags = CU_EVENT_DEFAULT;
    float time;
    std::string error_string = "cuLaunchKernel(";
    error_string += kernel_name;
    error_string += ")";

    // compute launch configuration
    dim3 grid;
    grid.x = cuDimProblem.x / cuDimBlock.x;
    grid.y = cuDimProblem.y / cuDimBlock.y;
    grid.z = cuDimProblem.z / cuDimBlock.z;

    cuEventCreate(&start, event_flags);
    cuEventCreate(&end, event_flags);
    cuEventRecord(start, 0);

    // launch the kernel
    err = cuLaunchKernel(cuFunction, grid.x, grid.y, grid.z, cuDimBlock.x, cuDimBlock.y, cuDimBlock.z, 0, NULL, cuArgs, NULL);
    checkErrDrv(err, error_string.c_str());
    err = cuCtxSynchronize();
    checkErrDrv(err, error_string.c_str());

    cuEventRecord(end, 0);
    cuEventSynchronize(end);
    cuEventElapsedTime(&time, start, end);

    cuEventDestroy(start);
    cuEventDestroy(end);

    std::cerr << "Kernel timing for '" << kernel_name << "' (" << cuDimBlock.x*cuDimBlock.y << ": " << cuDimBlock.x << "x" << cuDimBlock.y << "): " << time << "(ms)" << std::endl;

    // reset argument index
    cuArgIdx = 0;
}


// NVVM wrappers
CUdeviceptr nvvm_malloc_memory(size_t dev, void *host) { return malloc_memory(dev, host); }
void nvvm_free_memory(size_t dev, CUdeviceptr mem) { free_memory(dev, mem); }

void nvvm_write_memory(size_t dev, CUdeviceptr mem, void *host) { write_memory(dev, mem, host); }
void nvvm_read_memory(size_t dev, CUdeviceptr mem, void *host) { read_memory(dev, mem, host); }

void nvvm_load_kernel(size_t dev, const char *file_name, const char *kernel_name) { load_kernel(dev, file_name, kernel_name); }

void nvvm_set_kernel_arg(size_t dev, void *param) { set_kernel_arg(dev, param); }
void nvvm_set_kernel_arg_map(size_t dev, void *param) {
    CUdeviceptr mem = dev_mems2_[param];
    set_kernel_arg(dev, (void *)mem);
}
void nvvm_set_kernel_arg_tex(size_t dev, CUdeviceptr mem, char *name, CUarray_format format) {
    get_tex_ref(dev, name);
    bind_tex(dev, mem, format);
}
void nvvm_set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) { set_problem_size(dev, size_x, size_y, size_z); }
void nvvm_set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) { set_config_size(dev, size_x, size_y, size_z); }

void nvvm_launch_kernel(size_t dev, const char *kernel_name) { launch_kernel(dev, kernel_name); }
void nvvm_synchronize(size_t dev) { synchronize(dev); }

// helper functions
void *array(size_t elem_size, size_t width, size_t height) {
    void *mem = malloc(elem_size * width * height);
    host_mems_[mem] = {elem_size, width, height};
    return mem;
}
void *map_memory(size_t dev, size_t type_, void *from, size_t ox, size_t oy, size_t oz, size_t sx, size_t sy, size_t sz) {
    assert(oz==0 && sz==0 && "3D memory not yet supported");
    return (void *)malloc_memory(dev, from);
}
float random_val(int max) {
    return ((float)random() / RAND_MAX) * max;
}

int main(int argc, char *argv[]) {
    init_cuda();

    return main_impala();
}

