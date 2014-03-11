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

const int num_devices_ = 3;

// global variables ...
enum mem_type {
    Global      = 0,
    Texture     = 1,
    Constant    = 2,
    Shared      = 3
};

typedef struct mem_ {
    size_t elem;
    size_t width;
    size_t height;
} mem_;
std::unordered_map<void*, mem_> host_mems_;


class Memory {
    private:
        size_t count_;
        mem_id new_id() { return count_++; }
        void insert(size_t dev, void *mem, mem_id id) {
            idtomem[dev][id] = mem;
            memtoid[dev][mem] = id;
        }
        typedef struct mapping_ {
            void *cpu;
            CUdeviceptr gpu;
            mem_type type;
            size_t ox, oy, oz;
            size_t sx, sy, sz;
            mem_id id;
        } mapping_;
        std::unordered_map<mem_id, mapping_> mmap[num_devices_];
        std::unordered_map<mem_id, void*> idtomem[num_devices_];
        std::unordered_map<void*, mem_id> memtoid[num_devices_];

    public:
        Memory() : count_(42) {}

    mem_id get_id(size_t dev, void *mem) { return memtoid[dev][mem]; }
    mem_id map_memory(size_t dev, void *from, CUdeviceptr to, mem_type type,
            size_t ox, size_t oy, size_t oz, size_t sx, size_t sy, size_t sz) {
        mem_id id = new_id();
        mapping_ mem_map = { from, to, type, ox, oy, oz, sx, sy, sz, id };
        mmap[dev][id] = mem_map;
        insert(dev, from, id);
        return id;
    }
    mem_id map_memory(size_t dev, void *from, CUdeviceptr to, mem_type type) {
        mem_ info = host_mems_[from];
        return map_memory(dev, from, to, type, 0, 0, 0, info.width, info.height, 0);
    }

    void *get_host_mem(size_t dev, mem_id id) { return mmap[dev][id].cpu; }
    CUdeviceptr &get_dev_mem(size_t dev, mem_id id) { return mmap[dev][id].gpu; }

    void remove(size_t dev, mem_id id) {
        void *mem = idtomem[dev][id];
        memtoid[dev].erase(mem);
        idtomem[dev].erase(id);
    }

    void read_memory(size_t dev, mem_id id) {
        read_memory_size(dev, id, mmap[dev][id].cpu,
                mmap[dev][id].ox, mmap[dev][id].oy, mmap[dev][id].oz,
                mmap[dev][id].sx, mmap[dev][id].sy, mmap[dev][id].sz);
    }
    void write_memory(size_t dev, mem_id id, void *host) {
        assert(host==mmap[dev][id].cpu && "invalid host memory");
        write_memory_size(dev, id, host,
                mmap[dev][id].ox, mmap[dev][id].oy, mmap[dev][id].oz,
                mmap[dev][id].sx, mmap[dev][id].sy, mmap[dev][id].sz);
    }
};
Memory mem_manager;


CUdevice cuDevices[num_devices_];
CUcontext cuContexts[num_devices_];
CUmodule cuModules[num_devices_];
CUfunction cuFunctions[num_devices_];
CUtexref cuTextures[num_devices_];
void **cuArgs[num_devices_];
int cuArgIdx[num_devices_], cuArgIdxMax[num_devices_];
dim3 cuDimProblem[num_devices_], cuDimBlock[num_devices_];


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

        err = cuDeviceGet(&cuDevices[i+1], i);
        checkErrDrv(err, "cuDeviceGet()");
        err = cuDeviceGetName(name, 100, cuDevices[i+1]);
        checkErrDrv(err, "cuDeviceGetName()");
        err = cuDeviceComputeCapability(&major, &minor, cuDevices[i+1]);
        checkErrDrv(err, "cuDeviceComputeCapability()");

        if (i==0) std::cerr << "  [*] ";
        else std::cerr << "  [ ] ";
        std::cerr << "Name: " << name << " (" << i+1 << ")" << std::endl;
        std::cerr << "      Compute capability: " << major << "." << minor << std::endl;

        // create context
        err = cuCtxCreate(&cuContexts[i+1], 0, cuDevices[i+1]);
        checkErrDrv(err, "cuCtxCreate()");
    }

    // initialize cuArgs
    for (size_t i=0; i<num_devices_; ++i) {
        cuArgs[i] = (void **)malloc(sizeof(void *));
        cuArgIdx[i] = 0;
        cuArgIdxMax[i] = 1;
    }
}


// load ptx assembly, create a module and kernel
void create_module_kernel(size_t dev, const char *ptx, const char *kernel_name) {
    CUresult err = CUDA_SUCCESS;
    bool print_progress = true;
    CUjit_target_enum target_cc = CU_TARGET_COMPUTE_20;

    const int errorLogSize = 10240;
    char errorLogBuffer[errorLogSize] = {0};

    int num_options = 2;
    CUjit_option options[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_TARGET };
    void *optionValues[] = { (void *)errorLogBuffer, (void *)errorLogSize, (void *)target_cc };

    // load ptx source
    if (print_progress) std::cerr << "Compiling(" << dev << ") '" << kernel_name << "' .";
    err = cuModuleLoadDataEx(&cuModules[dev], ptx, num_options, options, optionValues);
    if (err != CUDA_SUCCESS) {
        std::cerr << "Error log: " << errorLogBuffer << std::endl;
    }
    checkErrDrv(err, "cuModuleLoadDataEx()");

    // get function entry point
    if (print_progress) std::cerr << ".";
    err = cuModuleGetFunction(&cuFunctions[dev], cuModules[dev], kernel_name);
    checkErrDrv(err, "cuModuleGetFunction()");
    if (print_progress) std::cerr << ". done" << std::endl;
}


// load ll intermediate and compile to ptx
void load_kernel(size_t dev, const char *file_name, const char *kernel_name) {
    cuCtxPushCurrent(cuContexts[dev]);
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
    create_module_kernel(dev, PTX, kernel_name);
    cuCtxPopCurrent(NULL);
}


void get_tex_ref(size_t dev, const char *name) {
    CUresult err = CUDA_SUCCESS;

    err = cuModuleGetTexRef(&cuTextures[dev], cuModules[dev], name);
    checkErrDrv(err, "cuModuleGetTexRef()");
}

void bind_tex(size_t dev, mem_id mem, CUarray_format format) {
    void *host = mem_manager.get_host_mem(dev, mem);
    CUdeviceptr dev_mem = mem_manager.get_dev_mem(dev, mem);
    mem_ info = host_mems_[host];
    checkErrDrv(cuTexRefSetFormat(cuTextures[dev], format, 1), "cuTexRefSetFormat()");
    checkErrDrv(cuTexRefSetFlags(cuTextures[dev], CU_TRSF_READ_AS_INTEGER), "cuTexRefSetFlags()");
    checkErrDrv(cuTexRefSetAddress(0, cuTextures[dev], dev_mem, info.elem * info.width * info.height), "cuTexRefSetAddress()");
}


mem_id malloc_memory_size(size_t dev, void *host, size_t ox, size_t oy, size_t oz, size_t sx, size_t sy, size_t sz) {
    cuCtxPushCurrent(cuContexts[dev]);
    CUresult err = CUDA_SUCCESS;
    mem_id mem = mem_manager.get_id(dev, host);
    mem_ info = host_mems_[host];

    if (!mem) {
        CUdeviceptr dev_mem;
        err = cuMemAlloc(&dev_mem, info.elem * sx * sy);
        checkErrDrv(err, "cuMemAlloc()");
        mem = mem_manager.map_memory(dev, host, dev_mem, Global, ox, oy, oz, sx, sy, sz);
        std::cerr << " * malloc memory(" << dev << "): " << dev_mem << " (" << mem << ") <-> host: " << host << std::endl;
    } else {
        std::cerr << " * malloc memory(" << dev << "): returning old copy " << mem << " for " << host << std::endl;
    }

    cuCtxPopCurrent(NULL);
    return mem;
}
mem_id malloc_memory(size_t dev, void *host) {
    mem_ info = host_mems_[host];
    return malloc_memory_size(dev, host, 0, 0, 0, info.width, info.height, 0);
}


void free_memory(size_t dev, mem_id mem) {
    cuCtxPushCurrent(cuContexts[dev]);
    CUresult err = CUDA_SUCCESS;

    std::cerr << " * free memory(" << dev << "): " << mem << std::endl;

    CUdeviceptr dev_mem = mem_manager.get_dev_mem(dev, mem);
    err = cuMemFree(dev_mem);
    checkErrDrv(err, "cuMemFree()");
    mem_manager.remove(dev, mem);
    cuCtxPopCurrent(NULL);
}


void write_memory_size(size_t dev, mem_id mem, void *host, size_t ox, size_t oy, size_t oz, size_t sx, size_t sy, size_t sz) {
    cuCtxPushCurrent(cuContexts[dev]);
    CUresult err = CUDA_SUCCESS;
    mem_ info = host_mems_[host];
    CUdeviceptr dev_mem = mem_manager.get_dev_mem(dev, mem);

    std::cerr << " * write memory(" << dev << "):  " << mem << " <- " << host << " (" << ox << "," << oy << "," << oz << ")x(" << sx << "," << sy << "," << sz << ")" << std::endl;

    err = cuMemcpyHtoD(dev_mem, (char*)host + info.elem * (oy*info.width + ox), info.elem * sx * sy);
    checkErrDrv(err, "cuMemcpyHtoD()");
    cuCtxPopCurrent(NULL);
}
void write_memory(size_t dev, mem_id mem, void *host) {
    mem_ info = host_mems_[host];
    return write_memory_size(dev, mem, host, 0, 0, 0, info.width, info.height, 0);
}


void read_memory_size(size_t dev, mem_id mem, void *host, size_t ox, size_t oy, size_t oz, size_t sx, size_t sy, size_t sz) {
    cuCtxPushCurrent(cuContexts[dev]);
    CUresult err = CUDA_SUCCESS;
    mem_ info = host_mems_[host];
    CUdeviceptr dev_mem = mem_manager.get_dev_mem(dev, mem);

    std::cerr << " * read memory(" << dev << "):   " << mem << " -> " << host << " (" << ox << "," << oy << "," << oz << ")x(" << sx << "," << sy << "," << sz << ")" << std::endl;

    err = cuMemcpyDtoH((char*)host + info.elem * (oy*info.width + ox), dev_mem, info.elem * sx * sy);
    checkErrDrv(err, "cuMemcpyDtoH()");
    cuCtxPopCurrent(NULL);
}
void read_memory(size_t dev, mem_id mem, void *host) {
    mem_ info = host_mems_[host];
    return read_memory_size(dev, mem, host, 0, 0, 0, info.width, info.height, 0);
}


void synchronize(size_t dev) {
    cuCtxPushCurrent(cuContexts[dev]);
    CUresult err = CUDA_SUCCESS;

    err = cuCtxSynchronize();
    checkErrDrv(err, "cuCtxSynchronize()");
    cuCtxPopCurrent(NULL);
}


void set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) {
    cuDimProblem[dev].x = size_x;
    cuDimProblem[dev].y = size_y;
    cuDimProblem[dev].z = size_z;
}


void set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) {
    cuDimBlock[dev].x = size_x;
    cuDimBlock[dev].y = size_y;
    cuDimBlock[dev].z = size_z;
}


void set_kernel_arg(size_t dev, void *param) {
    cuArgIdx[dev]++;
    if (cuArgIdx[dev] > cuArgIdxMax[dev]) {
        cuArgs[dev] = (void **)realloc(cuArgs[dev], sizeof(void *)*cuArgIdx[dev]);
        cuArgIdxMax[dev] = cuArgIdx[dev];
    }
    cuArgs[dev][cuArgIdx[dev]-1] = (void *)malloc(sizeof(void *));
    cuArgs[dev][cuArgIdx[dev]-1] = param;
}


void set_kernel_arg_map(size_t dev, mem_id mem) {
    CUdeviceptr *dev_mem = &mem_manager.get_dev_mem(dev, mem);
    std::cerr << " * set arg mapped(" << dev << "): " << mem << std::endl;
    set_kernel_arg(dev, dev_mem);
}


void launch_kernel(size_t dev, const char *kernel_name) {
    cuCtxPushCurrent(cuContexts[dev]);
    CUresult err = CUDA_SUCCESS;
    CUevent start, end;
    unsigned int event_flags = CU_EVENT_DEFAULT;
    float time;
    std::string error_string = "cuLaunchKernel(";
    error_string += kernel_name;
    error_string += ")";

    // compute launch configuration
    dim3 grid;
    grid.x = cuDimProblem[dev].x / cuDimBlock[dev].x;
    grid.y = cuDimProblem[dev].y / cuDimBlock[dev].y;
    grid.z = cuDimProblem[dev].z / cuDimBlock[dev].z;

    cuEventCreate(&start, event_flags);
    cuEventCreate(&end, event_flags);

    // launch the kernel
    cuEventRecord(start, 0);
    err = cuLaunchKernel(cuFunctions[dev], grid.x, grid.y, grid.z, cuDimBlock[dev].x, cuDimBlock[dev].y, cuDimBlock[dev].z, 0, NULL, cuArgs[dev], NULL);
    checkErrDrv(err, error_string.c_str());
    err = cuCtxSynchronize();
    checkErrDrv(err, error_string.c_str());

    cuEventRecord(end, 0);
    cuEventSynchronize(end);
    cuEventElapsedTime(&time, start, end);

    std::cerr << "Kernel timing on device " << dev
              << " for '" << kernel_name << "' ("
              << cuDimProblem[dev].x*cuDimProblem[dev].y << ": "
              << cuDimProblem[dev].x << "x" << cuDimProblem[dev].y << ", "
              << cuDimBlock[dev].x*cuDimBlock[dev].y << ": "
              << cuDimBlock[dev].x << "x" << cuDimBlock[dev].y << "): "
              << time << "(ms)" << std::endl;

    cuEventDestroy(start);
    cuEventDestroy(end);

    // reset argument index
    cuArgIdx[dev] = 0;
    cuCtxPopCurrent(NULL);
}


// NVVM wrappers
mem_id nvvm_malloc_memory(size_t dev, void *host) { return malloc_memory(dev, host); }
void nvvm_free_memory(size_t dev, mem_id mem) { free_memory(dev, mem); }

void nvvm_write_memory(size_t dev, mem_id mem, void *host) { write_memory(dev, mem, host); }
void nvvm_read_memory(size_t dev, mem_id mem, void *host) { read_memory(dev, mem, host); }

void nvvm_load_kernel(size_t dev, const char *file_name, const char *kernel_name) { load_kernel(dev, file_name, kernel_name); }

void nvvm_set_kernel_arg(size_t dev, void *param) { set_kernel_arg(dev, param); }
void nvvm_set_kernel_arg_map(size_t dev, mem_id mem) { set_kernel_arg_map(dev, mem); }
void nvvm_set_kernel_arg_tex(size_t dev, mem_id mem, char *name, CUarray_format format) {
    std::cerr << "set arg tex: " << mem << std::endl;
    get_tex_ref(dev, name);
    bind_tex(dev, mem, format);
}
void nvvm_set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) { set_problem_size(dev, size_x, size_y, size_z); }
void nvvm_set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) { set_config_size(dev, size_x, size_y, size_z); }

void nvvm_launch_kernel(size_t dev, const char *kernel_name) { launch_kernel(dev, kernel_name); }
void nvvm_synchronize(size_t dev) { synchronize(dev); }

// helper functions
void *array(size_t elem_size, size_t width, size_t height) {
    void *mem;
    posix_memalign(&mem, 16, elem_size * width * height);
    std::cerr << " * array() -> " << mem << std::endl;
    host_mems_[mem] = {elem_size, width, height};
    return mem;
}
void free_array(void *host) {
    // TODO: free associated device memory

    // free host memory
    free(host);
}
mem_id map_memory(size_t dev, size_t type_, void *from, int ox, int oy, int oz, int sx, int sy, int sz) {
    mem_type type = (mem_type)type_;
    mem_ info = host_mems_[from];

    assert(oz==0 && sz==0 && "3D memory not yet supported");

    mem_id mem = mem_manager.get_id(dev, from);
    if (mem) {
        std::cerr << " * map memory(" << dev << "):    returning old copy " << mem << " for " << from << std::endl;
        return mem;
    }

    if (type==Global || type==Texture) {
        assert(sx==info.width && "currently only the y-dimension can be split");

        if (sy==info.height) {
            // mapping the whole memory
            mem = malloc_memory(dev, from);
            write_memory(dev, mem, from);
            std::cerr << " * map memory(" << dev << "):    " << from << " -> " << mem << std::endl;
        } else {
            // mapping and slicing of a region
            assert(sy < info.height && "slice larger then original memory");
            mem = malloc_memory_size(dev, from, ox, oy, oz, sx, sy, sz);
            mem_manager.write_memory(dev, mem, from);
            std::cerr << " * map memory(" << dev << "):    " << from << " (" << ox << "," << oy << "," << oz <<")x(" << sx << "," << sy << "," << sz << ") -> " << mem << std::endl;
        }
    } else {
        std::cerr << "unsupported memory: " << type << std::endl;
        exit(EXIT_FAILURE);
    }

    return mem;
}
void unmap_memory(size_t dev, size_t type_, mem_id mem) {
    mem_manager.read_memory(dev, mem);
    std::cerr << " * unmap memory(" << dev << "):  " << mem << std::endl;
    // TODO: mark device memory as unmapped
}
float random_val(int max) {
    return ((float)random() / RAND_MAX) * max;
}

int main(int argc, char *argv[]) {
    init_cuda();

    return main_impala();
}

