#include "cu_runtime.h"

#if CUDA_VERSION < 6000
    #error "CUDA 6.0 or higher required!"
#endif

#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

#ifndef LIBDEVICE_DIR
#define LIBDEVICE_DIR ""
#endif
#ifndef KERNEL_DIR
#define KERNEL_DIR ""
#endif

#define BENCH
#ifdef BENCH
float total_timing = 0.0f;
#endif

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

// runtime forward declarations
CUdeviceptr malloc_memory(size_t dev, void *host, size_t size);
void read_memory(size_t dev, CUdeviceptr mem, void *host, size_t size);
void write_memory(size_t dev, CUdeviceptr mem, void *host, size_t size);
void free_memory(size_t dev, mem_id mem);

void load_kernel(size_t dev, const char *file_name, const char *kernel_name, bool is_nvvm);

void get_tex_ref(size_t dev, const char *name);
void bind_tex(size_t dev, mem_id mem, CUarray_format format);

void set_kernel_arg(size_t dev, void *param);
void set_kernel_arg_map(size_t dev, mem_id mem);
void set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);
void set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);

void launch_kernel(size_t dev, const char *kernel_name);
void synchronize(size_t dev);


// global variables ...
enum mem_type {
    Generic     = 0,
    Global      = 1,
    Texture     = 2,
    Shared      = 3,
    Constant    = 4
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
        std::unordered_map<size_t, size_t> ummap;

    public:
        Memory() : count_(42) {}

    void associate_device(size_t host_dev, size_t assoc_dev) {
        ummap[assoc_dev] = host_dev;
    }
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
        return map_memory(dev, from, to, type, 0, 0, 0, info.width, info.height, 1);
    }

    void *get_host_mem(size_t dev, mem_id id) { return mmap[dev][id].cpu; }
    CUdeviceptr &get_dev_mem(size_t dev, mem_id id) { return mmap[dev][id].gpu; }

    void remove(size_t dev, mem_id id) {
        void *mem = idtomem[dev][id];
        memtoid[dev].erase(mem);
        idtomem[dev].erase(id);
    }


    mem_id malloc(size_t dev, void *host, size_t ox, size_t oy, size_t oz, size_t sx, size_t sy, size_t sz) {
        mem_id id = get_id(dev, host);

        if (id) {
            std::cerr << " * malloc memory(" << dev << "): returning old copy " << id << " for " << host << std::endl;
            return id;
        }

        id = get_id(ummap[dev], host);
        if (id) {
            std::cerr << " * malloc memory(" << dev << "): returning old copy " << id << " from associated device " << ummap[dev] << " for " << host << std::endl;
            id = map_memory(dev, host, get_dev_mem(ummap[dev], id), Global,
                            ox, oy, oz, sx, sy, sz);
        } else {
            mem_ info = host_mems_[host];
            void *host_ptr = (char*)host + (ox + oy*info.width)*info.elem;
            CUdeviceptr mem = malloc_memory(dev, host_ptr, info.elem*sx*sy);
            id = map_memory(dev, host, mem, Global, ox, oy, oz, sx, sy, sz);
            std::cerr << " * malloc memory(" << dev << "): " << mem << " (" << id << ") <-> host: " << host << std::endl;
        }
        return id;
    }
    mem_id malloc(size_t dev, void *host) {
        mem_ info = host_mems_[host];
        return malloc(dev, host, 0, 0, 0, info.width, info.height, 1);
    }

    void read(size_t dev, mem_id id) {
        assert(mmap[dev][id].cpu && "invalid host memory");
        mem_ info = host_mems_[mmap[dev][id].cpu];
        void *host = mmap[dev][id].cpu;
        std::cerr << " * read memory(" << dev << "):   " << id << " -> " << host
                  << " (" << mmap[dev][id].ox << "," << mmap[dev][id].oy << "," << mmap[dev][id].oz << ")x("
                  << mmap[dev][id].sx << "," << mmap[dev][id].sy << "," << mmap[dev][id].sz << ")" << std::endl;
        void *host_ptr = (char*)host + (mmap[dev][id].ox + mmap[dev][id].oy*info.width)*info.elem;
        read_memory(dev, mmap[dev][id].gpu, host_ptr, info.elem * mmap[dev][id].sx * mmap[dev][id].sy);
    }
    void write(size_t dev, mem_id id, void *host) {
        mem_ info = host_mems_[host];
        assert(host==mmap[dev][id].cpu && "invalid host memory");
        std::cerr << " * write memory(" << dev << "):  " << id << " <- " << host
                  << " (" << mmap[dev][id].ox << "," << mmap[dev][id].oy << "," << mmap[dev][id].oz << ")x("
                  << mmap[dev][id].sx << "," << mmap[dev][id].sy << "," << mmap[dev][id].sz << ")" << std::endl;
        void *host_ptr = (char*)host + (mmap[dev][id].ox + mmap[dev][id].oy*info.width)*info.elem;
        write_memory(dev, mmap[dev][id].gpu, host_ptr, info.elem * mmap[dev][id].sx * mmap[dev][id].sy);
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

long global_time = 0;

void getMicroTime() {
    struct timespec now;
    #ifdef __APPLE__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    now.tv_sec = mts.tv_sec;
    now.tv_nsec = mts.tv_nsec;
    #else
    clock_gettime(CLOCK_MONOTONIC, &now);
    #endif

    if (global_time==0) {
        global_time = now.tv_sec*1000000LL + now.tv_nsec / 1000LL;
    } else {
        global_time = (now.tv_sec*1000000LL + now.tv_nsec / 1000LL) - global_time;
        std::cerr << "   timing: " << global_time * 1.0e-3f << "(ms)" << std::endl;
        global_time = 0;
    }
}

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
        err = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevices[i+1]);
        checkErrDrv(err, "cuDeviceGetAttribute()");
        err = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevices[i+1]);
        checkErrDrv(err, "cuDeviceGetAttribute()");

        if (i==0) std::cerr << "  [*] ";
        else std::cerr << "  [ ] ";
        std::cerr << "Name: " << name << " (" << i+1 << ")" << std::endl;
        std::cerr << "      Compute capability: " << major << "." << minor << std::endl;

        // create context
        err = cuCtxCreate(&cuContexts[i+1], 0, cuDevices[i+1]);
        checkErrDrv(err, "cuCtxCreate()");

        // TODO: check for unified memory support and add missing associations
        mem_manager.associate_device(i, i);
    }

    // initialize cuArgs
    for (size_t i=0; i<num_devices_; ++i) {
        cuArgs[i] = (void **)malloc(sizeof(void *));
        cuArgIdx[i] = 0;
        cuArgIdxMax[i] = 1;
    }
}


// load ptx assembly, create a module and kernel
void create_module_kernel(size_t dev, const char *ptx, const char *kernel_name, CUjit_target target_cc) {
    CUresult err = CUDA_SUCCESS;
    bool print_progress = true;

    const int errorLogSize = 10240;
    char errorLogBuffer[errorLogSize] = {0};

    int num_options = 3;
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


// computes occupancy for kernel function
void print_kernel_occupancy(size_t dev, const char *kernel_name) {
    #if CUDA_VERSION >= 6050
    CUresult err = CUDA_SUCCESS;
    int warp_size;
    int block_size = cuDimBlock[dev].x*cuDimBlock[dev].y;
    size_t dynamic_smem_bytes = 0;
    int block_size_limit = 0;

    err = cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevices[dev]);
    checkErrDrv(err, "cuDeviceGetAttribute()");

    int active_blocks;
    int min_grid_size, opt_block_size;
    err = cuOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, cuFunctions[dev], block_size, dynamic_smem_bytes);
    checkErrDrv(err, "cuOccupancyMaxActiveBlocksPerMultiprocessor()");
    err = cuOccupancyMaxPotentialBlockSize(&min_grid_size, &opt_block_size, cuFunctions[dev], NULL, dynamic_smem_bytes, block_size_limit);
    checkErrDrv(err, "cuOccupancyMaxPotentialBlockSize()");

    // re-compute with optimal block size
    int max_blocks;
    err = cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, cuFunctions[dev], opt_block_size, dynamic_smem_bytes);
    checkErrDrv(err, "cuOccupancyMaxActiveBlocksPerMultiprocessor()");

    int max_warps = max_blocks * (opt_block_size/warp_size);
    int active_warps = active_blocks * (block_size/warp_size);
    float occupancy = (float)active_warps/(float)max_warps;
    std::cerr << "Occupancy for kernel '" << kernel_name << "' is "
              << std::fixed << std::setprecision(2) << occupancy << ": "
              << active_warps << " out of " << max_warps << " warps" << std::endl
              << "Optimal block size for max occupancy: " << opt_block_size << std::endl;
    #endif
}


// compile CUDA source file to PTX assembly using nvcc compiler
void cuda_to_ptx(const char *file_name, std::string target_cc) {
    char line[FILENAME_MAX];
    FILE *fpipe;

    std::string command = "nvcc -ptx -arch=compute_" + target_cc + " ";
    command += std::string(file_name) + " -o ";
    command += std::string(file_name) + ".ptx 2>&1";

    if (!(fpipe = (FILE *)popen(command.c_str(), "r"))) {
        perror("Problems with pipe");
        exit(EXIT_FAILURE);
    }

    while (fgets(line, sizeof(char) * FILENAME_MAX, fpipe)) {
        std::cerr << line;
    }
    pclose(fpipe);
}


// load ll intermediate and compile to ptx
void load_kernel(size_t dev, const char *file_name, const char *kernel_name, bool is_nvvm) {
    cuCtxPushCurrent(cuContexts[dev]);
    nvvmProgram program;
    std::string srcString;
    char *PTX = NULL;

    int major, minor;
    CUresult err = CUDA_SUCCESS;
    err = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevices[dev]);
    checkErrDrv(err, "cuDeviceGetAttribute()");
    err = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevices[dev]);
    checkErrDrv(err, "cuDeviceGetAttribute()");
    CUjit_target target_cc = (CUjit_target)(major*10 + minor);

    if (is_nvvm) {
        nvvmResult err;
        // select libdevice module according to documentation
        const char *libdevice_file_name;
        switch (target_cc) {
            default:
                assert(false && "unsupported compute capability");
            case CU_TARGET_COMPUTE_20:
            case CU_TARGET_COMPUTE_21:
            case CU_TARGET_COMPUTE_32:
                libdevice_file_name = "libdevice.compute_20.10.bc"; break;
            case CU_TARGET_COMPUTE_30:
                libdevice_file_name = "libdevice.compute_30.10.bc"; break;
            case CU_TARGET_COMPUTE_35:
                libdevice_file_name = "libdevice.compute_35.10.bc"; break;
        }
        std::ifstream libdeviceFile(std::string(LIBDEVICE_DIR)+libdevice_file_name);
        if (!libdeviceFile.is_open()) {
            std::cerr << "ERROR: Can't open libdevice source file '" << libdevice_file_name << "'!" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string libdeviceString = std::string(std::istreambuf_iterator<char>(libdeviceFile),
                (std::istreambuf_iterator<char>()));

        std::ifstream srcFile(file_name);
        if (!srcFile.is_open()) {
            std::cerr << "ERROR: Can't open LL source file '" << file_name << "'!" << std::endl;
            exit(EXIT_FAILURE);
        }

        srcString = std::string(std::istreambuf_iterator<char>(srcFile),
                (std::istreambuf_iterator<char>()));

        err = nvvmCreateProgram(&program);
        checkErrNvvm(err, "nvvmCreateProgram()");

        err = nvvmAddModuleToProgram(program, libdeviceString.c_str(), libdeviceString.length(), libdevice_file_name);
        checkErrNvvm(err, "nvvmAddModuleToProgram()");

        err = nvvmAddModuleToProgram(program, srcString.c_str(), srcString.length(), file_name);
        checkErrNvvm(err, "nvvmAddModuleToProgram()");

        std::string compute_arch("-arch=compute_" + std::to_string(target_cc));
        int num_options = 2;
        const char *options[3];
        options[0] = compute_arch.c_str();
        options[1] = "-ftz=1";
        options[2] = "-g";

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

        size_t PTXSize;
        err = nvvmGetCompiledResultSize(program, &PTXSize);
        checkErrNvvm(err, "nvvmGetCompiledResultSize()");

        PTX = (char*)malloc(PTXSize);
        err = nvvmGetCompiledResult(program, PTX);
        if (err != NVVM_SUCCESS) free(PTX);
        checkErrNvvm(err, "nvvmGetCompiledResult()");

        err = nvvmDestroyProgram(&program);
        if (err != NVVM_SUCCESS) free(PTX);
        checkErrNvvm(err, "nvvmDestroyProgram()");
    } else {
        cuda_to_ptx(file_name, std::to_string(target_cc));
        std::string ptx_filename = file_name;
        ptx_filename += ".ptx";

        std::ifstream srcFile(ptx_filename.c_str());
        if (!srcFile.is_open()) {
            std::cerr << "ERROR: Can't open PTX source file '" << ptx_filename << "'!" << std::endl;
            exit(EXIT_FAILURE);
        }

        srcString = std::string(std::istreambuf_iterator<char>(srcFile),
                (std::istreambuf_iterator<char>()));
        PTX = (char *)srcString.c_str();
    }

    // compile ptx
    create_module_kernel(dev, PTX, kernel_name, target_cc);

    if (is_nvvm) free(PTX);
    cuCtxPopCurrent(NULL);
}


void get_tex_ref(size_t dev, const char *name) {
    CUresult err = CUDA_SUCCESS;

    err = cuModuleGetTexRef(&cuTextures[dev], cuModules[dev], name);
    checkErrDrv(err, "cuModuleGetTexRef()");
}

void bind_tex(size_t dev, mem_id mem, CUarray_format format) {
    void *host = mem_manager.get_host_mem(dev, mem);
    CUdeviceptr &dev_mem = mem_manager.get_dev_mem(dev, mem);
    mem_ info = host_mems_[host];
    checkErrDrv(cuTexRefSetFormat(cuTextures[dev], format, 1), "cuTexRefSetFormat()");
    checkErrDrv(cuTexRefSetFlags(cuTextures[dev], CU_TRSF_READ_AS_INTEGER), "cuTexRefSetFlags()");
    checkErrDrv(cuTexRefSetAddress(0, cuTextures[dev], dev_mem, info.elem * info.width * info.height), "cuTexRefSetAddress()");
}


CUdeviceptr malloc_memory(size_t dev, void *host, size_t size) {
    cuCtxPushCurrent(cuContexts[dev]);
    CUdeviceptr mem;
    CUresult err = CUDA_SUCCESS;

    // TODO: unified memory support using cuMemAllocManaged();
    err = cuMemAlloc(&mem, size);
    checkErrDrv(err, "cuMemAlloc()");

    cuCtxPopCurrent(NULL);
    return mem;
}


void free_memory(size_t dev, mem_id mem) {
    cuCtxPushCurrent(cuContexts[dev]);
    CUresult err = CUDA_SUCCESS;

    std::cerr << " * free memory(" << dev << "):   " << mem << std::endl;

    CUdeviceptr &dev_mem = mem_manager.get_dev_mem(dev, mem);
    err = cuMemFree(dev_mem);
    checkErrDrv(err, "cuMemFree()");
    mem_manager.remove(dev, mem);
    cuCtxPopCurrent(NULL);
}


void write_memory(size_t dev, CUdeviceptr mem, void *host, size_t size) {
    cuCtxPushCurrent(cuContexts[dev]);
    CUresult err = CUDA_SUCCESS;

    err = cuMemcpyHtoD(mem, host, size);
    checkErrDrv(err, "cuMemcpyHtoD()");
    cuCtxPopCurrent(NULL);
}


void read_memory(size_t dev, CUdeviceptr mem, void *host, size_t size) {
    cuCtxPushCurrent(cuContexts[dev]);
    CUresult err = CUDA_SUCCESS;

    err = cuMemcpyDtoH(host, mem, size);
    checkErrDrv(err, "cuMemcpyDtoH()");
    cuCtxPopCurrent(NULL);
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
    CUdeviceptr &dev_mem = mem_manager.get_dev_mem(dev, mem);
    std::cerr << " * set arg mapped(" << dev << "): " << mem << std::endl;
    set_kernel_arg(dev, &dev_mem);
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
    #ifdef BENCH
    std::vector<float> timings;
    for (size_t iter=0; iter<7; ++iter) {
    #endif
    cuEventRecord(start, 0);
    err = cuLaunchKernel(cuFunctions[dev], grid.x, grid.y, grid.z, cuDimBlock[dev].x, cuDimBlock[dev].y, cuDimBlock[dev].z, 0, NULL, cuArgs[dev], NULL);
    checkErrDrv(err, error_string.c_str());
    err = cuCtxSynchronize();
    checkErrDrv(err, error_string.c_str());

    cuEventRecord(end, 0);
    cuEventSynchronize(end);
    cuEventElapsedTime(&time, start, end);
    timings.emplace_back(time);
    #ifdef BENCH
    }
    std::sort(timings.begin(), timings.end());
    total_timing += timings[timings.size()/2];
    #endif

    std::cerr << "Kernel timing on device " << dev
              << " for '" << kernel_name << "' ("
              << cuDimProblem[dev].x*cuDimProblem[dev].y << ": "
              << cuDimProblem[dev].x << "x" << cuDimProblem[dev].y << ", "
              << cuDimBlock[dev].x*cuDimBlock[dev].y << ": "
              << cuDimBlock[dev].x << "x" << cuDimBlock[dev].y << "): "
              #ifdef BENCH
              << "median of " << timings.size() << " runs: "
              << timings[timings.size()/2]
              #else
              << time
              #endif
              << "(ms)" << std::endl;
    print_kernel_occupancy(dev, kernel_name);

    cuEventDestroy(start);
    cuEventDestroy(end);

    // reset argument index
    cuArgIdx[dev] = 0;
    cuCtxPopCurrent(NULL);
}


// NVVM wrappers
mem_id nvvm_malloc_memory(size_t dev, void *host) { return mem_manager.malloc(dev, host); }
void nvvm_free_memory(size_t dev, mem_id mem) { free_memory(dev, mem); }

void nvvm_write_memory(size_t dev, mem_id mem, void *host) { mem_manager.write(dev, mem, host); }
void nvvm_read_memory(size_t dev, mem_id mem, void *host) { mem_manager.read(dev, mem); }

void nvvm_load_nvvm_kernel(size_t dev, const char *file_name, const char *kernel_name) { load_kernel(dev, file_name, kernel_name, true); }
void nvvm_load_cuda_kernel(size_t dev, const char *file_name, const char *kernel_name) { load_kernel(dev, file_name, kernel_name, false); }

void nvvm_set_kernel_arg(size_t dev, void *param) { set_kernel_arg(dev, param); }
void nvvm_set_kernel_arg_map(size_t dev, mem_id mem) { set_kernel_arg_map(dev, mem); }
void nvvm_set_kernel_arg_tex(size_t dev, mem_id mem, const char *name, CUarray_format format) {
    std::cerr << "set arg tex: " << mem << std::endl;
    get_tex_ref(dev, name);
    bind_tex(dev, mem, format);
}
void nvvm_set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) { set_problem_size(dev, size_x, size_y, size_z); }
void nvvm_set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) { set_config_size(dev, size_x, size_y, size_z); }

void nvvm_launch_kernel(size_t dev, const char *kernel_name) { launch_kernel(dev, kernel_name); }
void nvvm_synchronize(size_t dev) { synchronize(dev); }

// helper functions
void thorin_init() { init_cuda(); }
void *thorin_malloc(size_t size) {
    void *mem;
    posix_memalign(&mem, 64, size);
    std::cerr << " * malloc() -> " << mem << std::endl;
    host_mems_[mem] = {1, size, 1};
    return mem;
}
void thorin_free(void *ptr) {
    // TODO: free associated device memory

    // free host memory
    free(ptr);
}
mem_id map_memory(size_t dev, size_t type_, void *from, int ox, int oy, int oz, int sx, int sy, int sz) {
    mem_type type = (mem_type)type_;
    mem_ info = host_mems_[from];

    assert(oz==0 && sz==1 && "3D memory not yet supported");

    mem_id mem = mem_manager.get_id(dev, from);
    if (mem) {
        std::cerr << " * map memory(" << dev << "):    returning old copy " << mem << " for " << from << std::endl;
        return mem;
    }

    if (type==Global || type==Texture) {
        assert(sx==info.width && "currently only the y-dimension can be split");

        if (sy==info.height) {
            // mapping the whole memory
            mem = mem_manager.malloc(dev, from);
            mem_manager.write(dev, mem, from);
            std::cerr << " * map memory(" << dev << "):    " << from << " -> " << mem << std::endl;
        } else {
            // mapping and slicing of a region
            assert(sy < info.height && "slice larger then original memory");
            mem = mem_manager.malloc(dev, from, ox, oy, oz, sx, sy, sz);
            mem_manager.write(dev, mem, from);
            std::cerr << " * map memory(" << dev << "):    " << from << " (" << ox << "," << oy << "," << oz <<")x(" << sx << "," << sy << "," << sz << ") -> " << mem << std::endl;
        }
    } else {
        std::cerr << "unsupported memory: " << type << std::endl;
        exit(EXIT_FAILURE);
    }

    return mem;
}
void unmap_memory(size_t dev, size_t type_, mem_id mem) {
    mem_manager.read(dev, mem);
    std::cerr << " * unmap memory(" << dev << "):  " << mem << std::endl;
    // TODO: mark device memory as unmapped
}
float random_val(int max) {
    return ((float)random() / RAND_MAX) * max;
}

#ifdef PROVIDE_MAIN
int main(int argc, char *argv[]) {
    init_cuda();

    int ret = main_impala();
    #ifdef BENCH
    std::cerr << "total timing: " << total_timing << " (ms)" << std::endl;
    #endif
    return ret;
}
#endif

