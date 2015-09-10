#include "cu_runtime.h"

#include "thorin_runtime.h"
#include "thorin_utils.h"

#if CUDA_VERSION < 6050
    #error "CUDA 6.5 or higher required!"
#endif

#if CUDA_VERSION >= 7000
#include <nvrtc.h>
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef LIBDEVICE_DIR
#define LIBDEVICE_DIR ""
#endif
#ifndef KERNEL_DIR
#define KERNEL_DIR ""
#endif

template <typename T>
void runtime_log(T t) {
    #ifndef NDEBUG
    std::clog << t;
    #endif
}

template <typename T, typename... Args>
void runtime_log(T t, Args... args) {
    #ifndef NDEBUG
    std::clog << t;
    runtime_log(args...);
    #endif
}

// define dim3
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};

typedef struct dim3 dim3;
// define dim3

// runtime forward declarations
CUdeviceptr malloc_memory(uint32_t dev, void* host, uint32_t size);
void read_memory(uint32_t dev, CUdeviceptr mem, void* host, uint32_t size);
void write_memory(uint32_t dev, CUdeviceptr mem, void* host, uint32_t size);
void free_memory(uint32_t dev, CUdeviceptr mem);

void load_kernel(uint32_t dev, std::string file_name, std::string kernel_name, bool is_nvvm);
void unload_module(uint32_t dev, CUmodule module);

void get_tex_ref(uint32_t dev, std::string name);
void bind_tex(uint32_t dev, mem_id mem, CUarray_format format);

void set_kernel_arg(uint32_t dev, void* param);
void set_kernel_arg_map(uint32_t dev, mem_id mem);
void set_kernel_arg_const(uint32_t dev, void* param, std::string name, uint32_t size);
void set_problem_size(uint32_t dev, uint32_t size_x, uint32_t size_y, uint32_t size_z);
void set_config_size(uint32_t dev, uint32_t size_x, uint32_t size_y, uint32_t size_z);

void launch_kernel(uint32_t dev, std::string kernel_name);
void synchronize(uint32_t dev);


// global variables ...
std::vector<CUdevice> devices_;
std::vector<CUcontext> contexts_;
std::vector<CUmodule> modules_;
std::vector<CUfunction> functions_;
std::vector<std::unordered_map<std::string, CUmodule>> module_cache_;
std::unordered_map<CUmodule, std::unordered_map<std::string, CUfunction>> function_cache_;
std::vector<CUtexref> textures_;
void** cuArgs;
int cuArgIdx, cuArgIdxMax;
dim3 cuDimProblem, cuDimBlock;

enum mem_type {
    Generic     = 0,
    Global      = 1,
    Texture     = 2,
    Shared      = 3,
    Constant    = 4
};


class Memory {
    private:
        uint32_t count_;
        typedef struct mapping_ {
            void* cpu;
            CUdeviceptr gpu;
            mem_type type;
            uint32_t offset, size;
            mem_id id;
        } mapping_;
        std::vector<std::unordered_map<mem_id, mapping_>> mmap;
        std::vector<std::unordered_map<mem_id, void*>> idtomem;
        std::vector<std::unordered_map<void*, mem_id>> memtoid;
        std::vector<std::unordered_map<mem_id, uint32_t>> mcount;
        std::unordered_map<uint32_t, uint32_t> ummap;
        std::unordered_map<void*, uint32_t> hostmem;

        mem_id new_id() { return count_++; }
        void insert(uint32_t dev, void* mem, mem_id id, mapping_ mem_map) {
            idtomem[dev][id] = mem;
            memtoid[dev][mem] = id;
            mmap[dev][id] = mem_map;
        }
        void remove(uint32_t dev, mem_id id) {
            void* mem = idtomem[dev][id];
            idtomem[dev].erase(id);
            memtoid[dev].erase(mem);
            mmap[dev].erase(id);
        }

    public:
        Memory() : count_(42) {}
        ~Memory() {
            uint32_t dev = 0;
            for (auto dmap : mmap) {
                for (auto it : dmap) free(dev, it.first);
                dev++;
            }
            // CUDA is shutting down when the destructor is called
            // unloading modules will fail
        }

    void reserve(uint32_t num) {
        mmap.resize(mmap.size() + num);
        idtomem.resize(idtomem.size() + num);
        memtoid.resize(memtoid.size() + num);
        mcount.resize(mcount.size() + num);
    }
    void associate_device(uint32_t host_dev, uint32_t assoc_dev) {
        ummap[assoc_dev] = host_dev;
    }
    mem_id get_id(uint32_t dev, void* mem) { return memtoid[dev][mem]; }
    mem_id map_memory(uint32_t dev, void* from, CUdeviceptr to, mem_type type, uint32_t offset, uint32_t size) {
        mem_id id = new_id();
        mapping_ mem_map = { from, to, type, offset, size, id };
        insert(dev, from, id, mem_map);
        return id;
    }
    mem_id map_memory(uint32_t dev, void* from, CUdeviceptr to, mem_type type) {
        assert(hostmem.count(from) && "memory not allocated by thorin");
        return map_memory(dev, from, to, type, 0, hostmem[from]);
    }

    void* get_host_mem(uint32_t dev, mem_id id) { return mmap[dev][id].cpu; }
    CUdeviceptr& get_dev_mem(uint32_t dev, mem_id id) { return mmap[dev][id].gpu; }


    void* malloc_host(uint32_t size) {
        void* mem = thorin_aligned_malloc(size, 64);
        runtime_log(" * malloc host(", size, ") -> ", mem, "\n");
        hostmem[mem] = size;
        return mem;
    }
    void free_host(void* ptr) {
        if (ptr==nullptr) return;
        // TODO: free associated device memory
        assert(hostmem.count(ptr) && "memory not allocated by thorin");
        runtime_log(" * free host(", ptr, ")\n");
        hostmem.erase(ptr);
        // free host memory
        thorin_aligned_free(ptr);
    }
    size_t host_mem_size(void* ptr) {
        assert(hostmem.count(ptr) && "memory not allocated by thorin");
        return hostmem[ptr];
    }

    mem_id malloc(uint32_t dev, void* host, uint32_t offset, uint32_t size) {
        assert(hostmem.count(host) && "memory not allocated by thorin");
        mem_id id = get_id(dev, host);

        if (id) {
            runtime_log(" * malloc memory(", dev, "): returning old copy ", id, " for ", host, "\n");
            mcount[dev][id]++;
            return id;
        }

        id = get_id(ummap[dev], host);
        if (id) {
            runtime_log(" * malloc memory(", dev, "): returning old copy ", id, " from associated device ", ummap[dev], " for ", host, "\n");
            id = map_memory(dev, host, get_dev_mem(ummap[dev], id), Global, offset, size);
        } else {
            void* host_ptr = (char*)host + offset;
            CUdeviceptr mem = malloc_memory(dev, host_ptr, size);
            id = map_memory(dev, host, mem, Global, offset, size);
            runtime_log(" * malloc memory(", dev, "): ", mem, " (", id, ") <-> host: ", host, "\n");
        }
        mcount[dev][id] = 1;
        return id;
    }
    mem_id malloc(uint32_t dev, void* host) {
        assert(hostmem.count(host) && "memory not allocated by thorin");
        return malloc(dev, host, 0, hostmem[host]);
    }

    void free(uint32_t dev, mem_id mem) {
        auto ref_count = --mcount[dev][mem];
        if (ref_count) {
            runtime_log(" * free memory(", dev, "):   ", mem, " update ref count to ", ref_count, "\n");
        } else {
            runtime_log(" * free memory(", dev, "):   ", mem, "\n");
            CUdeviceptr dev_mem = get_dev_mem(dev, mem);
            free_memory(dev, dev_mem);
            remove(dev, mem);
        }
    }

    void read(uint32_t dev, mem_id id) {
        assert(mmap[dev][id].cpu && "invalid host memory");
        void* host = mmap[dev][id].cpu;
        runtime_log(" * read memory(", dev, "):   ", id, " -> ", host,
                    " [", mmap[dev][id].offset, ":", mmap[dev][id].size, "]\n");
        void* host_ptr = (char*)host + mmap[dev][id].offset;
        read_memory(dev, mmap[dev][id].gpu, host_ptr, mmap[dev][id].size);
    }
    void write(uint32_t dev, mem_id id, void* host) {
        assert(host==mmap[dev][id].cpu && "invalid host memory");
        runtime_log(" * write memory(", dev, "):  ", id, " <- ", host,
                    " [", mmap[dev][id].offset, ":", mmap[dev][id].size, "]\n");
        void* host_ptr = (char*)host + mmap[dev][id].offset;
        write_memory(dev, mmap[dev][id].gpu, host_ptr, mmap[dev][id].size);
    }

    void munmap(mem_id id) {
        uint32_t dev = 0;
        for (auto dmap : mmap) {
            if (dmap.count(id)) {
                runtime_log(" * munmap memory(", dev, "): ", id, "\n");
                read(dev, id);
                free(dev, id);
                return;
            }
            dev++;
        }
        assert(false && "cannot find mapped device memory");
    }
};
Memory mem_manager;

#define check_dev(dev) __check_device(dev)
#define checkErrNvvm(err, name)  __checkNvvmErrors  (err, name, __FILE__, __LINE__)
#define checkErrNvrtc(err, name) __checkNvrtcErrors (err, name, __FILE__, __LINE__)
#define checkErrDrv(err, name)   __checkCudaErrors  (err, name, __FILE__, __LINE__)

std::string getCUDAErrorCodeStrDrv(CUresult errorCode) {
    const char* error_name;
    const char* error_string;
    cuGetErrorName(errorCode, &error_name);
    cuGetErrorString(errorCode, &error_string);
    return std::string(error_name) + ": " + std::string(error_string);
}

inline void __checkCudaErrors(CUresult err, std::string name, std::string file, const int line) {
    if (CUDA_SUCCESS != err) {
        std::cerr << "ERROR (Driver API): " << name << " (" << err << ")" << " [file " << file << ", line " << line << "]: " << std::endl;
        std::cerr << getCUDAErrorCodeStrDrv(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void __checkNvvmErrors(nvvmResult err, std::string name, std::string file, const int line) {
    if (NVVM_SUCCESS != err) {
        std::cerr << "ERROR (NVVM API): " << name << " (" << err << ")" << " [file " << file << ", line " << line << "]: " << std::endl;
        std::cerr << nvvmGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#if CUDA_VERSION >= 7000
inline void __checkNvrtcErrors(nvrtcResult err, std::string name, std::string file, const int line) {
    if (NVRTC_SUCCESS != err) {
        std::cerr << "ERROR (NVRTC API): " << name << " (" << err << ")" << " [file " << file << ", line " << line << "]: " << std::endl;
        std::cerr << nvrtcGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
#else
#define __checkNvrtcErrors(err, name, file, line)
#endif

inline void __check_device(uint32_t dev) {
    if (dev >= devices_.size()) {
        std::cerr << "ERROR: requested device #" << dev << ", but only " << devices_.size() << " CUDA devices [0.." << devices_.size()-1 << "] available!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// initialize CUDA device
void init_cuda() {
    int device_count = 0, driver_version = 0, nvvm_major = 0, nvvm_minor = 0;

    setenv("CUDA_CACHE_DISABLE", "1", 1);

    CUresult err = cuInit(0);
    checkErrDrv(err, "cuInit()");

    err = cuDeviceGetCount(&device_count);
    checkErrDrv(err, "cuDeviceGetCount()");
    assert(device_count && "no CUDA device found");

    err = cuDriverGetVersion(&driver_version);
    checkErrDrv(err, "cuDriverGetVersion()");

    nvvmResult errNvvm = nvvmVersion(&nvvm_major, &nvvm_minor);
    checkErrNvvm(errNvvm, "nvvmVersion()");

    runtime_log("CUDA Driver Version ", driver_version/1000, ".", (driver_version%100)/10, "\n");
    #if CUDA_VERSION >= 7000
    int nvrtc_major = 0, nvrtc_minor = 0;
    nvrtcResult errNvrtc = nvrtcVersion(&nvrtc_major, &nvrtc_minor);
    checkErrNvrtc(errNvrtc, "nvrtcVersion()");
    runtime_log("NVRTC Version ", nvrtc_major, ".", nvrtc_minor, "\n");
    #endif
    runtime_log("NVVM Version ", nvvm_major, ".", nvvm_minor, "\n");

    mem_manager.reserve(device_count);
    devices_.resize(device_count);
    contexts_.resize(device_count);
    modules_.resize(device_count);
    functions_.resize(device_count);
    module_cache_.resize(device_count);
    textures_.resize(device_count);

    for (int i=0; i<device_count; ++i) {
        int major, minor;
        char name[100];

        err = cuDeviceGet(&devices_[i], i);
        checkErrDrv(err, "cuDeviceGet()");
        err = cuDeviceGetName(name, 100, devices_[i]);
        checkErrDrv(err, "cuDeviceGetName()");
        err = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devices_[i]);
        checkErrDrv(err, "cuDeviceGetAttribute()");
        err = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devices_[i]);
        checkErrDrv(err, "cuDeviceGetAttribute()");

        runtime_log(" (", i, ")", " Name: ", name, "\n",
                    "     Compute capability: ", major, ".", minor, "\n");

        // create context
        err = cuCtxCreate(&contexts_[i], 0, devices_[i]);
        checkErrDrv(err, "cuCtxCreate()");

        // TODO: check for unified memory support and add missing associations
        mem_manager.associate_device(i, i);
    }

    // initialize cuArgs
    cuArgs = (void**)malloc(sizeof(void*));
    cuArgIdx = 0;
    cuArgIdxMax = 1;
}


// create module from ptx assembly
void create_module(uint32_t dev, const void* ptx, std::string file_name, CUjit_target target_cc) {
    CUresult err = CUDA_SUCCESS;
    const unsigned opt_level = 3;
    const int error_log_size = 10240;
    const int num_options = 4;
    char error_log_buffer[error_log_size] = { 0 };

    CUjit_option options[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_TARGET, CU_JIT_OPTIMIZATION_LEVEL };
    void* option_values[]  = { (void*)error_log_buffer, (void*)(size_t)error_log_size, (void*)target_cc, (void*)(size_t)opt_level };

    // load ptx source
    runtime_log("Compiling(", dev, ") '", file_name, "' .");
    err = cuModuleLoadDataEx(&modules_[dev], ptx, num_options, options, option_values);
    module_cache_[dev][file_name] = modules_[dev];

    if (err != CUDA_SUCCESS) {
        std::cerr << "Error log: " << error_log_buffer << std::endl;
    }
    checkErrDrv(err, "cuModuleLoadDataEx()");
    runtime_log(". done\n");
}


// get kernel from module
void create_kernel(uint32_t dev, std::string kernel_name) {
    // get function entry point
    CUresult err = cuModuleGetFunction(&functions_[dev], modules_[dev], kernel_name.c_str());
    checkErrDrv(err, "cuModuleGetFunction('" + kernel_name + "')");
    function_cache_[modules_[dev]][kernel_name] = functions_[dev];
}


// computes occupancy for kernel function
void print_kernel_occupancy(uint32_t dev, std::string kernel_name) {
    CUresult err = CUDA_SUCCESS;
    int warp_size;
    int block_size = cuDimBlock.x*cuDimBlock.y;
    size_t dynamic_smem_bytes = 0;
    int block_size_limit = 0;

    err = cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, devices_[dev]);
    checkErrDrv(err, "cuDeviceGetAttribute()");

    int active_blocks;
    int min_grid_size, opt_block_size;
    err = cuOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, functions_[dev], block_size, dynamic_smem_bytes);
    checkErrDrv(err, "cuOccupancyMaxActiveBlocksPerMultiprocessor()");
    err = cuOccupancyMaxPotentialBlockSize(&min_grid_size, &opt_block_size, functions_[dev], NULL, dynamic_smem_bytes, block_size_limit);
    checkErrDrv(err, "cuOccupancyMaxPotentialBlockSize()");

    // re-compute with optimal block size
    int max_blocks;
    err = cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, functions_[dev], opt_block_size, dynamic_smem_bytes);
    checkErrDrv(err, "cuOccupancyMaxActiveBlocksPerMultiprocessor()");

    block_size = ((block_size + warp_size - 1) / warp_size) * warp_size;
    int max_warps = max_blocks * (opt_block_size/warp_size);
    int active_warps = active_blocks * (block_size/warp_size);
    float occupancy = (float)active_warps/(float)max_warps;
    runtime_log("Occupancy for kernel '", kernel_name, "' is ", occupancy, ": ",
               active_warps, " out of ", max_warps, " warps\n",
               "Optimal block size for max occupancy: ", opt_block_size, "\n");
}

// load NVVM source and compile kernel
void compile_nvvm(uint32_t dev, std::string file_name, CUjit_target target_cc) {
    nvvmResult err;
    nvvmProgram program;

    // select libdevice module according to documentation
    std::string libdevice_file_name;
    switch (target_cc) {
        #if CUDA_VERSION == 6050
        case CU_TARGET_COMPUTE_37:
        #endif
        default:
            assert(false && "unsupported compute capability");
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
    if (!libdevice_file.is_open()) {
        std::cerr << "ERROR: Can't open libdevice source file '" << libdevice_file_name << "'!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string libdevice_string = std::string(std::istreambuf_iterator<char>(libdevice_file), (std::istreambuf_iterator<char>()));

    std::ifstream src_file(std::string(KERNEL_DIR) + file_name);
    if (!src_file.is_open()) {
        std::cerr << "ERROR: Can't open LL source file '" << KERNEL_DIR << file_name << "'!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string src_string = std::string(std::istreambuf_iterator<char>(src_file), (std::istreambuf_iterator<char>()));

    err = nvvmCreateProgram(&program);
    checkErrNvvm(err, "nvvmCreateProgram()");

    err = nvvmAddModuleToProgram(program, libdevice_string.c_str(), libdevice_string.length(), libdevice_file_name.c_str());
    checkErrNvvm(err, "nvvmAddModuleToProgram()");

    err = nvvmAddModuleToProgram(program, src_string.c_str(), src_string.length(), file_name.c_str());
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
        char* error_log = new char[log_size];
        nvvmGetProgramLog(program, error_log);
        std::cerr << "Error log: " << error_log << std::endl;
        delete[] error_log;
    }
    checkErrNvvm(err, "nvvmCompileProgram()");

    size_t ptx_size;
    err = nvvmGetCompiledResultSize(program, &ptx_size);
    checkErrNvvm(err, "nvvmGetCompiledResultSize()");

    char* ptx = new char[ptx_size];
    err = nvvmGetCompiledResult(program, ptx);
    checkErrNvvm(err, "nvvmGetCompiledResult()");

    err = nvvmDestroyProgram(&program);
    checkErrNvvm(err, "nvvmDestroyProgram()");

    // compile ptx
    create_module(dev, ptx, file_name, target_cc);
    delete[] ptx;
}

// load CUDA source and compile kernel
#if CUDA_VERSION >= 7000
void compile_cuda(uint32_t dev, std::string file_name, CUjit_target target_cc) {
    nvrtcResult err;
    nvrtcProgram program;

    std::ifstream src_file(std::string(KERNEL_DIR) + file_name);
    if (!src_file.is_open()) {
        std::cerr << "ERROR: Can't open CU source file '" << KERNEL_DIR << file_name << "'!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string src_string = std::string(std::istreambuf_iterator<char>(src_file), (std::istreambuf_iterator<char>()));

    err = nvrtcCreateProgram(&program, src_string.c_str(), file_name.c_str(), 0, NULL, NULL);
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
        char* error_log = new char[log_size];
        nvrtcGetProgramLog(program, error_log);
        std::cerr << "Error log: " << error_log << std::endl;
        delete[] error_log;
    }
    checkErrNvrtc(err, "nvrtcCompileProgram()");

    size_t ptx_size;
    err = nvrtcGetPTXSize(program, &ptx_size);
    checkErrNvrtc(err, "nvrtcGetPTXSize()");

    char* ptx = new char[ptx_size];
    err = nvrtcGetPTX(program, ptx);
    checkErrNvrtc(err, "nvrtcGetPTX()");

    err = nvrtcDestroyProgram(&program);
    checkErrNvrtc(err, "nvrtcDestroyProgram()");

    // compile ptx
    create_module(dev, ptx, file_name, target_cc);
    delete[] ptx;
}
#else
#ifndef NVCC_BIN
#define NVCC_BIN "nvcc"
#endif
void compile_cuda(uint32_t dev, std::string file_name, CUjit_target target_cc) {
    target_cc = target_cc == CU_TARGET_COMPUTE_21 ? CU_TARGET_COMPUTE_20 : target_cc; // compute_21 does not exist for nvcc
    std::string ptx_filename = file_name + ".ptx";
    std::string command = (NVCC_BIN " -O4 -ptx -arch=compute_") + std::to_string(target_cc) + " ";
    command += std::string(KERNEL_DIR) + file_name + " -o " + ptx_filename + " 2>&1";

    if (auto stream = popen(command.c_str(), "r")) {
        std::vector<std::string> log;
        char line[FILENAME_MAX];

        while (fgets(line, sizeof(char) * FILENAME_MAX, stream)) {
            log.push_back(line);
        }

        int exit_status = pclose(stream);
        if (!WEXITSTATUS(exit_status)) {
            for (auto line : log)
                runtime_log(line);
        } else {
            for (auto line : log)
                std::cerr << line;
            exit(EXIT_FAILURE);
        }
    } else {
        perror("Problems with pipe");
        exit(EXIT_FAILURE);
    }

    std::ifstream src_file(ptx_filename);
    if (!src_file.is_open()) {
        std::cerr << "ERROR: Can't open PTX source file '" << ptx_filename << "'!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string src_string(std::istreambuf_iterator<char>(src_file), (std::istreambuf_iterator<char>()));
    const char* ptx = (const char*)src_string.c_str();

    // compile ptx
    create_module(dev, ptx, file_name, target_cc);
}
#endif

// create module
void load_module(uint32_t dev, std::string file_name, bool is_nvvm) {
    int major, minor;
    CUresult err = CUDA_SUCCESS;
    err = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devices_[dev]);
    checkErrDrv(err, "cuDeviceGetAttribute()");
    err = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devices_[dev]);
    checkErrDrv(err, "cuDeviceGetAttribute()");
    CUjit_target target_cc = (CUjit_target)(major*10 + minor);

    if (is_nvvm) {
        compile_nvvm(dev, file_name, target_cc);
    } else {
        compile_cuda(dev, file_name, target_cc);
    }
}


// load CUDA/NVVM source and compile kernel
void load_kernel(uint32_t dev, std::string file_name, std::string kernel_name, bool is_nvvm) {
    // get module from cache
    if (module_cache_[dev].count(file_name)) {
        auto module = modules_[dev] = module_cache_[dev][file_name];
        // get function from cache
        if (function_cache_[module].count(kernel_name)) {
            functions_[dev] = function_cache_[module][kernel_name];
            runtime_log("Compiling(", dev, ") '", kernel_name, "' ... returning old copy!\n");
            return;
        } else {
            // no function
            cuCtxPushCurrent(contexts_[dev]);
            create_kernel(dev, kernel_name);
            cuCtxPopCurrent(NULL);
        }
    } else {
        // no module; no function
        cuCtxPushCurrent(contexts_[dev]);
        load_module(dev, file_name, is_nvvm);
        create_kernel(dev, kernel_name);
        cuCtxPopCurrent(NULL);
    }
}


void unload_module(uint32_t dev, CUmodule module) {
    runtime_log("unload module: ", module, "\n");
    cuCtxPushCurrent(contexts_[dev]);
    CUresult err = cuModuleUnload(module);
    checkErrDrv(err, "cuUnloadModule()");
    cuCtxPopCurrent(NULL);
}


void get_tex_ref(uint32_t dev, std::string name) {
    cuCtxPushCurrent(contexts_[dev]);
    CUresult err = cuModuleGetTexRef(&textures_[dev], modules_[dev], name.c_str());
    checkErrDrv(err, "cuModuleGetTexRef('" + name + "')");
    cuCtxPopCurrent(NULL);
}

void bind_tex(uint32_t dev, mem_id mem, CUarray_format format) {
    void* host = mem_manager.get_host_mem(dev, mem);
    CUdeviceptr& dev_mem = mem_manager.get_dev_mem(dev, mem);

    assert(mem_manager.host_mem_size(host) && "memory not allocated by thorin");
    size_t host_size = mem_manager.host_mem_size(host);

    checkErrDrv(cuTexRefSetFormat(textures_[dev], format, 1), "cuTexRefSetFormat()");
    checkErrDrv(cuTexRefSetFlags(textures_[dev], CU_TRSF_READ_AS_INTEGER), "cuTexRefSetFlags()");
    checkErrDrv(cuTexRefSetAddress(0, textures_[dev], dev_mem, host_size), "cuTexRefSetAddress()");
}


CUdeviceptr malloc_memory(uint32_t dev, void* /*host*/, uint32_t size) {
    if (!size) return 0;

    cuCtxPushCurrent(contexts_[dev]);
    CUdeviceptr mem;

    // TODO: unified memory support using cuMemAllocManaged();
    CUresult err = cuMemAlloc(&mem, size);
    checkErrDrv(err, "cuMemAlloc()");

    cuCtxPopCurrent(NULL);
    return mem;
}


void free_memory(uint32_t dev, CUdeviceptr mem) {
    cuCtxPushCurrent(contexts_[dev]);
    CUresult err = cuMemFree(mem);
    checkErrDrv(err, "cuMemFree()");
    cuCtxPopCurrent(NULL);
}


void write_memory(uint32_t dev, CUdeviceptr mem, void* host, uint32_t size) {
    cuCtxPushCurrent(contexts_[dev]);
    CUresult err = cuMemcpyHtoD(mem, host, size);
    checkErrDrv(err, "cuMemcpyHtoD()");
    cuCtxPopCurrent(NULL);
}


void read_memory(uint32_t dev, CUdeviceptr mem, void* host, uint32_t size) {
    cuCtxPushCurrent(contexts_[dev]);
    CUresult err = cuMemcpyDtoH(host, mem, size);
    checkErrDrv(err, "cuMemcpyDtoH()");
    cuCtxPopCurrent(NULL);
}


void synchronize(uint32_t dev) {
    cuCtxPushCurrent(contexts_[dev]);
    CUresult err = cuCtxSynchronize();
    checkErrDrv(err, "cuCtxSynchronize()");
    cuCtxPopCurrent(NULL);
}


void set_problem_size(uint32_t /*dev*/, uint32_t size_x, uint32_t size_y, uint32_t size_z) {
    cuDimProblem.x = size_x;
    cuDimProblem.y = size_y;
    cuDimProblem.z = size_z;
}


void set_config_size(uint32_t /*dev*/, uint32_t size_x, uint32_t size_y, uint32_t size_z) {
    cuDimBlock.x = size_x;
    cuDimBlock.y = size_y;
    cuDimBlock.z = size_z;
}


void set_kernel_arg(uint32_t /*dev*/, void* param) {
    cuArgIdx++;
    if (cuArgIdx > cuArgIdxMax) {
        cuArgs = (void**)realloc(cuArgs, sizeof(void*)*cuArgIdx);
        cuArgIdxMax = cuArgIdx;
    }
    cuArgs[cuArgIdx-1] = (void*)malloc(sizeof(void*));
    cuArgs[cuArgIdx-1] = param;
}


void set_kernel_arg_map(uint32_t dev, mem_id mem) {
    CUdeviceptr& dev_mem = mem_manager.get_dev_mem(dev, mem);
    runtime_log(" * set arg map(", dev, "):   ", mem, "\n");
    set_kernel_arg(dev, &dev_mem);
}


void set_kernel_arg_const(uint32_t dev, void* param, std::string name, uint32_t size) {
    size_t bytes;
    CUdeviceptr const_mem;
    runtime_log(" * set arg const(", dev, "): ", param);
    CUresult err = cuModuleGetGlobal(&const_mem, &bytes, modules_[dev], name.c_str());
    checkErrDrv(err, "cuModuleGetGlobal('" + name + "')");
    write_memory(dev, const_mem, param, size);
}


extern std::atomic_llong thorin_kernel_time;

void launch_kernel(uint32_t dev, std::string kernel_name) {
    cuCtxPushCurrent(contexts_[dev]);
    CUevent start, end;
    unsigned int event_flags = CU_EVENT_DEFAULT;
    float time;

    // compute launch configuration
    dim3 grid;
    grid.x = cuDimProblem.x / cuDimBlock.x;
    grid.y = cuDimProblem.y / cuDimBlock.y;
    grid.z = cuDimProblem.z / cuDimBlock.z;

    cuEventCreate(&start, event_flags);
    cuEventCreate(&end, event_flags);

    // launch the kernel
    cuEventRecord(start, 0);
    CUresult err = cuLaunchKernel(functions_[dev], grid.x, grid.y, grid.z, cuDimBlock.x, cuDimBlock.y, cuDimBlock.z, 0, NULL, cuArgs, NULL);
    checkErrDrv(err, "cuLaunchKernel(" + kernel_name + ")");
    err = cuCtxSynchronize();
    checkErrDrv(err, "cuLaunchKernel(" + kernel_name + ")");

    cuEventRecord(end, 0);
    cuEventSynchronize(end);
    cuEventElapsedTime(&time, start, end);
    thorin_kernel_time.fetch_add(time * 1000);

    runtime_log("Kernel timing on device ", dev,
                " for '", kernel_name, "' (",
                cuDimProblem.x*cuDimProblem.y, ": ",
                cuDimProblem.x, "x", cuDimProblem.y, ", ",
                cuDimBlock.x*cuDimBlock.y, ": ",
                cuDimBlock.x, "x", cuDimBlock.y, "): ",
                time, " (ms)\n");
    print_kernel_occupancy(dev, kernel_name);

    cuEventDestroy(start);
    cuEventDestroy(end);

    // reset argument index
    cuArgIdx = 0;
    cuCtxPopCurrent(NULL);
}

// NVVM wrappers
mem_id nvvm_malloc_memory(uint32_t dev, void* host) { check_dev(dev); return mem_manager.malloc(dev, host); }
void nvvm_free_memory(uint32_t dev, mem_id mem) { check_dev(dev); mem_manager.free(dev, mem); }

void nvvm_write_memory(uint32_t dev, mem_id mem, void* host) { check_dev(dev); mem_manager.write(dev, mem, host); }
void nvvm_read_memory(uint32_t dev, mem_id mem, void* /*host*/) { check_dev(dev); mem_manager.read(dev, mem); }

void nvvm_load_nvvm_kernel(uint32_t dev, const char* file_name, const char* kernel_name) { check_dev(dev); load_kernel(dev, file_name, kernel_name, true); }
void nvvm_load_cuda_kernel(uint32_t dev, const char* file_name, const char* kernel_name) { check_dev(dev); load_kernel(dev, file_name, kernel_name, false); }

void nvvm_set_kernel_arg(uint32_t dev, void* param) { check_dev(dev); set_kernel_arg(dev, param); }
void nvvm_set_kernel_arg_map(uint32_t dev, mem_id mem) { check_dev(dev); set_kernel_arg_map(dev, mem); }
void nvvm_set_kernel_arg_tex(uint32_t dev, mem_id mem, const char* name, CUarray_format format) {
    check_dev(dev);
    runtime_log(" * set arg tex(", dev, "):   ", mem, "\n");
    get_tex_ref(dev, name);
    bind_tex(dev, mem, format);
}
void nvvm_set_kernel_arg_const(uint32_t dev, void* param, const char* name, uint32_t size) { check_dev(dev); set_kernel_arg_const(dev, param, name, size); }
void nvvm_set_problem_size(uint32_t dev, uint32_t size_x, uint32_t size_y, uint32_t size_z) { check_dev(dev); set_problem_size(dev, size_x, size_y, size_z); }
void nvvm_set_config_size(uint32_t dev, uint32_t size_x, uint32_t size_y, uint32_t size_z) { check_dev(dev); set_config_size(dev, size_x, size_y, size_z); }

void nvvm_launch_kernel(uint32_t dev, const char* kernel_name) { check_dev(dev); launch_kernel(dev, kernel_name); }
void nvvm_synchronize(uint32_t dev) { check_dev(dev); synchronize(dev); }

// helper functions
void thorin_init() { init_cuda(); }
void* thorin_malloc(uint32_t size) { return mem_manager.malloc_host(size); }
void thorin_free(void* ptr) { mem_manager.free_host(ptr); }

mem_id map_memory(uint32_t dev, uint32_t type_, void* from, int offset, int size) {
    check_dev(dev);
    mem_type type = (mem_type)type_;
    assert(mem_manager.host_mem_size(from) && "memory not allocated by thorin");
    size_t from_size = mem_manager.host_mem_size(from);
    if (size == 0) size = from_size-offset;

    mem_id mem = mem_manager.get_id(dev, from);
    if (mem) {
        mem_id id = mem_manager.malloc(dev, from, offset, size);
        runtime_log(" * map memory(", dev, ") -> malloc memory: ", id, "\n");
        return id;
    }

    if (type==Global || type==Texture) {
        if ((size_t)size == from_size) {
            // mapping the whole memory
            mem = mem_manager.malloc(dev, from);
            mem_manager.write(dev, mem, from);
            runtime_log(" * map memory(", dev, "):    ", from, " -> ", mem, "\n");
        } else {
            // mapping and slicing of a region
            mem = mem_manager.malloc(dev, from, offset, size);
            mem_manager.write(dev, mem, from);
            runtime_log(" * map memory(", dev, "):    ", from, " [", offset, ":", size, "] -> ", mem, "\n");
        }
    } else {
        std::cerr << "unsupported memory: " << type << std::endl;
        exit(EXIT_FAILURE);
    }

    return mem;
}
void unmap_memory(mem_id mem) { mem_manager.munmap(mem); }
