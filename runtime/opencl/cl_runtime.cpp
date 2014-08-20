#include "cl_runtime.h"
#include "thorin_utils.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#ifndef KERNEL_DIR
#define KERNEL_DIR ""
#endif

#define BENCH
#ifdef BENCH
std::vector<std::pair<size_t, void *>> kernel_args;
float total_timing = 0.0f;
#endif

bool print_timing = true;

// runtime forward declarations
cl_mem malloc_buffer(size_t dev, void *host, cl_mem_flags flags, size_t size);
void read_buffer(size_t dev, cl_mem mem, void *host, size_t size);
void write_buffer(size_t dev, cl_mem mem, void *host, size_t size);
void free_buffer(size_t dev, mem_id mem);

void build_program_and_kernel(size_t dev, std::string file_name, std::string kernel_name, bool);

void set_kernel_arg(size_t dev, void *param, size_t size);
void set_kernel_arg_map(size_t dev, mem_id mem);
void set_kernel_arg_const(size_t dev, void *param, size_t size);
void set_kernel_arg_struct(size_t dev, void *param, size_t size);
void set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);
void set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z);

void launch_kernel(size_t dev, std::string kernel_name);
void synchronize(size_t dev);


// global variables ...
enum mem_type {
    Generic     = 0,
    Global      = 1,
    Texture     = 2,
    Shared      = 3,
    Constant    = 4
};


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
            cl_mem gpu;
            mem_type type;
            size_t offset, size;
            mem_id id;
        } mapping_;
        std::vector<std::unordered_map<mem_id, mapping_>> mmap;
        std::vector<std::unordered_map<mem_id, void*>> idtomem;
        std::vector<std::unordered_map<void*, mem_id>> memtoid;
        std::unordered_map<size_t, size_t> ummap;
        std::unordered_map<void*, size_t> hostmem;

    public:
        Memory() : count_(42) {}

    void reserve(size_t num) {
        mmap.resize(mmap.size() + num);
        idtomem.resize(idtomem.size() + num);
        memtoid.resize(memtoid.size() + num);
    }
    void associate_device(size_t host_dev, size_t assoc_dev) {
        ummap[assoc_dev] = host_dev;
    }
    mem_id get_id(size_t dev, void *mem) { return memtoid[dev][mem]; }
    mem_id map_memory(size_t dev, void *from, cl_mem to, mem_type type,
            size_t offset, size_t size) {
        mem_id id = new_id();
        mapping_ mem_map = { from, to, type, offset, size, id };
        mmap[dev][id] = mem_map;
        insert(dev, from, id);
        return id;
    }
    mem_id map_memory(size_t dev, void *from, cl_mem to, mem_type type) {
        assert(hostmem.count(from) && "memory not allocated by thorin");
        return map_memory(dev, from, to, type, 0, hostmem[from]);
    }

    void *get_host_mem(size_t dev, mem_id id) { return mmap[dev][id].cpu; }
    cl_mem &get_dev_mem(size_t dev, mem_id id) { return mmap[dev][id].gpu; }

    void remove(size_t dev, mem_id id) {
        void *mem = idtomem[dev][id];
        memtoid[dev].erase(mem);
        idtomem[dev].erase(id);
    }

    void *malloc_host(size_t size) {
        void *mem;
        posix_memalign(&mem, 4096, size);
        std::cerr << " * malloc host(" << size << ") -> " << mem << std::endl;
        hostmem[mem] = size;
        return mem;
    }
    void free_host(void *ptr) {
        if (ptr==nullptr) return;
        // TODO: free associated device memory
        assert(hostmem.count(ptr) && "memory not allocated by thorin");
        std::cerr << " * free host(" << ptr << ")" << std::endl;
        hostmem.erase(ptr);
        // free host memory
        free(ptr);
    }
    size_t host_mem_size(void *ptr) {
        assert(hostmem.count(ptr) && "memory not allocated by thorin");
        return hostmem[ptr];
    }

    mem_id malloc(size_t dev, void *host, size_t offset, size_t size) {
        assert(hostmem.count(host) && "memory not allocated by thorin");
        mem_id id = get_id(dev, host);

        if (id) {
            std::cerr << " * malloc buffer(" << dev << "): returning old copy " << id << " for " << host << std::endl;
            return id;
        }

        id = get_id(ummap[dev], host);
        if (id) {
            std::cerr << " * malloc buffer(" << dev << "): returning old copy " << id << " from associated device " << ummap[dev] << " for " << host << std::endl;
            id = map_memory(dev, host, get_dev_mem(ummap[dev], id), Global, offset, size);
        } else {
            void *host_ptr = (char*)host + offset;
            // CL_MEM_ALLOC_HOST_PTR -> OpenCL allocates memory that can be shared - preferred on AMD hardware ?
            // CL_MEM_USE_HOST_PTR   -> use existing, properly aligned and sized memory
            cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
            cl_mem mem = malloc_buffer(dev, host_ptr, flags, size);
            id = map_memory(dev, host, mem, Global, offset, size);
            std::cerr << " * malloc buffer(" << dev << "): " << mem << " (" << id << ") <-> host: " << host << std::endl;
        }
        return id;
    }
    mem_id malloc(size_t dev, void *host) {
        assert(hostmem.count(host) && "memory not allocated by thorin");
        return malloc(dev, host, 0, hostmem[host]);
    }

    void read(size_t dev, mem_id id) {
        assert(mmap[dev][id].cpu && "invalid host memory");
        void *host = mmap[dev][id].cpu;
        std::cerr << " * read buffer(" << dev << "):   " << id << " -> " << host
                  << " [" << mmap[dev][id].offset << ":" << mmap[dev][id].size
                  << "]" << std::endl;
        void *host_ptr = (char*)host + mmap[dev][id].offset;
        read_buffer(dev, mmap[dev][id].gpu, host_ptr, mmap[dev][id].size);
    }
    void write(size_t dev, mem_id id, void *host) {
        assert(host==mmap[dev][id].cpu && "invalid host memory");
        std::cerr << " * write buffer(" << dev << "):  " << id << " <- " << host
                  << " [" << mmap[dev][id].offset << ":" << mmap[dev][id].size
                  << "]" << std::endl;
        void *host_ptr = (char*)host + mmap[dev][id].offset;
        write_buffer(dev, mmap[dev][id].gpu, host_ptr, mmap[dev][id].size);
    }
};
Memory mem_manager;

std::vector<cl_device_id> devices_;
std::vector<cl_context> contexts_;
std::vector<cl_command_queue> command_queues_;
cl_kernel kernel;
std::vector<cl_mem> kernel_structs_;
int clArgIdx;
size_t local_work_size[3], global_work_size[3];

std::string getOpenCLErrorCodeStr(int errorCode) {
    switch (errorCode) {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        #ifdef CL_VERSION_1_1
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        #endif
        #ifdef CL_VERSION_1_2
        case CL_COMPILE_PROGRAM_FAILURE:
            return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:
            return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:
            return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:
            return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        #endif
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        #ifdef CL_VERSION_1_1
        case CL_INVALID_PROPERTY:
            return "CL_INVALID_PROPERTY";
        #endif
        #ifdef CL_VERSION_1_2
        case CL_INVALID_IMAGE_DESCRIPTOR:
            return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:
            return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:
            return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";
        #endif
        default:
            return "unknown error code";
    }
}

#define check_dev(dev) __check_device(dev)
#define checkErr(err, name)  __checkOpenCLErrors(err, name, __FILE__, __LINE__)

inline void __checkOpenCLErrors(cl_int err, std::string name, std::string file, const int line) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << " [file " << file << ", line " << line << "]: " << std::endl;
        std::cerr << getOpenCLErrorCodeStr(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void __check_device(size_t dev) {
    if (dev >= devices_.size()) {
        std::cerr << "ERROR: requested device #" << dev << ", but only " << devices_.size() << " OpenCL devices [0.." << devices_.size()-1 << "] available!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// create context and command queue(s) for device(s) of a given platform
void create_context_command_queue(cl_platform_id platform, std::vector<cl_device_id> &&devices,  std::vector<size_t> &&device_ids) {
    if (devices.size() == 0) return;
    contexts_.resize(devices_.size());
    command_queues_.resize(devices_.size());

    // create context
    cl_int err = CL_SUCCESS;
    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_context context = clCreateContext(cprops, devices.size(), devices.data(), NULL, NULL, &err);
    mem_manager.associate_device(device_ids.front(), device_ids.front());
    checkErr(err, "clCreateContext()");

    for (size_t i : device_ids) {
        // associate context
        contexts_[i] = context;
        mem_manager.associate_device(device_ids.front(), i);
        // create command queue
        command_queues_[i] = clCreateCommandQueue(context, devices_[i], CL_QUEUE_PROFILING_ENABLE, &err);
        checkErr(err, "clCreateCommandQueue()");
    }
}


// initialize OpenCL device
void init_opencl() {
    char pnBuffer[1024], pvBuffer[1024], pv2Buffer[1024], pdBuffer[1024], pd2Buffer[1024], pd3Buffer[1024];
    cl_uint num_platforms, num_devices;
    cl_int err = CL_SUCCESS;


    // get OpenCL platform count
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    checkErr(err, "clGetPlatformIDs()");

    std::cerr << "Number of available Platforms: " << num_platforms << std::endl;
    if (num_platforms == 0) {
        exit(EXIT_FAILURE);
    } else {
        cl_platform_id *platforms = new cl_platform_id[num_platforms];

        err = clGetPlatformIDs(num_platforms, platforms, NULL);
        checkErr(err, "clGetPlatformIDs()");

        // get platform info for each platform
        for (size_t i=0; i<num_platforms; ++i) {
            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1024, &pnBuffer, NULL);
            err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 1024, &pvBuffer, NULL);
            err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 1024, &pv2Buffer, NULL);
            checkErr(err, "clGetPlatformInfo()");

            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
            checkErr(err, "clGetDeviceIDs()");

            cl_device_id *devices = new cl_device_id[num_devices];
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, &num_devices);
            checkErr(err, "clGetDeviceIDs()");

            mem_manager.reserve(num_devices);
            std::vector<cl_device_id> unified_devices;
            std::vector<size_t> unified_device_ids;

            // get platform info
            std::cerr << "  Platform Name: " << pnBuffer << std::endl;
            std::cerr << "  Platform Vendor: " << pvBuffer << std::endl;
            std::cerr << "  Platform Version: " << pv2Buffer << std::endl;

            // get device info for each device
            for (size_t j=0; j<num_devices; ++j) {
                cl_device_type dev_type;
                cl_uint device_vendor_id;
                cl_bool has_unified;

                err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(pnBuffer), &pnBuffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(pvBuffer), &pvBuffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR_ID, sizeof(device_vendor_id), &device_vendor_id, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(pdBuffer), &pdBuffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(pd2Buffer), &pd2Buffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, sizeof(pd3Buffer), &pd3Buffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(has_unified), &has_unified, NULL);
                checkErr(err, "clGetDeviceInfo()");

                devices_.push_back(devices[j]);
                if (has_unified) {
                    unified_devices.push_back(devices_.back());
                    unified_device_ids.push_back(devices_.size()-1);
                } else {
                    create_context_command_queue(platforms[i], { devices_.back() }, { devices_.size()-1 } );
                }

                // use first device of desired type
                std::string extensions(pd3Buffer);
                size_t found = extensions.find("cl_khr_spir");
                bool has_spir = found!=std::string::npos;
                std::cerr << "  (" << devices_.size()-1 << ") ";
                std::cerr << "Device Name: " << pnBuffer << " (";
                if (dev_type & CL_DEVICE_TYPE_CPU) std::cerr << "CL_DEVICE_TYPE_CPU";
                if (dev_type & CL_DEVICE_TYPE_GPU) std::cerr << "CL_DEVICE_TYPE_GPU";
                if (dev_type & CL_DEVICE_TYPE_ACCELERATOR) std::cerr << "CL_DEVICE_TYPE_ACCELERATOR";
                #ifdef CL_VERSION_1_2
                if (dev_type & CL_DEVICE_TYPE_CUSTOM) std::cerr << "CL_DEVICE_TYPE_CUSTOM";
                #endif
                if (dev_type & CL_DEVICE_TYPE_DEFAULT) std::cerr << "|CL_DEVICE_TYPE_DEFAULT";
                std::cerr << ")";
                std::cerr << std::endl;
                std::cerr << "      Device Vendor: " << pvBuffer << " (ID: " << device_vendor_id << ")" << std::endl;
                std::cerr << "      Device OpenCL Version: " << pdBuffer << std::endl;
                std::cerr << "      Device Driver Version: " << pd2Buffer << std::endl;
                //std::cerr << "      Device Extensions: " << pd3Buffer << std::endl;
                std::cerr << "      Device SPIR Support: " << has_spir << std::endl;
                #ifdef CL_DEVICE_SPIR_VERSIONS
                err = clGetDeviceInfo(devices[j], CL_DEVICE_SPIR_VERSIONS, sizeof(pd3Buffer), &pd3Buffer, NULL);
                checkErr(err, "clGetDeviceInfo()");
                std::cerr << "      Device SPIR Version: " << pd3Buffer << std::endl;
                #endif
                std::cerr << "      Device Host Unified Memory: " << has_unified << std::endl;
            }

            // create context and command queues for unified memory devices
            create_context_command_queue(platforms[i], std::move(unified_devices), std::move(unified_device_ids));
            delete[] devices;
        }
        delete[] platforms;
    }

    // initialize clArgIdx
    clArgIdx = 0;
    local_work_size[0] = 256;
    local_work_size[1] = 1;
    local_work_size[2] = 1;
    global_work_size[0] = 0;
    global_work_size[1] = 0;
    global_work_size[2] = 0;
}


// get binary from OpenCL program and dump it to stderr
void dump_program_binary(cl_program program, cl_device_id device) {
    cl_int err = CL_SUCCESS;
    cl_uint num_devices;

    // get the number of devices associated with the program
    err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);

    // get the associated device ids
    std::vector<cl_device_id> devices(num_devices);
    err |= clGetProgramInfo(program, CL_PROGRAM_DEVICES, devices.size() * sizeof(cl_device_id), devices.data(), 0);

    // get the sizes of the binaries
    std::vector<size_t> binary_sizes(num_devices);
    err |= clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, binary_sizes.size() * sizeof(size_t), binary_sizes.data(), NULL);

    // get the binaries
    std::vector<unsigned char *> binaries(num_devices);
    for (size_t i=0; i<binaries.size(); ++i) {
        binaries[i] = new unsigned char[binary_sizes[i]];
    }
    err |= clGetProgramInfo(program, CL_PROGRAM_BINARIES,  sizeof(unsigned char *)*binaries.size(), binaries.data(), NULL);
    checkErr(err, "clGetProgramInfo()");

    for (size_t i=0; i<devices.size(); ++i) {
        if (devices[i] == device) {
            std::cerr << "OpenCL binary : " << std::endl;
            // binary can contain any character, emit char by char
            for (size_t n=0; n<binary_sizes[i]; ++n) {
                std::cerr << binaries[i][n];
            }
            std::cerr << std::endl;
        }
    }

    for (size_t i=0; i<num_devices; i++) {
        delete[] binaries[i];
    }
}


// load OpenCL source file, build program, and create kernel
void build_program_and_kernel(size_t dev, std::string file_name, std::string kernel_name, bool is_binary) {
    cl_program program;
    cl_int err = CL_SUCCESS;
    bool print_progress = true;
    bool print_log = false;
    bool dump_binary = false;

    std::ifstream srcFile(std::string(KERNEL_DIR) + file_name);
    if (!srcFile.is_open()) {
        std::cerr << "ERROR: Can't open "
                  << (is_binary?"SPIR binary":"OpenCL source")
                  << " file '" << file_name << "'!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string clString(std::istreambuf_iterator<char>(srcFile), (std::istreambuf_iterator<char>()));
    std::string options = "-cl-single-precision-constant -cl-denorms-are-zero";

    const size_t length = clString.length();
    const char *c_str = clString.c_str();

    if (print_progress) std::cerr << "Compiling(" << dev << ") '" << kernel_name << "' .";
    if (is_binary) {
        options += " -x spir -spir-std=1.2";
        program = clCreateProgramWithBinary(contexts_[dev], 1, &devices_[dev], &length, (const unsigned char **)&c_str, NULL, &err);
        checkErr(err, "clCreateProgramWithBinary()");
    } else {
        program = clCreateProgramWithSource(contexts_[dev], 1, (const char **)&c_str, &length, &err);
        checkErr(err, "clCreateProgramWithSource()");
    }

    err = clBuildProgram(program, 0, NULL, options.c_str(), NULL, NULL);
    if (print_progress) std::cerr << ".";

    cl_build_status build_status;
    clGetProgramBuildInfo(program, devices_[dev], CL_PROGRAM_BUILD_STATUS, sizeof(build_status), &build_status, NULL);

    if (build_status == CL_BUILD_ERROR || err != CL_SUCCESS || print_log) {
        // determine the size of the options and log
        size_t log_size, options_size;
        err |= clGetProgramBuildInfo(program, devices_[dev], CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &options_size);
        err |= clGetProgramBuildInfo(program, devices_[dev], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // allocate memory for the options and log
        char *program_build_options = new char[options_size];
        char *program_build_log = new char[log_size];

        // get the options and log
        err |= clGetProgramBuildInfo(program, devices_[dev], CL_PROGRAM_BUILD_OPTIONS, options_size, program_build_options, NULL);
        err |= clGetProgramBuildInfo(program, devices_[dev], CL_PROGRAM_BUILD_LOG, log_size, program_build_log, NULL);
        if (print_progress) {
            if (err != CL_SUCCESS) std::cerr << ". failed!" << std::endl;
            else std::cerr << ".";
        }
        std::cerr << std::endl
                  << "OpenCL build options : "
                  << program_build_options << std::endl
                  << "OpenCL build log : " << std::endl
                  << program_build_log << std::endl;

        // free memory for options and log
        delete[] program_build_options;
        delete[] program_build_log;
    }
    checkErr(err, "clBuildProgram(), clGetProgramBuildInfo()");

    if (dump_binary) dump_program_binary(program, devices_[dev]);

    kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    checkErr(err, "clCreateKernel()");
    if (print_progress) std::cerr << ". done" << std::endl;

    // release program
    err = clReleaseProgram(program);
    checkErr(err, "clReleaseProgram()");
}


cl_mem malloc_buffer(size_t dev, void *host, cl_mem_flags flags, size_t size) {
    if (!size) return 0;

    cl_int err = CL_SUCCESS;
    void *host_ptr = nullptr;

    if (flags & CL_MEM_USE_HOST_PTR || flags & CL_MEM_COPY_HOST_PTR) host_ptr = (char*)host;
    cl_mem mem = clCreateBuffer(contexts_[dev], flags, size, host_ptr, &err);
    checkErr(err, "clCreateBuffer()");

    return mem;
}


void free_buffer(size_t dev, mem_id mem) {
    cl_int err = CL_SUCCESS;

    cl_mem dev_mem = mem_manager.get_dev_mem(dev, mem);
    err = clReleaseMemObject(dev_mem);
    checkErr(err, "clReleaseMemObject()");
    mem_manager.remove(dev, mem);
}


void write_buffer(size_t dev, cl_mem mem, void *host, size_t size) {
    cl_int err = CL_SUCCESS;
    cl_event event;
    cl_ulong end, start;

    auto time = thorin_get_micro_time();
    err = clEnqueueWriteBuffer(command_queues_[dev], mem, CL_FALSE, 0, size, host, 0, NULL, &event);
    err |= clFinish(command_queues_[dev]);
    checkErr(err, "clEnqueueWriteBuffer()");
    thorin_print_micro_time(thorin_get_micro_time() - time);

    if (print_timing) {
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
        err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
        checkErr(err, "clGetEventProfilingInfo()");
        std::cerr << "   timing for write buffer: "
                  << (end-start)*1.0e-6f << "(ms)" << std::endl;
    }
    err = clReleaseEvent(event);
    checkErr(err, "clReleaseEvent()");
}


void read_buffer(size_t dev, cl_mem mem, void *host, size_t size) {
    cl_int err = CL_SUCCESS;
    cl_event event;
    cl_ulong end, start;

    auto time = thorin_get_micro_time();
    err = clEnqueueReadBuffer(command_queues_[dev], mem, CL_FALSE, 0, size, host, 0, NULL, &event);
    err |= clFinish(command_queues_[dev]);
    checkErr(err, "clEnqueueReadBuffer()");
    thorin_print_micro_time(thorin_get_micro_time() - time);

    if (print_timing) {
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
        err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
        checkErr(err, "clGetEventProfilingInfo()");
        std::cerr << "   timing for read buffer: "
                  << (end-start)*1.0e-6f << "(ms)" << std::endl;
    }
    err = clReleaseEvent(event);
    checkErr(err, "clReleaseEvent()");
}


void synchronize(size_t dev) {
    cl_int err = CL_SUCCESS;

    err |= clFinish(command_queues_[dev]);
    checkErr(err, "clFinish()");
}


void set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) {
    global_work_size[0] = size_x;
    global_work_size[1] = size_y;
    global_work_size[2] = size_z;
}


void set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) {
    local_work_size[0] = size_x;
    local_work_size[1] = size_y;
    local_work_size[2] = size_z;
}


void set_kernel_arg(size_t dev, void *param, size_t size) {
    //std::cerr << " * set arg(" << dev << "):       " << param << std::endl;
    #ifdef BENCH
    kernel_args.emplace_back(size, param);
    #else
    cl_int err = clSetKernelArg(kernel, clArgIdx++, size, param);
    checkErr(err, "clSetKernelArg()");
    #endif
}


void set_kernel_arg_map(size_t dev, mem_id mem) {
    cl_mem &dev_mem = mem_manager.get_dev_mem(dev, mem);
    std::cerr << " * set arg map(" << dev << "):   " << mem << std::endl;
    set_kernel_arg(dev, &dev_mem, sizeof(dev_mem));
}


void set_kernel_arg_const(size_t dev, void *param, size_t size) {
    cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    cl_mem const_buf = malloc_buffer(dev, param, flags, size);
    kernel_structs_.push_back(const_buf);
    cl_mem &buf = kernel_structs_.back();
    set_kernel_arg(dev, &buf, sizeof(cl_mem));
}


void set_kernel_arg_struct(size_t dev, void *param, size_t size) {
    cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
    cl_mem struct_buf = malloc_buffer(dev, param, flags, size);
    kernel_structs_.push_back(struct_buf);
    cl_mem &buf = kernel_structs_.back();
    std::cerr << " * set arg struct(" << dev << "): " << buf << std::endl;
    set_kernel_arg(dev, &buf, sizeof(cl_mem));
}


void launch_kernel(size_t dev, std::string kernel_name) {
    cl_int err = CL_SUCCESS;
    cl_event event;
    cl_ulong end, start;
    float time;

    // launch the kernel
    #ifdef BENCH
    std::vector<float> timings;
    for (size_t iter=0; iter<7; ++iter) {
        // set kernel arguments
        for (size_t j=0; j<kernel_args.size(); ++j) {
            err |= clSetKernelArg(kernel, j, kernel_args[j].first, kernel_args[j].second);
        }
        checkErr(err, "clSetKernelArg()");
    #endif
    err = clEnqueueNDRangeKernel(command_queues_[dev], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
    err |= clFinish(command_queues_[dev]);
    checkErr(err, "clEnqueueNDRangeKernel()");

    err = clWaitForEvents(1, &event);
    checkErr(err, "clWaitForEvents()");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    checkErr(err, "clGetEventProfilingInfo()");
    time = (end-start)*1.0e-6f;
    #ifdef BENCH
    timings.emplace_back(time);
    }
    kernel_args.clear();
    std::sort(timings.begin(), timings.end());
    total_timing += timings[timings.size()/2];
    #endif

    err = clReleaseEvent(event);
    checkErr(err, "clReleaseEvent()");

    if (print_timing) {
        std::cerr << "Kernel timing on device " << dev
                  << " for '" << kernel_name << "' ("
                  << global_work_size[0]*global_work_size[1] << ": "
                  << global_work_size[0] << "x" << global_work_size[1] << ", "
                  << local_work_size[0]*local_work_size[1] << ": "
                  << local_work_size[0] << "x" << local_work_size[1] << "): "
                  #ifdef BENCH
                  << "median of " << timings.size() << " runs: "
                  << timings[timings.size()/2]
                  #else
                  << time
                  #endif
                  << "(ms)" << std::endl;
    }

    // release kernel
    err = clReleaseKernel(kernel);
    checkErr(err, "clReleaseKernel()");
    // release temporary buffers for struct arguments
    for (cl_mem buf : kernel_structs_) {
        err = clReleaseMemObject(buf);
        checkErr(err, "clReleaseMemObject()");
    }
    kernel_structs_.clear();
    // reset argument index
    clArgIdx = 0;
}


// SPIR wrappers
mem_id spir_malloc_buffer(size_t dev, void *host) { check_dev(dev); return mem_manager.malloc(dev, host); }
void spir_free_buffer(size_t dev, mem_id mem) { check_dev(dev); free_buffer(dev, mem); }

void spir_write_buffer(size_t dev, mem_id mem, void *host) { check_dev(dev); mem_manager.write(dev, mem, host); }
void spir_read_buffer(size_t dev, mem_id mem, void *host) { check_dev(dev); mem_manager.read(dev, mem); }

void spir_build_program_and_kernel_from_binary(size_t dev, const char *file_name, const char *kernel_name) { check_dev(dev); build_program_and_kernel(dev, file_name, kernel_name, true); }
void spir_build_program_and_kernel_from_source(size_t dev, const char *file_name, const char *kernel_name) { check_dev(dev); build_program_and_kernel(dev, file_name, kernel_name, false); }

void spir_set_kernel_arg(size_t dev, void *param, size_t size) { check_dev(dev); set_kernel_arg(dev, param, size); }
void spir_set_kernel_arg_map(size_t dev, mem_id mem) { check_dev(dev); set_kernel_arg_map(dev, mem); }
void spir_set_kernel_arg_const(size_t dev, void *param, size_t size) { check_dev(dev); set_kernel_arg_const(dev, param, size); }
void spir_set_kernel_arg_struct(size_t dev, void *param, size_t size) { check_dev(dev); set_kernel_arg_struct(dev, param, size); }
void spir_set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) { check_dev(dev); set_problem_size(dev, size_x, size_y, size_z); }
void spir_set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) { check_dev(dev); set_config_size(dev, size_x, size_y, size_z); }

void spir_launch_kernel(size_t dev, const char *kernel_name) { check_dev(dev); launch_kernel(dev, kernel_name); }
void spir_synchronize(size_t dev) { check_dev(dev); synchronize(dev); }

// helper functions
void thorin_init() { init_opencl(); }
void *thorin_malloc(size_t size) { return mem_manager.malloc_host(size); }
void thorin_free(void *ptr) { mem_manager.free_host(ptr); }
void thorin_print_total_timing() {
    #ifdef BENCH
    std::cerr << "total accumulated timing: " << total_timing << " (ms)" << std::endl;
    #endif
}
mem_id map_memory(size_t dev, size_t type_, void *from, int offset, int size) {
    check_dev(dev);
    mem_type type = (mem_type)type_;
    assert(mem_manager.host_mem_size(from) && "memory not allocated by thorin");
    size_t from_size = mem_manager.host_mem_size(from);

    mem_id mem = mem_manager.get_id(dev, from);
    if (mem) {
        std::cerr << " * map memory(" << dev << "):    returning old copy " << mem << " for " << from << std::endl;
        return mem;
    }

    if (type==Global || type==Texture) {
        if (size == from_size) {
            // mapping the whole memory
            mem = mem_manager.malloc(dev, from);
            mem_manager.write(dev, mem, from);
            std::cerr << " * map memory(" << dev << "):    " << from << " -> " << mem << std::endl;

            #if 0
            cl_event event;
            cl_ulong end, start;
            cl_bool blocking_map = CL_TRUE;
            //cl_map_flags map_flags = CL_MAP_READ | CL_MAP_WRITE;
            cl_map_flags map_flags = CL_MAP_READ;
            cl_mem dev_mem = mem_manager.get_dev_mem(dev, mem);
            cl_int err = CL_SUCCESS;
            void *mapped_mem = clEnqueueMapBuffer(command_queues_[dev], dev_mem, blocking_map, map_flags, 0, size, 0, NULL, &event, &err);
            checkErr(err, "clEnqueueMapBuffer()");
            if (print_timing) {
                err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
                err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
                checkErr(err, "clGetEventProfilingInfo()");
                std::cerr << "   timing for map_memory: "
                          << (end-start)*1.0e-6f << "(ms)" << std::endl;
            }
            err = clReleaseEvent(event);
            checkErr(err, "clReleaseEvent()");
            #endif
        } else {
            // mapping and slicing of a region
            mem = mem_manager.malloc(dev, from, offset, size);
            mem_manager.write(dev, mem, from);
            std::cerr << " * map memory(" << dev << "):    " << from << " (" << offset << "x(" << size << ") -> " << mem << std::endl;

            #if 0
            cl_mem_flags mem_flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
            cl_buffer_region buffer_region = { offset, size };
            cl_mem sub_buffer = clCreateSubBuffer(buffer, mem_flags, CL_BUFFER_CREATE_TYPE_REGION, buffer_region, err);
            checkErr(err, "clCreateSubBuffer()");
            #endif
        }
    } else {
        std::cerr << "unsupported memory: " << type << std::endl;
        exit(EXIT_FAILURE);
    }

    return mem;
}
void unmap_memory(size_t dev, size_t type_, mem_id mem) {
    check_dev(dev);
    mem_manager.read(dev, mem);
    std::cerr << " * unmap memory(" << dev << "):  " << mem << std::endl;
    // TODO: mark device memory as unmapped
}
