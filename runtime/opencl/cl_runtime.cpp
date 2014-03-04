#include "cl_runtime.h"

#include <time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include <cassert>
#include <fstream>
#include <iostream>
#include <unordered_map>

bool print_timing = true;

// define machine as seen/used by OpenCL
// each tuple consists of platform and device
#ifdef __MACH__
// Macbook Pro
const int num_devices_ = 3;
int the_machine[][2] = {
    {0, 0}, // Dummy
    {0, 0}, // Intel, i5-4288U
    {0, 1}, // Intel, Iris
};
#else
// Desktop
const int num_devices_ = 4;
int the_machine[][2] = {
    {0, 0}, // Dummy
    {1, 0}, // Intel, i7-3770K
    {2, 0}, // NVIDIA, GTX 680
    {2, 1}  // NVIDIA, GTX 580
};
#endif

// global variables ...
enum mem_type {
    Global      = 0,
    Texture     = 1,
    Constant    = 2,
    Shared      = 3
};

typedef struct array_t {
    void *mem;
    size_t id;
} array_t;

typedef struct mem_ {
    size_t elem;
    size_t width;
    size_t height;
} mem_;

std::unordered_map<void*, mem_> host_mems_;
std::unordered_map<cl_mem, void*> dev_mems_;
std::unordered_map<void*, cl_mem> dev_mems2_;
cl_device_id devices_[num_devices_];
cl_context contexts_[num_devices_];
cl_command_queue command_queues_[num_devices_];
cl_program program;
cl_kernel kernel;
int clArgIdx;
size_t local_work_size[3], global_work_size[3];

long global_time = 0;

void getMicroTime() {
    struct timespec now;
    #ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
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

const char *getOpenCLErrorCodeStr(int errorCode) {
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

#define checkErr(err, name)  __checkOpenCLErrors(err, name, __FILE__, __LINE__)

inline void __checkOpenCLErrors(cl_int err, const char *name, const char *file, const int line) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << " [file " << file << ", line " << line << "]: " << std::endl;
        std::cerr << getOpenCLErrorCodeStr(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


// create context and command queue(s) for device(s) of a given platform
void create_context_command_queue(cl_platform_id platform, cl_device_id *device, size_t num_devices, size_t num) {
    cl_int err = CL_SUCCESS;

    // create context
    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    contexts_[num] = clCreateContext(cprops, num_devices, device, NULL, NULL, &err);
    checkErr(err, "clCreateContext()");
    for (size_t i=1; i<num_devices; ++i) {
        contexts_[num+i] = contexts_[num];
    }

    // create command queues
    for (size_t i=0; i<num_devices; ++i) {
        command_queues_[num+i] = clCreateCommandQueue(contexts_[num+i], device[i], CL_QUEUE_PROFILING_ENABLE, &err);
        checkErr(err, "clCreateCommandQueue()");
    }
}


// initialize OpenCL device
void init_opencl() {
    char pnBuffer[1024], pvBuffer[1024], pv2Buffer[1024], pdBuffer[1024], pd2Buffer[1024], pd3Buffer[1024];
    cl_uint num_platforms, num_devices;
    cl_platform_id *platforms;
    cl_device_id *devices;
    cl_int err = CL_SUCCESS;


    // get OpenCL platform count
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    checkErr(err, "clGetPlatformIDs()");

    std::cerr << "Number of available Platforms: " << num_platforms << std::endl;
    if (num_platforms == 0) {
        exit(EXIT_FAILURE);
    } else {
        platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));

        err = clGetPlatformIDs(num_platforms, platforms, NULL);
        checkErr(err, "clGetPlatformIDs()");

        int n_dev = sizeof(the_machine)/sizeof(int[2]);
        int c_dev = 1;
        int c_pf_id = the_machine[c_dev][0];
        int c_dev_id = the_machine[c_dev][1];

        // get platform info for each platform
        for (unsigned int i=0; i<num_platforms; ++i) {
            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1024, &pnBuffer, NULL);
            err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 1024, &pvBuffer, NULL);
            err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 1024, &pv2Buffer, NULL);
            checkErr(err, "clGetPlatformInfo()");

            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
            checkErr(err, "clGetDeviceIDs()");

            devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, &num_devices);
            checkErr(err, "clGetDeviceIDs()");

            // use first platform supporting desired device type
            if (c_pf_id==i) {
                std::cerr << "  [*] Platform Name: " << pnBuffer << std::endl;
            } else {
                std::cerr << "  [ ] Platform Name: " << pnBuffer << std::endl;
            }
            std::cerr << "      Platform Vendor: " << pvBuffer << std::endl;
            std::cerr << "      Platform Version: " << pv2Buffer << std::endl;

            // get device info for each device
            for (unsigned int j=0; j<num_devices; ++j) {
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

                // use first device of desired type
                std::string extensions(pd3Buffer);
                size_t found = extensions.find("cl_khr_spir");
                bool has_spir = found!=std::string::npos;
                if (c_pf_id == i && j == c_dev_id) {
                    devices_[c_dev] = devices[j];
                    create_context_command_queue(platforms[i], &devices[j], 1, c_dev);
                    if (++c_dev < n_dev) {
                        c_pf_id = the_machine[c_dev][0];
                        c_dev_id = the_machine[c_dev][1];
                    }
                    std::cerr << "      [*] ";
                } else {
                    std::cerr << "      [ ] ";
                }
                std::cerr << "Device Name: " << pnBuffer << " (";
                if (dev_type & CL_DEVICE_TYPE_CPU) std::cerr << "CL_DEVICE_TYPE_CPU";
                if (dev_type & CL_DEVICE_TYPE_GPU) std::cerr << "CL_DEVICE_TYPE_GPU";
                if (dev_type & CL_DEVICE_TYPE_ACCELERATOR) std::cerr << "CL_DEVICE_TYPE_ACCELERATOR";
                #ifdef CL_VERSION_1_2
                if (dev_type & CL_DEVICE_TYPE_CUSTOM) std::cerr << "CL_DEVICE_TYPE_CUSTOM";
                #endif
                if (dev_type & CL_DEVICE_TYPE_DEFAULT) std::cerr << "|CL_DEVICE_TYPE_DEFAULT";
                std::cerr << ")" << std::endl;
                std::cerr << "          Device Vendor: " << pvBuffer << " (ID: " << device_vendor_id << ")" << std::endl;
                std::cerr << "          Device OpenCL Version: " << pdBuffer << std::endl;
                std::cerr << "          Device Driver Version: " << pd2Buffer << std::endl;
                //std::cerr << "          Device Extensions: " << pd3Buffer << std::endl;
                std::cerr << "          Device SPIR Support: " << has_spir << std::endl;
                #ifdef CL_DEVICE_SPIR_VERSIONS
                err = clGetDeviceInfo(devices[j], CL_DEVICE_SPIR_VERSIONS, sizeof(pd3Buffer), &pd3Buffer, NULL);
                checkErr(err, "clGetDeviceInfo()");
                std::cerr << "          Device SPIR Version: " << pd3Buffer << std::endl;
                #endif
                std::cerr << "          Device Host Unified Memory: " << has_unified << std::endl;
            }
            free(devices);
        }

        if (c_dev != n_dev) {
            std::cerr << "No suitable OpenCL platform available, aborting ..." << std::endl;
            exit(EXIT_FAILURE);
        }

        free(platforms);
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
    cl_device_id *devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
    err |= clGetProgramInfo(program, CL_PROGRAM_DEVICES, num_devices * sizeof(cl_device_id), devices, 0);

    // get the sizes of the binaries
    size_t *binary_sizes = (size_t *)malloc(num_devices * sizeof(size_t));
    err |= clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, num_devices * sizeof(size_t), binary_sizes, NULL);

    // get the binaries
    unsigned char **binary = (unsigned char **)malloc(num_devices * sizeof(unsigned char *));
    for (unsigned int i=0; i<num_devices; i++) {
        binary[i] = (unsigned char *)malloc(binary_sizes[i]);
    }
    err |= clGetProgramInfo(program, CL_PROGRAM_BINARIES,  sizeof(unsigned char *)*num_devices, binary, NULL);
    checkErr(err, "clGetProgramInfo()");

    for (unsigned int i=0; i<num_devices; i++) {
        if (devices[i] == device) {
            std::cerr << "OpenCL binary : " << std::endl;
            // binary can contain any character, emit char by char
            for (unsigned int n=0; n<binary_sizes[i]; n++) {
                std::cerr << binary[i][n];
            }
            std::cerr << std::endl;
        }
    }

    for (unsigned int i=0; i<num_devices; i++) {
        free(binary[i]);
    }
    free(binary);
    free(binary_sizes);
    free(devices);
}


// load OpenCL source file, build program, and create kernel
void build_program_and_kernel(size_t dev, const char *file_name, const char *kernel_name, bool is_binary) {
    cl_int err = CL_SUCCESS;
    bool print_progress = true;
    bool print_log = false;
    bool dump_binary = false;

    std::ifstream srcFile(file_name);
    if (!srcFile.is_open()) {
        std::cerr << "ERROR: Can't open"
                  << (is_binary?"SPIR binary":"OpenCL source")
                  << " file '" << file_name << "'!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string clString(std::istreambuf_iterator<char>(srcFile), (std::istreambuf_iterator<char>()));

    const size_t length = clString.length();
    const char *c_str = clString.c_str();


    if (print_progress) std::cerr << "Compiling(" << dev << ") '" << kernel_name << "' .";
    if (is_binary) {
        program = clCreateProgramWithBinary(contexts_[dev], 1, &devices_[dev], &length, (const unsigned char **)&c_str, NULL, &err);
        checkErr(err, "clCreateProgramWithBinary()");
    } else {
        program = clCreateProgramWithSource(contexts_[dev], 1, (const char **)&c_str, &length, &err);
        checkErr(err, "clCreateProgramWithSource()");
    }

    std::string options = "-cl-single-precision-constant -cl-denorms-are-zero";
    // according to the OpenCL specification -x spir -spir-std=1.2 has to be
    // specified as compiler option. however, these options are not recognized
    // by the Intel compiler
    //options += " -x spir -spir-std=1.2";

    err = clBuildProgram(program, 0, NULL, options.c_str(), NULL, NULL);
    if (print_progress) std::cerr << ".";

    if (err != CL_SUCCESS || print_log) {
        // determine the size of the options and log
        size_t log_size, options_size;
        err |= clGetProgramBuildInfo(program, devices_[dev], CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &options_size);
        err |= clGetProgramBuildInfo(program, devices_[dev], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // allocate memory for the options and log
        char *program_build_options = (char *)malloc(options_size);
        char *program_build_log = (char *)malloc(log_size);

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
        free(program_build_options);
        free(program_build_log);
    }
    checkErr(err, "clBuildProgram(), clGetProgramBuildInfo()");

    if (dump_binary) dump_program_binary(program, devices_[dev]);

    kernel = clCreateKernel(program, kernel_name, &err);
    checkErr(err, "clCreateKernel()");
    if (print_progress) std::cerr << ". done" << std::endl;
}


cl_mem malloc_buffer(size_t dev, void *host, cl_mem_flags flags) {
    cl_int err = CL_SUCCESS;
    cl_mem mem;
    mem_ info = host_mems_[host];

    void *host_ptr = NULL;
    if (flags & CL_MEM_USE_HOST_PTR) host_ptr = host;
    mem = clCreateBuffer(contexts_[dev], flags, info.elem * info.width * info.height, host_ptr, &err);
    checkErr(err, "clCreateBuffer()");
    std::cerr << " * malloc buffer(" << dev << "): dev " << mem << " <-> host: " << host << std::endl;
    dev_mems_[mem] = host;
    dev_mems2_[host] = mem;

    return mem;
}


void free_buffer(size_t dev, cl_mem mem) {
    cl_int err = CL_SUCCESS;

    err = clReleaseMemObject(mem);
    checkErr(err, "clReleaseMemObject()");
    void * host = dev_mems_[mem];
    dev_mems_.erase(mem);
    dev_mems2_.erase(host);
}


void write_buffer(size_t dev, cl_mem mem, void *host) {
    cl_int err = CL_SUCCESS;
    cl_event event;
    cl_ulong end, start;
    mem_ info = host_mems_[host];

    std::cerr << " * write_buffer(" << dev << "): " << mem << " <- " << host << std::endl;
    getMicroTime();
    err = clEnqueueWriteBuffer(command_queues_[dev], mem, CL_FALSE, 0, info.elem * info.width * info.height, host, 0, NULL, &event);
    err |= clFinish(command_queues_[dev]);
    checkErr(err, "clEnqueueWriteBuffer()");
    getMicroTime();

    if (print_timing) {
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
        err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
        checkErr(err, "clGetEventProfilingInfo()");
        std::cerr << "   timing for write_buffer: "
                  << (end-start)*1.0e-6f << "(ms)" << std::endl;
    }
}


void read_buffer(size_t dev, cl_mem mem, void *host) {
    cl_int err = CL_SUCCESS;
    cl_event event;
    cl_ulong end, start;
    mem_ info = host_mems_[host];

    std::cerr << " * read_buffer(" << dev << "): " << mem << " -> " << host << std::endl;
    getMicroTime();
    err = clEnqueueReadBuffer(command_queues_[dev], mem, CL_FALSE, 0, info.elem * info.width * info.height, host, 0, NULL, &event);
    err |= clFinish(command_queues_[dev]);
    checkErr(err, "clEnqueueReadBuffer()");
    getMicroTime();

    if (print_timing) {
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
        err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
        checkErr(err, "clGetEventProfilingInfo()");
        std::cerr << "   timing for read_buffer: "
                  << (end-start)*1.0e-6f << "(ms)" << std::endl;
    }
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
    cl_int err = CL_SUCCESS;

    std::cerr << " * set arg(" << dev << "): " << param << std::endl;

    err = clSetKernelArg(kernel, clArgIdx++, size, param);
    checkErr(err, "clSetKernelArg()");
}


void set_kernel_arg_map(size_t dev, void *param, size_t size) {
    cl_mem mem = dev_mems2_[param];

    std::cerr << " * set arg mapped(" << dev << "): " << param << " (map: " << mem << ")" << std::endl;
    set_kernel_arg(dev, &mem, sizeof(cl_mem));
}


void launch_kernel(size_t dev, const char *kernel_name) {
    cl_int err = CL_SUCCESS;
    cl_event event;
    cl_ulong end, start;

    getMicroTime();
    err = clEnqueueNDRangeKernel(command_queues_[dev], kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
    err |= clFinish(command_queues_[dev]);
    getMicroTime();
    checkErr(err, "clEnqueueNDRangeKernel()");

    err = clWaitForEvents(1, &event);
    checkErr(err, "clWaitForEvents()");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    checkErr(err, "clGetEventProfilingInfo()");

    err = clReleaseEvent(event);
    checkErr(err, "clReleaseEvent()");

    if (print_timing) {
        std::cerr << "Kernel timing on device " << dev
                  << " for '" << kernel_name << " ("
                  << global_work_size[0]*global_work_size[1] << ": "
                  << global_work_size[0] << "x" << global_work_size[1] << ", "
                  << local_work_size[0]*local_work_size[1] << ": "
                  << local_work_size[0] << "x" << local_work_size[1] << "): "
                  << (end-start)*1.0e-6f << "(ms)" << std::endl;
    }

    // reset argument index
    clArgIdx = 0;
}


// SPIR wrappers
cl_mem spir_malloc_buffer(size_t dev, void *host) { return malloc_buffer(dev, host); }
void spir_free_buffer(size_t dev, cl_mem mem) { free_buffer(dev, mem); }

void spir_write_buffer(size_t dev, cl_mem mem, void *host) { write_buffer(dev, mem, host); }
void spir_read_buffer(size_t dev, cl_mem mem, void *host) { read_buffer(dev, mem, host); }

void spir_build_program_and_kernel_from_binary(size_t dev, const char *file_name, const char *kernel_name) { build_program_and_kernel(dev, file_name, kernel_name, true); }
void spir_build_program_and_kernel_from_source(size_t dev, const char *file_name, const char *kernel_name) { build_program_and_kernel(dev, file_name, kernel_name, false); }

void spir_set_kernel_arg(size_t dev, void *host, size_t size) { set_kernel_arg(dev, host, size); }
void spir_set_kernel_arg_map(size_t dev, void *host, size_t size) { set_kernel_arg_map(dev, host, size); }
void spir_set_problem_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) { set_problem_size(dev, size_x, size_y, size_z); }
void spir_set_config_size(size_t dev, size_t size_x, size_t size_y, size_t size_z) { set_config_size(dev, size_x, size_y, size_z); }

void spir_launch_kernel(size_t dev, const char *kernel_name) { launch_kernel(dev, kernel_name); }
void spir_synchronize(size_t dev) { synchronize(dev); }

// helper functions
void *array(size_t elem_size, size_t width, size_t height) {
    void *mem = malloc(elem_size * width * height);
    std::cerr << " * array() -> " << mem << std::endl;
    host_mems_[mem] = {elem_size, width, height};
    return mem;
}
void *map_memory(size_t dev, size_t type_, void *from, size_t ox, size_t oy, size_t oz, size_t sx, size_t sy, size_t sz) {
    cl_int err = CL_SUCCESS;
    mem_type type = (mem_type)type_;
    mem_ info = host_mems_[from];

    assert(oz==0 && sz==0 && "3D memory not yet supported");

    if (type==Global) {
        std::cerr << " * map_memory(" << dev << "): from " << from << " " << std::endl;
        cl_mem_flags mem_flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
        cl_mem mem = malloc_buffer(dev, from, mem_flags);

        cl_event event;
        cl_ulong end, start;
        cl_bool blocking_map = CL_TRUE;
        cl_map_flags map_flags = CL_MAP_READ | CL_MAP_WRITE;
        void *mapped_mem = clEnqueueMapBuffer(command_queues_[dev], mem, blocking_map, map_flags, 0, info.elem * info.width * info.height, 0, NULL, &event, &err);
        checkErr(err, "clEnqueueMapBuffer()");
        if (print_timing) {
            err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
            err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
            checkErr(err, "clGetEventProfilingInfo()");
            std::cerr << "   timing for map_memory: "
                      << (end-start)*1.0e-6f << "(ms)" << std::endl;
        }
    } else {
        std::cerr << "unsupported memory: " << type << std::endl;
        exit(EXIT_FAILURE);
    }

    return from;
}
float random_val(int max) {
    return ((float)random() / RAND_MAX) * max;
}

int main(int argc, char *argv[]) {
    init_opencl();

    return main_impala();
}

