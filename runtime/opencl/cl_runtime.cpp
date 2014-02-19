#include "cl_runtime.h"

#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

//#define USE_SPIR

// global variables ...
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue command_queue;
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
        std::cerr << "timing: " << global_time * 1.0e-3f << "(ms)" << std::endl;
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


// create context and command queue for device
void create_context_command_queue() {
    cl_int err = CL_SUCCESS;

    // create context
    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    context = clCreateContext(cprops, 1, &device, NULL, NULL, &err);
    checkErr(err, "clCreateContext()");

    // create command queue
    command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkErr(err, "clCreateCommandQueue()");
}


// initialize OpenCL device
void init_opencl(cl_device_type dev_type=CL_DEVICE_TYPE_CPU) {
    char pnBuffer[1024], pvBuffer[1024], pv2Buffer[1024], pdBuffer[1024], pd2Buffer[1024], pd3Buffer[1024];
    int platform_number = -1, device_number = -1;
    cl_uint num_platforms, num_devices, num_devices_type;
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

        // get platform info for each platform
        for (unsigned int i=0; i<num_platforms; ++i) {
            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1024, &pnBuffer, NULL);
            err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 1024, &pvBuffer, NULL);
            err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 1024, &pv2Buffer, NULL);
            checkErr(err, "clGetPlatformInfo()");

            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
            err |= clGetDeviceIDs(platforms[i], dev_type, 0, NULL, &num_devices_type);

            // check if the requested device type was not found for this platform
            if (err != CL_DEVICE_NOT_FOUND) checkErr(err, "clGetDeviceIDs()");

            devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, &num_devices);
            checkErr(err, "clGetDeviceIDs()");

            // check if this platform has a device that supports SPIR
            bool has_spir = false;
            #ifndef USE_SPIR
            has_spir = true;
            #endif
            for (unsigned int j=0; j<num_devices; ++j) {
                cl_device_type this_dev_type;

                err = clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(this_dev_type), &this_dev_type, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, sizeof(pd3Buffer), &pd3Buffer, NULL);
                checkErr(err, "clGetDeviceInfo()");

                // use first device of desired type
                std::string extensions(pd3Buffer);
                size_t found = extensions.find("cl_khr_spir");
                if (found!=std::string::npos && (this_dev_type & dev_type)) {
                    has_spir = true;
                }
            }

            // use first platform supporting desired device type
            if (has_spir && platform_number==-1 && num_devices_type > 0) {
                std::cerr << "  [*] Platform Name: " << pnBuffer << std::endl;
                platform_number = i;
                platform = platforms[platform_number];
            } else {
                std::cerr << "  [ ] Platform Name: " << pnBuffer << std::endl;
            }
            std::cerr << "      Platform Vendor: " << pvBuffer << std::endl;
            std::cerr << "      Platform Version: " << pv2Buffer << std::endl;

            // get device info for each device
            for (unsigned int j=0; j<num_devices; ++j) {
                cl_device_type this_dev_type;
                cl_uint device_vendor_id;
                cl_bool has_unified;

                err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(pnBuffer), &pnBuffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(pvBuffer), &pvBuffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR_ID, sizeof(device_vendor_id), &device_vendor_id, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(this_dev_type), &this_dev_type, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(pdBuffer), &pdBuffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(pd2Buffer), &pd2Buffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, sizeof(pd3Buffer), &pd3Buffer, NULL);
                err |= clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(has_unified), &has_unified, NULL);
                checkErr(err, "clGetDeviceInfo()");

                // use first device of desired type
                std::string extensions(pd3Buffer);
                size_t found = extensions.find("cl_khr_spir");
                bool has_spir = found!=std::string::npos;
                #ifndef USE_SPIR
                has_spir = true;
                #endif
                if (has_spir && platform_number == (int)i && device_number == -1 && (this_dev_type & dev_type)) {
                    std::cerr << "      [*] ";
                    device = devices[j];
                    device_number = j;
                } else {
                    std::cerr << "      [ ] ";
                }
                std::cerr << "Device Name: " << pnBuffer << " (";
                if (this_dev_type & CL_DEVICE_TYPE_CPU) std::cerr << "CL_DEVICE_TYPE_CPU";
                if (this_dev_type & CL_DEVICE_TYPE_GPU) std::cerr << "CL_DEVICE_TYPE_GPU";
                if (this_dev_type & CL_DEVICE_TYPE_ACCELERATOR) std::cerr << "CL_DEVICE_TYPE_ACCELERATOR";
                #ifdef CL_VERSION_1_2
                if (this_dev_type & CL_DEVICE_TYPE_CUSTOM) std::cerr << "CL_DEVICE_TYPE_CUSTOM";
                #endif
                if (this_dev_type & CL_DEVICE_TYPE_DEFAULT) std::cerr << "|CL_DEVICE_TYPE_DEFAULT";
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

        if (platform_number == -1) {
            std::cerr << "No suitable OpenCL platform available, aborting ..." << std::endl;
            exit(EXIT_FAILURE);
        }

        free(platforms);
    }

    create_context_command_queue();

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
void build_program_and_kernel(const char *file_name, const char *kernel_name, bool is_binary) {
    cl_int err = CL_SUCCESS;
    bool print_progress = true;
    bool print_log = true;
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


    if (print_progress) std::cerr << "<Thorin:> Compiling '" << kernel_name << "' .";
    if (is_binary) {
        program = clCreateProgramWithBinary(context, 1, &device, &length, (const unsigned char **)&c_str, NULL, &err);
        checkErr(err, "clCreateProgramWithBinary()");
    } else {
        program = clCreateProgramWithSource(context, 1, (const char **)&c_str, &length, &err);
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
        err |= clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &options_size);
        err |= clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // allocate memory for the options and log
        char *program_build_options = (char *)malloc(options_size);
        char *program_build_log = (char *)malloc(log_size);

        // get the options and log
        err |= clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS, options_size, program_build_options, NULL);
        err |= clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, program_build_log, NULL);
        if (print_progress) {
            if (err != CL_SUCCESS) std::cerr << ". failed!" << std::endl;
            else std::cerr << "." << std::endl;
        }
        std::cerr << "<Thorin:> OpenCL build options : " << std::endl << program_build_options << std::endl;
        std::cerr << "<Thorin:> OpenCL build log : " << std::endl << program_build_log << std::endl;

        // free memory for options and log
        free(program_build_options);
        free(program_build_log);
    }
    checkErr(err, "clBuildProgram(), clGetProgramBuildInfo()");

    if (dump_binary) dump_program_binary(program, device);

    kernel = clCreateKernel(program, kernel_name, &err);
    checkErr(err, "clCreateKernel()");
    if (print_progress) std::cerr << ". done" << std::endl;
}


cl_mem malloc_buffer(size_t size) {
    cl_int err = CL_SUCCESS;
    cl_mem mem;
    cl_mem_flags flags = CL_MEM_READ_WRITE;

    mem = clCreateBuffer(context, flags, size * sizeof(float), NULL, &err);
    checkErr(err, "clCreateBuffer()");

    return mem;
}


void free_buffer(cl_mem mem) {
    cl_int err = CL_SUCCESS;

    err = clReleaseMemObject(mem);
    checkErr(err, "clReleaseMemObject()");
}


void write_buffer(cl_mem mem, void *host_mem, size_t size) {
    cl_int err = CL_SUCCESS;

    err = clEnqueueWriteBuffer(command_queue, mem, CL_FALSE, 0, size * sizeof(float), host_mem, 0, NULL, NULL);
    err |= clFinish(command_queue);
    checkErr(err, "clEnqueueWriteBuffer()");
}


void read_buffer(cl_mem mem, void *host_mem, size_t size) {
    cl_int err = CL_SUCCESS;

    err = clEnqueueReadBuffer(command_queue, mem, CL_FALSE, 0, size * sizeof(float), host_mem, 0, NULL, NULL);
    err |= clFinish(command_queue);
    checkErr(err, "clEnqueueReadBuffer()");
}


void synchronize() {
    cl_int err = CL_SUCCESS;

    err |= clFinish(command_queue);
    checkErr(err, "clFinish()");
}


void set_problem_size(size_t size_x, size_t size_y, size_t size_z) {
    global_work_size[0] = size_x;
    global_work_size[1] = size_y;
    global_work_size[2] = size_z;
}


void set_config_size(size_t size_x, size_t size_y, size_t size_z) {
    local_work_size[0] = size_x;
    local_work_size[1] = size_y;
    local_work_size[2] = size_z;
}


void set_kernel_arg(void *param, size_t size) {
    cl_int err = CL_SUCCESS;

    err = clSetKernelArg(kernel, clArgIdx++, size, param);
    checkErr(err, "clSetKernelArg()");
}


void launch_kernel(const char *kernel_name) {
    cl_int err = CL_SUCCESS;
    cl_event event;
    cl_ulong end, start;
    bool print_timing = true;

    getMicroTime();
    err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
    err |= clFinish(command_queue);
    getMicroTime();
    checkErr(err, "clEnqueueNDRangeKernel()");

    err = clWaitForEvents(1, &event);
    checkErr(err, "clWaitForEvents()");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    checkErr(err, "clGetEventProfilingInfo()");
    start *= 1e-3;
    end *= 1e-3;

    err = clReleaseEvent(event);
    checkErr(err, "clReleaseEvent()");

    if (print_timing) {
        std::cerr << "Kernel timing for '" << kernel_name << "' ("
                  << global_work_size[0]*global_work_size[1] << ": "
                  << global_work_size[0] << "x" << global_work_size[1] << ", "
                  << local_work_size[0]*local_work_size[1] << ": "
                  << local_work_size[0] << "x" << local_work_size[1] << "): "
                  << (end-start)*1.0e-3f << "(ms)" << std::endl;
    }

    // reset argument index
    clArgIdx = 0;
}

int main(int argc, char *argv[]) {
    init_opencl(CL_DEVICE_TYPE_GPU);

    return main_impala();
}

