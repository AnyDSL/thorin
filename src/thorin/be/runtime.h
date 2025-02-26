#ifndef THORIN_RUNTIME_H
#define THORIN_RUNTIME_H

/// Backend-agnostic information to interface with the runtime component
namespace thorin {

enum Platform {
    CPU_PLATFORM,
    CUDA_PLATFORM,
    OPENCL_PLATFORM,
    HSA_PLATFORM,
    PAL_PLATFORM,
    LEVEL_ZERO_PLATFORM,
    SHADY_PLATFORM,
};

enum Device_code {GPU, FPGA_HLS, FPGA_CL, AIE_CGRA};

template<Device_code T>
struct KernelLaunchArgs {};

template<>
struct KernelLaunchArgs<GPU> {
    enum {
        Mem = 0,
        Device,
        Space,
        Config,
        Body,
        Return,
        Num
    };
};

template<>
struct KernelLaunchArgs<FPGA_CL> : KernelLaunchArgs<GPU> {};

template<>
struct KernelLaunchArgs<AIE_CGRA> {
    enum {
        Mem = 0,
        Device,
        Runtime_ratio,
        Location,
        Vector_size,
        Body,
        Return,
        Num
    };
};

//TODO: KernelLaunchArgs<FPGA_HLS>?

}

#endif
