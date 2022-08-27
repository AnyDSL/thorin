#ifndef THORIN_CODEGEN_H
#define THORIN_CODEGEN_H

#include "thorin/transform/importer.h"
#include "thorin/be/kernel_config.h"

namespace thorin {

class CodeGen {
protected:
    CodeGen(World& world, bool debug);
public:
    virtual ~CodeGen() {}

    virtual void emit_stream(std::ostream& stream) = 0;
    virtual const char* file_ext() const = 0;

    /// @name getters
    //@{
    World& world() const { return world_; }
    bool debug() const { return debug_; }
    //@}

private:
    World& world_;
    bool debug_;
};

enum device_code {GPU, FPGA_HLS, FPGA_CL, AIE_CGRA};
template<device_code T>
struct LaunchArgs {};
template <>
struct LaunchArgs<GPU> {
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
struct LaunchArgs<AIE_CGRA> {
    enum {
        Mem = 0,
        Device,
        Body,
        Return,
        Num
    };
};

struct DeviceBackends {
    DeviceBackends(World& world, int opt, bool debug, std::string& hls_flags);

    Cont2Config kernel_config;
    std::vector<Continuation*> kernels;

    enum { CUDA, NVVM, OpenCL, AMDGPU, CGRA, HLS, BackendCount };
    std::array<std::unique_ptr<CodeGen>, BackendCount> cgs;
private:
    std::vector<Importer> importers_;
};

}

#endif
