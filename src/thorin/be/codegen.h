#ifndef THORIN_CODEGEN_H
#define THORIN_CODEGEN_H

#include "thorin/transform/importer.h"
#include "thorin/be/kernel_config.h"

namespace thorin {

class CodeGen {
protected:
    CodeGen(Thorin& thorin, bool debug);
public:
    virtual ~CodeGen() {}

    virtual void emit_stream(std::ostream& stream) = 0;
    virtual const char* file_ext() const = 0;

    /// @name getters
    //@{
    Thorin& thorin() const { return thorin_; }
    World& world() const { return thorin().world(); }
    bool debug() const { return debug_; }
    //@}

private:
    Thorin& thorin_;
    bool debug_;
};

struct LaunchArgs {
    enum {
        Mem = 0,
        Device,
        Space,
        Config,
        LocalMem,
        Body,
        Return,
        Num
    };
};

struct DeviceBackends {
    DeviceBackends(World& world, int opt, bool debug, std::string& hls_flags);

    Cont2Config kernel_config;
    std::vector<Continuation*> kernels;

    enum { CUDA, NVVM, OpenCL, AMDGPU_HSA, AMDGPU_PAL, HLS, Shady, BackendCount };
    std::array<std::unique_ptr<CodeGen>, BackendCount> cgs;
private:
    std::array<const char*, BackendCount> backend_names = { "CUDA", "NVVM", "OpenCL", "AMDGPU_HSA", "AMDGPU_PAL", "HLS", "Shady" };
    std::vector<Thorin> accelerator_code;
};

}

#endif
