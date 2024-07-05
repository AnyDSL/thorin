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

struct DeviceBackends;

struct Backend {
    Backend(DeviceBackends& backends, World& src);

    Cont2Config& kernel_configs() { return kernel_configs_; };
    virtual std::unique_ptr<CodeGen> create_cg(const Cont2Config& config) = 0;

    Thorin& thorin() { return device_code_; }
    Importer& importer() { return *importer_; }

protected:
    DeviceBackends& backends_;
    Thorin device_code_;
    std::unique_ptr<Importer> importer_;

    std::vector<Continuation*> kernels_;
    Cont2Config kernel_configs_;

    void prepare_kernel_configs();
    friend DeviceBackends;
};

struct DeviceBackends {
    DeviceBackends(World& world, int opt, bool debug, std::string& hls_flags);

    World& world();
    std::vector<std::unique_ptr<CodeGen>> cgs;

    int opt();
    bool debug();

    void register_backend(std::unique_ptr<Backend>);
    using GetKernelConfigFn = std::function<std::unique_ptr<KernelConfig>(const App*, Continuation*)>;
    void register_intrinsic(Intrinsic, Backend&, GetKernelConfigFn);

private:
    World& world_;
    std::vector<std::unique_ptr<Backend>> backends_;
    std::unordered_map<Intrinsic, std::pair<Backend*, GetKernelConfigFn>> intrinsics_;

    int opt_;
    bool debug_;

    void search_for_device_code();
friend Backend;
};

}

#endif
