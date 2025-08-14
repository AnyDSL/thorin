#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/hls_channels.h"
#include "thorin/be/emitter.h"
#include "thorin/util/stream.h"
#include "c.h"

#include <cctype>
#include <cmath>
#include <regex>
#include <sstream>
#include <type_traits>
#include <unordered_map> // TODO don't use std::unordered_*
#include <variant>

namespace thorin::c {

struct BB {
    BB() = default;

    Continuation* cont = nullptr;
    StringStream head;
    StringStream body;
    StringStream tail;

    friend void swap(BB& a, BB& b) {
        using std::swap;
        swap(a.cont, b.cont);
        swap(a.head, b.head);
        swap(a.body, b.body);
        swap(a.tail, b.tail);
    }
};

using FuncMode = ChannelMode;

enum class CLDialect : uint8_t {
    STD    = 0, ///< Standard OpenCL
    INTEL  = 1, ///< Intel FPGA extension
    XILINX = 2  ///< Xilinx FPGA extension
};

inline std::string cl_dialect_guard(CLDialect dialect) {
    switch (dialect) {
        case CLDialect::STD:    return "STD_OPENCL";
        case CLDialect::INTEL:  return "INTELFPGA_CL";
        case CLDialect::XILINX: return "__xilinx__";
        default: THORIN_UNREACHABLE;
    }
}

template<typename Fn>
inline std::string guarded_statement(const std::string guard, Fn fn) {
    StringStream s;
    s.fmt("#ifdef {}\n", guard);
    fn(s);
    s.fmt("#endif\n");
    return s.str();
}

enum class HlsInterface : uint8_t {
    SOC,        ///< SoC HW module (Embedded)
    HPC,        ///< HPC accelerator (HLS for HPC via OpenCL/XRT + XDMA)
    HPC_STREAM, ///< HPC accelerator (HLS for HPC via XRT + QDMA)
    GMEM_OPT,   ///< Dedicated global memory interfaces and memory banks
    None
};

class CCodeGen : public thorin::Emitter<std::string, std::string, BB, CCodeGen> {
public:
    CCodeGen(Thorin& thorin, const Cont2Config& kernel_config, Stream& stream, Lang lang, bool debug, std::string& flags)
        : thorin_(thorin)
        , forest_(world())
        , kernel_config_(kernel_config)
        , lang_(lang)
        , fn_mem_(world().fn_type({world().mem_type()}))
        , debug_(debug)
        , flags_(flags)
        , stream_(stream)
    {}

    World& world() const { return thorin_.world(); }
    void emit_module();
    void emit_c_int();
    void emit_epilogue(Continuation*);

    std::string emit_bb(BB&, const Def*);
    std::string emit_constant(const Def*);
    std::string emit_bottom(const Type*);
    std::string emit_def(BB*, const Def*);
    void emit_jump(BB& bb, const Def* callee, ArrayRef<std::string>);
    void emit_access(Stream&, const Type*, const Def*, const std::string_view& = ".");
    bool is_valid(const std::string& s) { return !s.empty(); }
    std::string emit_fun_head(Continuation*, bool = false);
    std::string emit_fun_decl(Continuation*);
    std::string prepare(const Scope&);
    void prepare(Continuation*, const std::string&);
    void finalize(const Scope&);
    void finalize(Continuation*);

private:
    void convert_primtype(StringStream&s, PrimTypeTag tag, int len);
    std::string convert(const Type*);
    std::string addr_space_prefix(AddrSpace);
    std::string constructor_prefix(const Type*);
    std::string device_prefix();
    Stream& emit_debug_info(Stream&, const Def*);
    const Type* mangle_return_type(const ReturnType* return_type);
    bool get_interface(HlsInterface &interface, HlsInterface &gmem);
    const Param* get_channel_read_output(Continuation*);

    template <typename T, typename IsInfFn, typename IsNanFn>
    std::string emit_float(T, IsInfFn, IsNanFn);

    std::string array_name(const DefiniteArrayType*);
    std::string tuple_name(const TupleType*);
    std::string closure_name(const ClosureType*);
    std::string return_name(const ReturnType*);
    std::string fn_name(const FnType*);

    Thorin& thorin_;
    ScopesForest forest_;
    const Cont2Config& kernel_config_;
    Lang lang_;
    const FnType* fn_mem_;
    bool use_math_ = false;
    bool use_fp_64_ = false;
    bool use_fp_16_ = false;
    bool use_channels_ = false;
    bool use_align_of_ = false;
    bool use_memcpy_ = false;
    bool use_longjmp_ = false;
    bool use_malloc_ = false;
    bool debug_;
    std::string flags_;
    Stream& stream_;

    StringStream func_impls_;
    StringStream func_decls_;
    StringStream type_decls_;
    StringStream vars_decls_;
    /// Tracks defs that have been emitted as local variables of the current function
    DefSet func_defs_;

    std::ostringstream macro_xilinx_;
    std::ostringstream macro_intel_;

    ContinuationMap<FuncMode> builtin_funcs_; // OpenCL builtin functions
};

static inline const std::string lang_as_string(Lang lang) {
    switch (lang) {
        default:     THORIN_UNREACHABLE;
        case Lang::C99:    return "C99";
        case Lang::HLS:    return "HLS";
        case Lang::CUDA:   return "CUDA";
        case Lang::OpenCL: return "OpenCL";
    }
}

// TODO I think we should have a full-blown channel type
inline bool is_channel_type(const StructType* struct_type) {
    return struct_type->name().str().find("channel") != std::string::npos;
}

/// Returns true when the def carries concrete data in the final generated code
inline bool is_concrete(const Def* def) { return !is_mem(def) && !is_unit(def);}
inline bool is_concrete_type(const Type* t) { return !t->isa<MemType>() && t != t->world().unit_type(); }
inline bool has_concrete_params(Continuation* cont) {
    return std::any_of(cont->params().begin(), cont->params().end(), [](const Param* param) { return is_concrete(param); });
}

bool CCodeGen::get_interface(HlsInterface &interface, HlsInterface &gmem) {
    auto fpga_env = flags_;
    if (!fpga_env.empty()) {
        std::string fpga_env_str = fpga_env;
        for (auto& ch : fpga_env_str)
            ch = std::toupper(ch, std::locale());
        std::istringstream fpga_env_stream(fpga_env_str);
        std::string token;
        gmem = HlsInterface::None;
        bool set_interface = false;

        while (std::getline(fpga_env_stream, token, ',')) {
            if (token.compare("GMEM_OPT") == 0) {
                gmem = HlsInterface::GMEM_OPT;
                continue;
            } else if (token.compare("SOC") == 0 ) {
                interface = HlsInterface::SOC;
                set_interface = true;
                continue;
            } else if (token.compare("HPC") == 0 ) {
                interface = HlsInterface::HPC;
                set_interface = true;
                continue;
            } else if (token.compare("HPC_STREAM") == 0 ) {
                interface = HlsInterface::HPC_STREAM;
                set_interface = true;
                continue;
            } else {
                continue;
            }
        }
        return (set_interface ? true : false);
    }
    return false;
}

inline const char* stddef_primtype_name(PrimTypeTag tag) {
    switch (tag) {
        case PrimType_ps8:  case PrimType_qs8:  return "int8_t";
        case PrimType_pu8:  case PrimType_qu8:  return "uint8_t";
        case PrimType_ps16: case PrimType_qs16: return "int16_t";
        case PrimType_pu16: case PrimType_qu16: return "uint16_t";
        case PrimType_ps32: case PrimType_qs32: return "int32_t";
        case PrimType_pu32: case PrimType_qu32: return "uint32_t";
        case PrimType_ps64: case PrimType_qs64: return "int64_t";
        case PrimType_pu64: case PrimType_qu64: return "uint64_t";
        case PrimType_pf16: case PrimType_qf16: return "half";
        default: THORIN_UNREACHABLE;
    }
}

inline const char* cuda_scalar_primtype(PrimTypeTag tag) {
    switch (tag) {
        case PrimType_ps8:  case PrimType_qs8:  return "char";
        case PrimType_pu8:  case PrimType_qu8:  return "unsigned char";
        case PrimType_ps16: case PrimType_qs16: return "short";
        case PrimType_pu16: case PrimType_qu16: return "unsigned short";
        case PrimType_ps32: case PrimType_qs32: return "int";
        case PrimType_pu32: case PrimType_qu32: return "unsigned int";
        case PrimType_ps64: case PrimType_qs64: return "long long";
        case PrimType_pu64: case PrimType_qu64: return "unsigned long long";
        case PrimType_pf16: case PrimType_qf16: return "f16"; // typedef'd with macro magic
        default: THORIN_UNREACHABLE;
    }
}

/// OpenCL uses these for scalar and vectors.
/// CUDA actually uses the same prefixes for its vectors
/// See
/// https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/vectorDataTypes.html
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
inline const char* opencl_cuda_vectorbase(PrimTypeTag tag) {
    switch (tag) {
        case PrimType_ps8:  case PrimType_qs8:  return "char";
        case PrimType_pu8:  case PrimType_qu8:  return "uchar";
        case PrimType_ps16: case PrimType_qs16: return "short";
        case PrimType_pu16: case PrimType_qu16: return "ushort";
        case PrimType_ps32: case PrimType_qs32: return "int";
        case PrimType_pu32: case PrimType_qu32: return "uint";
        case PrimType_ps64: case PrimType_qs64: return "long";
        case PrimType_pu64: case PrimType_qu64: return "ulong";
        case PrimType_pf16: case PrimType_qf16: return "half"; // NB: cuda has no vectors of half.
        default: THORIN_UNREACHABLE;
    }
}

void CCodeGen::convert_primtype(StringStream& s, PrimTypeTag tag, int len) {
    assert(len > 0);

    // Enable special code paths for f16 and f64
    switch (tag) {
        case PrimType_pf16: case PrimType_qf16: use_fp_16_ = true; break;
        case PrimType_pf64: case PrimType_qf64: use_fp_64_ = true; break;
        default: break;
    }

    // 'bool', 'float' and 'double' are identical everywhere
    switch (tag) {
        case PrimType_bool: s << "bool"; break;
        case PrimType_pf32: case PrimType_qf32: s << "float"; break;
        case PrimType_pf64: case PrimType_qf64: s << "double"; break;
        default: {
            if (lang_ == Lang::CUDA && len == 1)
                s << cuda_scalar_primtype(tag);
            else if (lang_ == Lang::CUDA || lang_ == Lang::OpenCL)
                s << opencl_cuda_vectorbase(tag);
            else
                s << stddef_primtype_name(tag);
            break;
        }
    }

    // length suffixes
    if (len == 1)
        return;
    switch (lang_) {
        case Lang::CUDA:
        case Lang::OpenCL:
            s << len;
            break;
        case Lang::HLS:
        case Lang::C99:
            s.fmt(" __attribute__ ((ext_vector_size ({})))", len);
            break;
    }
}

/*
 * convert
 */

std::string CCodeGen::convert(const Type* type) {
    if (auto res = types_.lookup(type)) return *res;

    StringStream s;
    std::string name;

    if (type == world().unit_type() || type->isa<MemType>() || type->isa<FrameType>())
        s << "void";
    else if (auto primtype = type->isa<PrimType>()) {
        convert_primtype(s, primtype->primtype_tag(), vector_length(primtype));
    } else if (auto array = type->isa<IndefiniteArrayType>()) {
        return types_[type] = convert(array->elem_type()); // IndefiniteArrayType always occurs within a pointer
    } else if (auto closure_t = type->isa<ClosureType>()) {
        types_[closure_t] = name = closure_name(closure_t);
        auto as_fnt = world().fn_type(closure_t->types());
        s.fmt("typedef struct {{\t\n");
        s.fmt("{} f;\n", convert(as_fnt));
        s.fmt("{} data;", convert(Closure::environment_type(world())));
        s.fmt("\b\n}} {};\n", name);
    } else if (auto pi = type->isa<FnType>()) {
        assert(lang_ == Lang::C99 || lang_ == Lang::CUDA && "Only C and CUDA support function pointers");
        name = fn_name(pi);
        std::vector<std::string> dom;
        for (auto p : pi->domain()) {
            if (!is_concrete_type(p))
                continue;
            dom.push_back(convert(p));
        }
        s.fmt("typedef {} (*{})({, });\n", convert(mangle_return_type(pi->return_param_type())), name, dom);
    } else if (auto ptr = type->isa<PtrType>()) {
        // CUDA supports generic pointers, so there is no need to annotate them (moreover, annotating them triggers a bug in NVCC 11)
        s.fmt("{}{}*", lang_ != Lang::CUDA ? addr_space_prefix(ptr->addr_space()) : "", convert(ptr->pointee()));
    } else if (auto array = type->isa<DefiniteArrayType>()) {
        name = array_name(array);
        auto elem_type = convert(array->elem_type());
        s.fmt("typedef struct {{\t\n{} e[{}];\b\n}} {};\n", elem_type, array->dim(), name);
    } else if (auto tuple = type->isa<TupleType>()) {
        name = tuple_name(tuple);
        s.fmt("typedef struct {{\t\n");
        s.rangei(tuple->ops(), "\n", [&](size_t i) { s.fmt("{} e{};", convert(tuple->types()[i]), i); });
        s.fmt("\b\n}} {};\n", name);
    } else if (auto variant = type->isa<VariantType>()) {
        types_[variant] = name = variant->name().str();
        auto tag_type =
            variant->num_ops() < (UINT64_C(1) <<  8u) ? world().type_qu8()  :
            variant->num_ops() < (UINT64_C(1) << 16u) ? world().type_qu16() :
            variant->num_ops() < (UINT64_C(1) << 32u) ? world().type_qu32() :
                                                        world().type_qu64();
        s.fmt("typedef struct {{\t\n");

        // This is required because we have zero-sized types but C/C++ do not
        if (variant->has_payload()) {
            s.fmt("union {{\t\n");
            s.rangei(variant->ops(), "\n", [&] (size_t i) {
                if (is_type_unit(variant->types()[i]))
                    s << "// ";
                s.fmt("{} {};", convert(variant->types()[i]), variant->op_name(i));
            });
            s.fmt("\b\n}} data;\n");
        }

        s.fmt("{} tag;", convert(tag_type));
        s.fmt("\b\n}} {};\n", name);
    } else if (auto struct_type = type->isa<StructType>()) {
        types_[struct_type] = name = struct_type->name().str();
        if ((lang_ == Lang::OpenCL || lang_ == Lang::HLS) && is_channel_type(struct_type))
            use_channels_ = true;
        if (lang_ == Lang::OpenCL && use_channels_) {
            s.fmt("typedef {} {}_{};\n", convert(struct_type->types()[0]), name, struct_type->gid());
            name = (struct_type->name().str() + "_" + std::to_string(type->gid()));
        } else if (is_channel_type(struct_type) && lang_ == Lang::HLS) {
            s.fmt("typedef {} {}_{};\n", convert(struct_type->types()[0]), name, struct_type->gid());
            name = ("hls::stream<" + name + "_" + std::to_string(type->gid()) + ">");
        } else {
            s.fmt("typedef struct {{\t\n");
            s.rangei(struct_type->ops(), "\n", [&] (size_t i) { s.fmt("{} {};", convert(struct_type->types()[i]), struct_type->op_name(i)); });
            s.fmt("\b\n}} {};\n", name);
        }
    } else {
        THORIN_UNREACHABLE;
    }

    if (name.empty()) {
        return types_[type] = s.str();
    } else {
        assert(!s.str().empty());
        type_decls_ << s.str();
        return types_[type] = name;
    }
}

std::string CCodeGen::addr_space_prefix(AddrSpace addr_space) {
    if (lang_ == Lang::OpenCL) {
        switch (addr_space) {
            default:
            case AddrSpace::Generic: return "";
            case AddrSpace::Global:  return "__global ";
            case AddrSpace::Shared:  return "__local ";
        }
    } else if (lang_ == Lang::CUDA) {
        switch (addr_space) {
            default:
            case AddrSpace::Global:
            case AddrSpace::Generic: return "";
            case AddrSpace::Shared:  return "__shared__ ";
        }
    } else {
        assert(lang_ != Lang::C99 || addr_space == AddrSpace::Generic);
        return "";
    }
}

std::string CCodeGen::constructor_prefix(const Type* type) {
    auto type_name = convert(type);
    if (lang_ == Lang::C99 || lang_ == Lang::OpenCL)
        return "(" + type_name + ")";
    return type_name;
}

std::string CCodeGen::device_prefix() {
    switch (lang_) {
        default:           return "";
        case Lang::CUDA:   return "__device__ ";
        case Lang::OpenCL:
            if (use_channels_)
                return "PIPE ";
            else
                return "__constant ";
    }
}

/*
 * emit
 */

HlsInterface interface, gmem_config;
bool interface_status, hls_top_scope = false;

void CCodeGen::emit_module() {
    Continuation* hls_top = nullptr;
    interface_status = get_interface(interface, gmem_config);

    forest_.for_each([&] (const Scope& scope) {
        if (scope.entry()->name() == "hls_top")
            hls_top = scope.entry();
        else if (scope.entry()->cc() != CC::Thorin && scope.entry()->is_returning())
            queue_scope(scope.entry());
    });
    if (hls_top) {
        hls_top_scope = true;
        queue_scope(hls_top);
    }

    emit_scopes(forest_);

    if (lang_ == Lang::OpenCL) {
        if (use_channels_) {
            std::string write_channel_params = "(channel, val) ";
            std::string read_channel_params = "(val, channel) ";

            macro_xilinx_ << " #define PIPE pipe\n";
            macro_intel_  << " #pragma OPENCL EXTENSION cl_intel_channels : enable\n"
                          << " #define PIPE channel\n";
            for (auto map : builtin_funcs_) {
                if (map.first->is_channel()) {
                    if (map.second == FuncMode::Write) {
                        macro_xilinx_ << " #define " << map.first->name() << write_channel_params << "write_pipe_block(channel, &val)\n";
                        macro_intel_  << " #define " << map.first->name() << write_channel_params << "write_channel_intel(channel, val)\n";
                    } else if (map.second == FuncMode::Read) {
                        macro_xilinx_ << " #define " << map.first->name() << read_channel_params << "read_pipe_block(channel, &val)\n";
                        macro_intel_  << " #define " << map.first->name() << read_channel_params << "val = read_channel_intel(channel)\n";
                    }
                }
            }
            stream_ << "#if defined(__xilinx__)\n";
            stream_ << macro_xilinx_.str();

            stream_ << "#elif defined(INTELFPGA_CL)\n";
            stream_ << macro_intel_.str();

            stream_ << "#else\n"
                       " #define PIPE pipe\n";
            stream_ << "#endif\n";
        }

        if (use_fp_16_)
            stream_ << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        if (use_fp_64_)
            stream_ << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    }

    stream_.endl();

    if (lang_ == Lang::C99) {
        stream_.fmt(    "#include <stdbool.h>\n"    // for the 'bool' type
                        "#include <stdint.h>\n"     // for the fixed-width integer types
                        "#include <stddef.h>\n");   // for size_t
        if (use_align_of_)
            stream_.fmt("#include <stdalign.h>\n"); // for 'alignof'
        if (use_memcpy_)
            stream_.fmt("#include <string.h>\n");   // for 'memcpy'
        if (use_malloc_)
            stream_.fmt("#include <stdlib.h>\n");   // for 'malloc'
        if (use_longjmp_)
            stream_.fmt("#include <setjmp.h>\n");   // for 'longjmp'
        if (use_math_)
            stream_.fmt("#include <math.h>\n");     // for 'cos'/'sin'/...
    }

    if (lang_ == Lang::HLS) {
        stream_.fmt("#include <hls_stream.h>\n"
                    "#include <hls_math.h>\n");
        if (use_fp_16_)
            stream_.fmt("#include <hls_half.h>\n");
    }

    if (lang_ == Lang::CUDA) {
        if (use_fp_16_) {
            stream_.fmt("#include <cuda_fp16.h>\n\n");
            stream_.fmt("#if __CUDACC_VER_MAJOR__ <= 8\n"
                        "typedef               half f16;\n"
                        "#else\n"
                        "typedef         __half_raw f16;\n"
                        "#endif\n");
        }
    }

    if (lang_ == Lang::CUDA || lang_ == Lang::HLS) {
        stream_.fmt("extern \"C\" {{\n");
    }

    stream_ << type_decls_.str() << "\n";

    // For Xilinx hardware, we have to ifdef the function declarations away
    // In CL mode we don't want them at all, for HLS we only want them when doing simulations and not for synthesis
    if (lang_ == Lang::OpenCL)
        stream_ << "#ifndef __xilinx__\n";
    else if (lang_ == Lang::HLS)
        stream_ << "#ifndef __SYNTHESIS__\n";

    stream_ << func_decls_.str();

    if (lang_ == Lang::OpenCL)
        stream_ << "#endif /* __xilinx__ */\n";
    else if (lang_ == Lang::HLS)
        stream_ << "#endif /* __SYNTHESIS__ */\n";

    stream_ << vars_decls_.str();

    if (lang_ == Lang::CUDA) {
        stream_.endl();
        for (auto x : std::array {'x', 'y', 'z'}) {
            stream_.fmt("__device__ inline int threadIdx_{}() {{ return threadIdx.{}; }}\n", x, x);
            stream_.fmt("__device__ inline int blockIdx_{}() {{ return blockIdx.{}; }}\n", x, x);
            stream_.fmt("__device__ inline int blockDim_{}() {{ return blockDim.{}; }}\n", x, x);
            stream_.fmt("__device__ inline int gridDim_{}() {{ return gridDim.{}; }}\n", x, x);
        }
    }

    stream_.endl() << func_impls_.str();

    if (lang_ == Lang::CUDA || lang_ == Lang::HLS)
        stream_.fmt("}} /* extern \"C\" */\n");
}

static inline bool is_passed_via_buffer(const Param* param) {
    return param->type()->isa<DefiniteArrayType>()
        || param->type()->isa<StructType>()
        || param->type()->isa<TupleType>();
}

std::vector<int> get_concrete_param_indices(ArrayRef<const Type*> param_types) {
    int i = 0;
    std::vector<int> r;
    for (auto pt : param_types) {
        if (is_concrete_type(pt))
            r.push_back(i);
        i++;
    }
    return r;
}

std::vector<std::string> unpack_args(ArrayRef<const Type*> param_types, std::string packed) {
    std::vector<std::string> ret_args;
    auto ret_val_swizzle = get_concrete_param_indices(param_types);
    if (ret_val_swizzle.size() == 1)
        ret_args = { packed };
    else {
        for (size_t i = 0; i < ret_val_swizzle.size(); i++) {
            ret_args.push_back(packed + std::string(".e") + std::to_string(i));
        }
    }
    return ret_args;
}

const Type* CCodeGen::mangle_return_type(const ReturnType* return_type) {
    // treat non-returning calls as if they return nothing, for now
    if (!return_type)
        return world().unit_type();
    return return_type->mangle_for_codegen();
}

static inline const Type* pointee_or_elem_type(const PtrType* ptr_type) {
    auto elem_type = ptr_type->as<PtrType>()->pointee();
    if (auto array_type = elem_type->isa<ArrayType>())
        elem_type = array_type->elem_type();
    return elem_type;
}

std::string CCodeGen::prepare(const Scope& scope) {
    auto cont = scope.entry();

    StringStream hls_pragmas_;

    for (auto param : cont->params()) {
        defs_[param] = param->unique_name();
        if (lang_ == Lang::HLS && cont->is_exported() && param->type()->isa<PtrType>()) {
            auto elem_type = pointee_or_elem_type(param->type()->as<PtrType>());
            if ((elem_type->isa<StructType>() || elem_type->isa<DefiniteArrayType>()) && !hls_top_scope)
                hls_pragmas_.fmt("#pragma HLS data_pack variable={} struct_level\n", param->unique_name());
        }
    }

    func_impls_.fmt("{} {{", emit_fun_head(cont));
    func_impls_.fmt("\t\n");

    if (lang_ == Lang::HLS && cont->is_exported()) {
        if (cont->name() == "hls_top") {
            if (interface_status) {
                if (cont->num_params() > 2) {
                    size_t hls_gmem_index = 0;
                    for (auto param : cont->params()) {
                        if (!is_concrete(param))
                            continue;
                        if (param->type()->isa<PtrType>() && param->type()->as<PtrType>()->pointee()->isa<ArrayType>()) {
                            if (interface == HlsInterface::SOC)
                                func_impls_ << "#pragma HLS INTERFACE axis port = " << param->unique_name() << "\n";
                            else if (interface == HlsInterface::HPC) {
                                if (gmem_config == HlsInterface::GMEM_OPT)
                                    hls_gmem_index++;
                                func_impls_ << "#pragma HLS INTERFACE m_axi" << std::string(5, ' ') << "port = " << param->unique_name()
                                            << " bundle = gmem" << hls_gmem_index << std::string(2, ' ') << "offset = slave" << "\n";
                                func_impls_ << "#pragma HLS INTERFACE s_axilite"<<" port = " << param->unique_name() << "\n";
                            } else if (interface == HlsInterface::HPC_STREAM) {
                                func_impls_ << "#pragma HLS INTERFACE axis port = " << param->unique_name() << "\n";
                            }
                        } else {
                            if (interface == HlsInterface::SOC)
                                func_impls_ << "#pragma HLS INTERFACE s_axilite port = " << param->unique_name() << "\n";
                            else if (interface == HlsInterface::HPC)
                                func_impls_ << "#pragma HLS INTERFACE s_axilite port = " << param->unique_name() << " bundle = control" << "\n";
                        }

                        func_impls_ << "#pragma HLS STABLE variable = " << param->unique_name() << "\n";
                    }
                    if (interface == HlsInterface::SOC || interface == HlsInterface::HPC_STREAM)
                        func_impls_ << "#pragma HLS INTERFACE ap_ctrl_none port = return\n";
                    else if (interface == HlsInterface::HPC)
                        func_impls_ << "#pragma HLS INTERFACE ap_ctrl_chain port = return\n";
                }
            } else {
                interface = HlsInterface::None;
                world().WLOG("HLS accelerator generated with no interface");
            }
            func_impls_ << "#pragma HLS top name = hls_top\n";
            if (use_channels_)
                func_impls_ << "#pragma HLS DATAFLOW\n";
        } else if (use_channels_) {
            func_impls_ << "#pragma HLS INLINE off\n";
        }
    }

    func_impls_ << hls_pragmas_.str();

    // Load OpenCL structs from buffers
    // TODO: See above
    for (auto param : cont->params()) {
        if (!is_concrete(param))
            continue;
        if (lang_ == Lang::OpenCL && cont->is_exported() && is_passed_via_buffer(param))
            func_impls_.fmt("{} {} = *{}_;\n", convert(param->type()), param->unique_name(), param->unique_name());
    }
    return {};
}

void CCodeGen::prepare(Continuation* cont, const std::string&) {
    auto& bb = cont2bb_[cont];
    bb.cont = cont;
    bb.head.indent(2);
    bb.body.indent(2);
    bb.tail.indent(2);
    // The parameters of the entry continuation have already been emitted.
    if (cont != entry_) {
        for (auto param : cont->params()) {
            if (!is_concrete(param)) {
                defs_[param] = {};
                continue;
            }
            // Note: Having two versions of a phi is necessary, since we may have a loop which
            // invokes itself like this: `loop(param1 + 1, param1 + 2)`. In this case, the C
            // code generator will emit two assignments to the phis nodes, but the second one
            // depends on the current value of the phi node.
            // Lookup "lost copy problem" and "swap problem" in literature for SSA destruction for more information.
            func_impls_.fmt("{}   {};\n", convert(param->type()), param->unique_name());
            func_impls_.fmt("{} p_{};\n", convert(param->type()), param->unique_name());
            bb.head.fmt("{} = p_{};\n", param->unique_name(), param->unique_name());
            defs_[param] = param->unique_name();
        }
    }
}

static inline std::string make_identifier(const std::string& str) {
    auto copy = str;
    // Transform non-alphanumeric characters
    std::transform(copy.begin(), copy.end(), copy.begin(), [] (auto c) {
        if (c == '*') return 'p';
        if (!std::isalnum(c)) return '_';
        return c;
    });
    // First character must be a letter or '_'
    if (!std::isalpha(copy[0]) && copy[0] != '_')
        copy.insert(copy.begin(), '_');
    return copy;
}

static inline std::string label_name(const Def* def) {
    return make_identifier(def->as_nom<Continuation>()->unique_name());
}

void CCodeGen::finalize(const Scope&) {
    for (auto& def : func_defs_) {
        assert(defs_.contains(def) && "sanity check, should have been emitted if it's here");
        defs_.erase(def);
    }
    func_defs_.clear();
    func_impls_.fmt("}}\n\n");
}

void CCodeGen::finalize(Continuation* cont) {
    auto&& bb = cont2bb_[cont];
    if (cont != entry_)
        func_impls_.fmt("{}: \t", label_name(cont));
    func_impls_.fmt("{{\t\n{}{}{}\b\n}}\b\n", bb.head.str(), bb.body.str(), bb.tail.str());
}

const Param* CCodeGen::get_channel_read_output(Continuation* cont) {
    size_t num_params = cont->num_params();
    size_t n = 0;
    Array<const Param*> values(num_params);
    for (auto param : cont->params()) {
        if (!is_mem(param) && !is_unit(param)) {
            values[n] = param;
            n++;
        }
    }
    return n == 1 ? values[0] : nullptr;
}

void CCodeGen::emit_jump(BB& bb, const Def* callee, ArrayRef<std::string> args) {
    if (auto ret_pt = callee->isa<ReturnPoint>())
        callee = ret_pt->continuation();

    if (auto ret_type = callee->type()->isa<ReturnType>()) { // return
        assert((callee == entry_->ret_param()) && "In the C backend, the only callable return typed values should be fn returns");
        std::string packed_return;
        switch (args.size()) {
            case 0: packed_return = lang_ == Lang::HLS ? "void()" : ""; break;
            case 1: packed_return = args[0]; break;
            default:
                auto tuple = convert(mangle_return_type(ret_type));
                bb.tail.fmt("{} fn_ret_val;\n", tuple);
                for (size_t i = 0, e = args.size(); i != e; ++i)
                    bb.tail.fmt("fn_ret_val.e{} = {};\n", i, args[i]);
                packed_return = "fn_ret_val";
                break;
        }

        // local return
        if (packed_return.empty())
            bb.tail.fmt("return;");
        else
            bb.tail.fmt("return {};", packed_return);
        return;
    }

    if (auto cont = callee->isa_nom<Continuation>()) {
        auto& scope = forest_.get_scope(entry_);
        // Use goto syntax when calling within the local scope
        // (and it's not recursion)
        if (scope.contains(cont) && cont != entry_) {
            //assert(cont->num_params() == args.size());
            for (size_t i = 0; i != args.size(); ++i) {
                if (auto arg = args[i]; !arg.empty())
                    bb.tail.fmt("p_{} = {};\n", cont->param(i)->unique_name(), arg);
            }
            bb.tail.fmt("goto {};", label_name(cont));
            return;
        }
    }

    auto ecallee = emit(callee);
    if (auto closure_t = callee->type()->isa<ClosureType>()) {
        auto appended = concat(args, ecallee);
        auto as_fnt = world().fn_type(concat(closure_t->types(), closure_t->as<Type>()));
        bb.tail.fmt("(({}) {}.f)({, });", convert(as_fnt), ecallee, appended);
    } else
        bb.tail.fmt("{}({, });", ecallee, args);
}

void CCodeGen::emit_epilogue(Continuation* cont) {
    auto&& bb = cont2bb_[cont];
    assert(cont->has_body());
    auto body = cont->body();
    if (body->num_args() > 0) {
        emit_debug_info(bb.tail, body->arg(0));
    }

    if ((lang_ == Lang::OpenCL || (lang_ == Lang::HLS && hls_top_scope)) && (cont->is_exported()))
        emit_fun_decl(cont);

    if (body->callee() == world().branch()) {
        emit_unsafe(body->arg(0));
        auto c = emit(body->arg(1));
        auto t = label_name(body->arg(2));
        auto f = label_name(body->arg(3));
        bb.tail.fmt("if ({}) goto {}; else goto {};", c, t, f);
    } else if (auto callee = body->callee()->isa_nom<Continuation>(); callee && callee->intrinsic() == Intrinsic::Match) {
        emit_unsafe(body->arg(0));

        bb.tail.fmt("switch ({}) {{\t\n", emit(body->arg(1)));

        for (size_t i = 3; i < body->num_args(); i++) {
            auto arg = body->arg(i)->as<Tuple>();
            bb.tail.fmt("case {}: goto {};\n", emit_constant(arg->op(0)), label_name(arg->op(1)));
        }

        bb.tail.fmt("default: goto {};", label_name(body->arg(2)));
        bb.tail.fmt("\b\n}}");
    } else if (body->callee()->isa<Bottom>()) {
        bb.tail.fmt("return;  // bottom: unreachable");
    } else if (auto callee = body->callee()->isa_nom<Continuation>(); callee && callee->is_intrinsic()) {
        if (callee->intrinsic() == Intrinsic::Reserve) {
            emit_unsafe(body->arg(0));
            if (!body->arg(1)->isa<PrimLit>())
                world().edef(body->arg(1), "reserve_shared: couldn't extract memory size");

            auto ret_cont = body->arg(2)->as_nom<Continuation>();
            auto elem_type = ret_cont->param(1)->type()->as<PtrType>()->pointee()->as<ArrayType>()->elem_type();
            func_impls_.fmt("{}{} {}_reserved[{}];\n",
                addr_space_prefix(AddrSpace::Shared), convert(elem_type),
                cont->unique_name(), emit_constant(body->arg(1)));
            if (lang_ == Lang::HLS && !hls_top_scope) {
                func_impls_.fmt("#pragma HLS dependence variable={}_reserved inter false\n", cont->unique_name());
                func_impls_.fmt("#pragma HLS data_pack  variable={}_reserved\n", cont->unique_name());
                func_impls_<< "#if defined( __VITIS_HLS__ )\n   __attribute__((packed))\n  #endif\n";
            }
            bb.tail.fmt("p_{} = {}_reserved;\n", ret_cont->param(1)->unique_name(), cont->unique_name());
            bb.tail.fmt("goto {};", label_name(ret_cont));
        } else if (callee->intrinsic() == Intrinsic::Pipeline) {
            assert((lang_ == Lang::OpenCL || lang_ == Lang::HLS) && "pipelining not supported on this backend");

            emit_unsafe(body->arg(0));
            std::string interval;
            if (body->arg(1)->as<PrimLit>()->value().get_s32() != 0)
                interval = emit_constant(body->arg(1));


            auto begin = emit(body->arg(2));
            auto end   = emit(body->arg(3));
            // HLS/OpenCL Pipeline loop-index
            bb.tail.fmt("int i{};\n", body->callee()->gid());
            if (lang_ == Lang::OpenCL) {
                bb.tail << guarded_statement(cl_dialect_guard(CLDialect::INTEL), [&](Stream& s){
                    s.fmt("#pragma ii {}\n", !interval.empty() ? interval : "1");
                });
                bb.tail << guarded_statement(cl_dialect_guard(CLDialect::XILINX), [&](Stream& s){
                    s.fmt("__attribute__((xcl_pipeline_loop({})))\n", !interval.empty() ? interval : "1");
                });
            }
            // The next instruction pipeline pragmas/attributes need to see is just a loop-head.
            // No any other instructions should come in between.
            bb.tail.fmt("for (i{} = {}; i{} < {}; i{}++) {{\t\n",
                callee->gid(), begin, callee->gid(), end, callee->gid());
            if (lang_ == Lang::HLS) {
                bb.tail << "#pragma HLS PIPELINE";
                if (!interval.empty())
                    bb.tail.fmt(" II={}", interval);
                bb.tail.fmt("\n");
            }

            auto pbody = body->arg(4)->as_nom<Continuation>();
            bb.tail.fmt("p_{} = i{};\n", pbody->param(1)->unique_name(), callee->gid());
            bb.tail.fmt("goto {};\n", label_name(pbody));

            // Emit a label that can be used by the "pipeline_continue()" intrinsic.
            bb.tail.fmt("\b\n{}: continue;\n}}\n", label_name(body->arg(6)));
            bb.tail.fmt("goto {};", label_name(body->arg(5)));
        } else if (callee->intrinsic() == Intrinsic::PipelineContinue) {
            emit_unsafe(body->arg(0));
            bb.tail.fmt("goto {};", label_name(callee));
        } else {
            THORIN_UNREACHABLE;
        }
    } else { // function/closure call
        auto callee_type = body->callee()->type()->as<FnType>();
        const Def* ret_arg = body->ret_arg();

        std::vector<std::string> args;
        for (auto arg : body->args()) {
            if (arg == ret_arg) continue;
            if (auto emitted_arg = emit_unsafe(arg); !emitted_arg.empty())
                args.emplace_back(emitted_arg);
        }

        // Do not store the result of `void` calls
        auto returned_value_type = mangle_return_type(callee_type->return_param_type());
        if (!is_type_unit(returned_value_type))
            bb.tail.fmt("{} ret_val = ", convert(returned_value_type));

        // TODO: tailcalls
        emit_jump(bb, body->callee(), args);

        // forward the returned values to the return param (typically a BB)
        if (ret_arg) {
            bb.tail.fmt("\n");
            emit_jump(bb, ret_arg, unpack_args(ret_arg->type()->as<ReturnType>()->domain(), "ret_val"));
        }
    }
}

static inline bool is_definite_to_indefinite_array_cast(const PtrType* from, const PtrType* to) {
    return
        from->pointee()->isa<DefiniteArrayType>() &&
        to->pointee()->isa<IndefiniteArrayType>() &&
        from->pointee()->as<ArrayType>()->elem_type() ==
        to->pointee()->as<ArrayType>()->elem_type();
}

std::string CCodeGen::emit_bottom(const Type* type) {
    if (auto definite_array = type->isa<DefiniteArrayType>()) {
        StringStream s;
        s.fmt("{} ", constructor_prefix(type));
        s << "{ ";
        auto op = emit_bottom(definite_array->elem_type());
        for (size_t i = 0, n = definite_array->dim(); i < n; ++i) {
            s << op;
            if (i != n - 1)
                s << ", ";
        }
        s << " }";
        return s.str();
    } else if (type->isa<StructType>() || type->isa<TupleType>()) {
        StringStream s;
        s.fmt("{} ", constructor_prefix(type));
        s << "{ ";
        s.range(type->ops(), ", ", [&] (const Def* op) { s << emit_bottom(op->as<Type>()); });
        s << " }";
        return s.str();
    } else if (auto variant_type = type->isa<VariantType>()) {
        if (variant_type->has_payload()) {
            auto non_unit = *std::find_if(variant_type->types().begin(), variant_type->types().end(),
                [] (const Type* op) { return !is_type_unit(op); });
            return constructor_prefix(type) + " { { " + emit_bottom(non_unit) + " }, 0 }";
        }
        return constructor_prefix(type) + "{ 0 }";
    } else if (type->isa<PtrType>() || type->isa<PrimType>()) {
        return "0";
    } else {
        THORIN_UNREACHABLE;
    }
}

void CCodeGen::emit_access(Stream& s, const Type* agg_type, const Def* index, const std::string_view& prefix) {
    if (agg_type->isa<DefiniteArrayType>()) {
        s.fmt("{}e[{}]", prefix, emit(index));
    } else if (agg_type->isa<IndefiniteArrayType>()) {
        s.fmt("[{}]", emit(index));
    } else if (agg_type->isa<TupleType>()) {
        s.fmt("{}e{}", prefix, primlit_value<size_t>(index));
    } else if (agg_type->isa<StructType>()) {
        s.fmt("{}{}", prefix, agg_type->as<StructType>()->op_name(primlit_value<size_t>(index)));
    } else if (agg_type->isa<VectorType>()) {
        std::ostringstream os;
        // OpenCL indices must be in hex format
        if (index->isa<PrimLit>())
            os << std::hex << primlit_value<size_t>(index);
        else
            world().edef(index, "only constants are supported as vector component indices");
        s.fmt("{}s{}", prefix, os.str());
    } else if (agg_type->isa<ClosureType>()) {
        switch (primlit_value<size_t>(index)) {
            case 0: s.fmt("{}f", prefix); break;
            case 1: s.fmt("{}data", prefix); break;
            default: assert(false);
        }
    } else {
        THORIN_UNREACHABLE;
    }
}

std::string CCodeGen::emit_bb(BB& bb, const Def* def) {
    return emit_def(&bb, def);
}

std::string CCodeGen::emit_constant(const Def* def) {
    return emit_def(nullptr, def);
}

/// If bb is nullptr, then we are emitting a constant, otherwise we emit the def as a local variable
std::string CCodeGen::emit_def(BB* bb, const Def* def) {
    auto sp = std::make_unique<StringStream>();
    auto& s = *sp;
    auto name = def->unique_name();
    const Type* emitted_type = def->type();

    if (is_unit(def)) return "";
    else if (auto bin = def->isa<BinOp>()) {
        const char* op = "";
        if (auto cmp = bin->isa<Cmp>()) {
            switch (cmp->cmp_tag()) {
                case Cmp_eq: op = "=="; break;
                case Cmp_ne: op = "!="; break;
                case Cmp_gt: op = ">";  break;
                case Cmp_ge: op = ">="; break;
                case Cmp_lt: op = "<";  break;
                case Cmp_le: op = "<="; break;
            }
        } else if (auto arithop = bin->isa<ArithOp>()) {
            switch (arithop->arithop_tag()) {
                case ArithOp_add: op = "+";  break;
                case ArithOp_sub: op = "-";  break;
                case ArithOp_mul: op = "*";  break;
                case ArithOp_div: op = "/";  break;
                case ArithOp_rem: op = "%";  break;
                case ArithOp_and: op = "&";  break;
                case ArithOp_or:  op = "|";  break;
                case ArithOp_xor: op = "^";  break;
                case ArithOp_shl: op = "<<"; break;
                case ArithOp_shr: op = ">>"; break;
            }
        }
        s.fmt("({} {} {})", emit_unsafe(bin->lhs()), op, emit_unsafe(bin->rhs()));
    } else if (auto mathop = def->isa<MathOp>()) {
        use_math_ = true;
        auto make_key = [] (MathOpTag tag, unsigned bitwidth) { return (static_cast<unsigned>(tag) << 16) | bitwidth; };
        static const std::unordered_map<unsigned, std::string> function_names = {
#define MATH_FUNCTION(name) \
            { make_key(MathOp_##name, 32), #name "f" }, \
            { make_key(MathOp_##name, 64), #name },
            MATH_FUNCTION(fabs)
            MATH_FUNCTION(copysign)
            MATH_FUNCTION(round)
            MATH_FUNCTION(floor)
            MATH_FUNCTION(ceil)
            MATH_FUNCTION(fmin)
            MATH_FUNCTION(fmax)
            MATH_FUNCTION(cos)
            MATH_FUNCTION(sin)
            MATH_FUNCTION(tan)
            MATH_FUNCTION(acos)
            MATH_FUNCTION(asin)
            MATH_FUNCTION(atan)
            MATH_FUNCTION(atan2)
            MATH_FUNCTION(sqrt)
            MATH_FUNCTION(cbrt)
            MATH_FUNCTION(pow)
            MATH_FUNCTION(exp)
            MATH_FUNCTION(exp2)
            MATH_FUNCTION(log)
            MATH_FUNCTION(log2)
            MATH_FUNCTION(log10)
#undef MATH_FUNCTION
        };
        int bitwidth = num_bits(mathop->type()->primtype_tag());
        assert(function_names.count(make_key(mathop->mathop_tag(), bitwidth)) > 0);
        if (lang_ == Lang::OpenCL && bitwidth == 32)
            bitwidth = 64; // OpenCL uses overloading
        s.fmt("{}(", function_names.at(make_key(mathop->mathop_tag(), bitwidth)));
        s.range(mathop->ops(), ", ", [&](const Def* op) { s << emit_unsafe(op); });
        s.fmt(")");
    } else if (auto conv = def->isa<ConvOp>()) {
        auto s_type = conv->from()->type();
        auto d_type = conv->type();
        auto s_ptr = s_type->isa<PtrType>();
        auto d_ptr = d_type->isa<PtrType>();
        auto src = emit_unsafe(conv->from());

        auto s_t = convert(s_type);
        auto d_t = convert(d_type);

        if (s_ptr && d_ptr && is_definite_to_indefinite_array_cast(s_ptr, d_ptr)) {
            s.fmt("(({})->e)", src);
        } else if (s_ptr && d_ptr && s_ptr->addr_space() == d_ptr->addr_space()) {
            s.fmt("(({}) {})", d_t, src);
        } else if (s_ptr && d_ptr && s_ptr->addr_space() != d_ptr->addr_space() && lang_ == Lang::OpenCL) {
            s.fmt("(({}) ((size_t) {}))", d_t, src);
        } else if (conv->isa<Cast>()) {
            auto s_prim = s_type->isa<PrimType>();
            auto d_prim = d_type->isa<PrimType>();

            if (lang_ == Lang::CUDA && s_prim && (s_prim->primtype_tag() == PrimType_pf16 || s_prim->primtype_tag() == PrimType_qf16)) {
                s.fmt("__half2float({})", src);
            } else if (lang_ == Lang::CUDA && d_prim && (d_prim->primtype_tag() == PrimType_pf16 || d_prim->primtype_tag() == PrimType_qf16)) {
                s.fmt("__float2half({})", src);
            } else {
                s.fmt("(({}) {})", d_t, src);
            }
        } else if (conv->isa<Bitcast>()) {
            assert(bb && "re-interpreting types is only possible within a basic block");
            func_impls_.fmt("{} {};\n", convert(emitted_type), name);
            func_defs_.insert(def);
            if (lang_ == Lang::OpenCL) {
                // OpenCL explicitly supports type punning via unions (6.4.4.1)
                bb->body.fmt("union {{\t\n");
                bb->body.fmt("{} src;\n",   s_t);
                bb->body.fmt("{} dst;\b\n", d_t);
                bb->body.fmt("}} {}_u;\n", name);
                bb->body.fmt("{}_u.src = {};\n", name, src);
                bb->body.fmt("{} = {}_u.dst;\n", name, name);
            } else {
                bb->body.fmt("memcpy(&{}, &{}, sizeof({}));\n", name, src, name);
                use_memcpy_ = true;
            }
            return name;
        } else THORIN_UNREACHABLE;
    } else if (auto align_of = def->isa<AlignOf>()) {
        if (lang_ == Lang::C99 || lang_ == Lang::OpenCL) {
            world().wdef(def, "alignof() is only available in C11");
            use_align_of_ = true;
        }
        return "alignof(" + convert(align_of->of()) + ")";
    } else if (auto size_of = def->isa<SizeOf>()) {
        return "sizeof(" + convert(size_of->of()) + ")";
    } else if (def->isa<IndefiniteArray>()) {
        assert(bb && "we cannot emit indefinite arrays except within a basic block.");
        func_impls_.fmt("{} {}; // indefinite array: bottom\n", convert(def->type()), name);
        func_defs_.insert(def);
        return name;
    } else if (def->isa<Closure>()) {
        if (bb) {
            if (!func_defs_.contains(def))
                func_impls_.fmt("{} {};\n", convert(def->type()), name);
            func_defs_.insert(def);
            bb->body << name;
            emit_access(bb->body, def->type(), world().literal(thorin::pu64{ 0 }));
            auto fn_t = world().fn_type(def->type()->as<ClosureType>()->types());
            bb->body.fmt(" = ({}) {};\n", convert(fn_t), emit_unsafe(def->op(0)));

            auto env = def->op(1);
            if (!is_unit(env)) {
                auto eenv = emit_unsafe(env);
                bb->body << name;
                emit_access(bb->body, def->type(), world().literal(thorin::pu64{ 1 }));
                // thin environment
                bb->body.fmt(" = ({}) {};\n", convert(Closure::environment_type(world())), eenv);
            }
        } else {
            assert(false && "todo");
        }
        return name;
    } else if (auto captured_ret = def->isa<CaptureReturn>()) {
        assert(bb);
        world().WLOG("setjmp required at {}", captured_ret->loc());

        // the only valid return to capture is that of the parent cont
        use_longjmp_ = true;

        auto closure_t = captured_ret->type()->as<ClosureType>();
        auto ret_value_t = mangle_return_type(world().return_type(closure_t->types()));

        // setup a setjmp return point
        if (is_type_unit(ret_value_t))
            func_impls_.fmt("struct {{ jmp_buf opaque; }} {}_captured_return;\n", name);
        else
            func_impls_.fmt("struct {{ {} value; jmp_buf opaque; }} {}_captured_return;\n", convert(ret_value_t), name);
        bb->tail.fmt("if (setjmp({}_captured_return.opaque) != 0) {{\t\n", name);
        emit_jump(*bb, captured_ret->op(0), unpack_args(closure_t->domain(), name + "_captured_return.value"));
        bb->tail.fmt("\b\n}}\n");

        // create a closure that can call longjmp
        func_impls_.fmt("{} {};\n", convert(def->type()), name);
        bb->body << name;
        emit_access(bb->body, def->type(), world().literal(thorin::pu64{ 0 }));
        auto fn_t = world().fn_type(def->type()->as<ClosureType>()->types());
        bb->body.fmt(" = ({}) NULL /* TODO longjmp wrapper */;\n", convert(fn_t));
        bb->body << name;
        emit_access(bb->body, def->type(), world().literal(thorin::pu64{ 1 }));
        // thin environment
        bb->body.fmt(" = ({}) &{}_captured_return;\n", convert(Closure::environment_type(world())), name);
        return name;
    } else if (def->isa<Aggregate>()) {
        if (bb) {
            func_impls_.fmt("{} {};\n", convert(def->type()), name);
            func_defs_.insert(def);
            for (size_t i = 0, n = def->num_ops(); i < n; ++i) {
                auto op = emit_unsafe(def->op(i));
                bb->body << name;
                emit_access(bb->body, def->type(), world().literal(thorin::pu64{i}));
                bb->body.fmt(" = {};\n", op);
            }
            return name;
        } else {
            auto is_array = def->isa<DefiniteArray>();
            s.fmt("{} ", constructor_prefix(def->type()));
            s.fmt(is_array ? "{{ {{ " : "{{ ");
            s.range(def->ops(), ", ", [&] (const Def* op) { s << emit_constant(op); });
            s.fmt(is_array ? " }} }}" : " }}");
        }
    } else if (auto agg_op = def->isa<AggOp>()) {
        if (auto agg = emit_unsafe(agg_op->agg()); !agg.empty()) {
            emit_unsafe(agg_op->index());
            if (auto extract = def->isa<Extract>()) {
                if (is_mem(extract) || extract->type()->isa<FrameType>())
                    return "";

                s.fmt("({}", agg);
                if (!extract->agg()->isa<MemOp>() && !extract->agg()->isa<Assembly>())
                    emit_access(s, extract->agg()->type(), extract->index());
                s.fmt(")");
            } else if (auto insert = def->isa<Insert>()) {
                assert(bb && "cannot emit insert operations without a basic block");
                if (auto value = emit_unsafe(insert->value()); !value.empty()) {
                    func_impls_.fmt("{} {};\n", convert(insert->type()), name);
                    func_defs_.insert(def);
                    bb->body.fmt("{} = {};\n", name, agg);
                    bb->body.fmt("{}", name);
                    emit_access(bb->body, insert->agg()->type(), insert->index());
                    bb->body.fmt(" = {};\n", value);
                    return name;
                }
            }
        } else {
            return "";
        }
    } else if (auto primlit = def->isa<PrimLit>()) {
        switch (primlit->primtype_tag()) {
            case PrimType_bool:                     return primlit->bool_value() ? "true" : "false";
            case PrimType_ps8:  case PrimType_qs8:  return std::to_string(  (signed) primlit->ps8_value());
            case PrimType_pu8:  case PrimType_qu8:  return std::to_string((unsigned) primlit->pu8_value());
            case PrimType_ps16: case PrimType_qs16: return std::to_string(primlit->ps16_value());
            case PrimType_pu16: case PrimType_qu16: return std::to_string(primlit->pu16_value());
            case PrimType_ps32: case PrimType_qs32: return std::to_string(primlit->ps32_value());
            case PrimType_pu32: case PrimType_qu32: return std::to_string(primlit->pu32_value());
            case PrimType_ps64: case PrimType_qs64: return std::to_string(primlit->ps64_value()) + "LL";
            case PrimType_pu64: case PrimType_qu64: return std::to_string(primlit->pu64_value()) + "ULL";
            case PrimType_pf16:
            case PrimType_qf16:
                return emit_float<half>(primlit->pf16_value(),
                                        [](half v) { return half_float::isinf(v); },
                                        [](half v) { return half_float::isnan(v); });
            case PrimType_pf32:
            case PrimType_qf32:
                return emit_float<float>(primlit->pf32_value(),
                                         [](float v) { return std::isinf(v); },
                                         [](float v) { return std::isnan(v); });
            case PrimType_pf64:
            case PrimType_qf64:
                return emit_float<double>(primlit->pf64_value(),
                                          [](double v) { return std::isinf(v); },
                                          [](double v) { return std::isnan(v); });
        }
    } else if (auto variant = def->isa<Variant>()) {
        if (bb) {
            func_impls_.fmt("{} {};\n", convert(variant->type()), name);
            func_defs_.insert(def);
            if (auto value = emit_unsafe(variant->value()); !value.empty())
                bb->body.fmt("{}.data.{} = {};\n", name, variant->type()->as<VariantType>()->op_name(variant->index()),
                            value);
            bb->body.fmt("{}.tag = {};\n", name, variant->index());
            return name;
        } else {
            auto variant_type = variant->type()->as<VariantType>();
            s.fmt("{} ", constructor_prefix(variant_type));
            if (variant_type->has_payload()) {
                if (auto value = emit_constant(variant->value()); !value.empty()) // TODO what exactly does the value.empty() case represent and why do we emit bottom instead ?
                    s.fmt("{{ {{ {} }}, ", value);
                else
                    s.fmt("{{ {{ {} }}, ", emit_constant(world().bottom(variant->value()->type())));
            }
            s.fmt("{} }}", variant->index());
        }
    } else if (auto variant_index = def->isa<VariantIndex>()) {
        s.fmt("{}.tag", emit(variant_index->op(0)));
    } else if (auto variant_extract = def->isa<VariantExtract>()) {
        auto variant_type = variant_extract->value()->type()->as<VariantType>();
        s.fmt("{}.data.{}", emit(variant_extract->value()), variant_type->op_name(variant_extract->index()));
    } else if (auto load = def->isa<Load>()) {
        emit_unsafe(load->mem());
        auto ptr = emit(load->ptr());
        emitted_type = load->out_val()->type();
        s.fmt("*{}", ptr);
    } else if (auto store = def->isa<Store>()) {
        // TODO: IndefiniteArray should be removed
        if (store->val()->isa<IndefiniteArray>())
            return "";
        emit_unsafe(store->mem());
        emitted_type = store->val()->type();
        s.fmt("(*{} = {})", emit(store->ptr()), emit(store->val()));
        if (bb) {
            // stores can be expressions in C, but we probably don't care about that.
            bb->body.fmt("{};\n", s.str());
            return "";
        }
    } else if (auto slot = def->isa<Slot>()) {
        assert(bb && "basic block is required for slots");
        emit_unsafe(slot->frame());
        auto t = convert(slot->alloced_type());
        func_impls_.fmt("{} {}_slot;\n", t, name);
        func_impls_.fmt("{}* {} = &{}_slot;\n", t, name, name);
        func_defs_.insert(def);
        if (hls_top_scope)
            func_impls_ <<"#pragma HLS STREAM variable = "<< name << " depth = 5" << "\n";
        return name;
    } else if (auto alloc = def->isa<Alloc>()) {
        assert(bb && "basic block is required for allocating");
        use_malloc_ = true;
        emit_unsafe(alloc->mem());
        auto t = convert(alloc->alloced_type());
        func_impls_.fmt("{}* {};\n", t, name);
        func_defs_.insert(def);

        if (alloc->alloced_type()->isa<IndefiniteArrayType>()) {
            auto extra = emit(alloc->extra());
            bb->body.fmt("{} = malloc(sizeof({}) * {});\n", name, t, extra);
        } else {
            bb->body.fmt("{} = malloc(sizeof({}));\n", name, t);
        }
        return name;
    } else if (auto cell = def->isa<Cell>()) {
        use_malloc_ = true;
        auto ptr_t = cell->type()->as<PtrType>();
        if (cell->is_heap_allocated())
            func_impls_.fmt("{} {} = malloc(sizeof({})); // heap allocation\n", convert(ptr_t), name, convert(cell->contents()->type()));
        else
            func_impls_.fmt("{} {} = alloca(sizeof({})); // stack allocation\n", convert(ptr_t), name, convert(cell->contents()->type()));
        defs_[cell] = name;
        bb->body.fmt("*{} = {};\n", name, emit(cell->contents()));
        return name;
    } else if (auto enter = def->isa<Enter>()) {
        return emit_unsafe(enter->mem());
    } else if (auto lea = def->isa<LEA>()) {
        auto ptr = emit(lea->ptr());
        s.fmt("(&({})", ptr);
        emit_access(s, lea->ptr_pointee(), lea->index(), "->");
        s.fmt(")");
    } else if (auto ass = def->isa<Assembly>()) {
        assert(bb && "basic block is required for asm");
        if (ass->is_alignstack() || ass->is_inteldialect())
            world().wdef(ass, "stack alignment and inteldialect flags unsupported for C output");

        emit_unsafe(ass->mem());
        size_t num_inputs = ass->num_inputs();
        auto inputs = Array<std::string>(num_inputs);
        for (size_t i = 0; i != num_inputs; ++i)
            inputs[i] = emit_unsafe(ass->input(i));

        std::vector<std::string> outputs;
        if (auto tup = ass->type()->isa<TupleType>()) {
            for (size_t i = 1, e = tup->num_ops(); i != e; ++i) {
                auto name = ass->out(i)->unique_name();
                func_impls_.fmt("{} {};\n", convert(tup->types()[i]), name);
                func_defs_.insert(ass->out(i));
                outputs.emplace_back(name);
                defs_[ass->out(i)] = name;
            }
        }

        auto s = ass->asm_template();

        // escape chars
        for (auto [esc, subst] : {std::pair("\a", "\\a"), std::pair("\b", "\\b"),
                                  std::pair("\f", "\\f"), std::pair("\n", "\\n"),
                                  std::pair("\r", "\\r"), std::pair("\t", "\\t"), std::pair("\v", "\\v")}) {
            s = std::regex_replace(s, std::regex(esc), subst);
        }

        // TODO maybe we only want to do this conversion for certain C dialects?
        s = std::regex_replace(s, std::regex("(%)([[:alpha:]])"),  "%%$2");   // %eax -> %%eax
        s = std::regex_replace(s, std::regex("(\\$)([[:digit:]])"), "%$2");   // $1 -> %1, $$1 -> $%1
        s = std::regex_replace(s, std::regex("(\\$%)([[:digit:]])"), "$$$2"); // $%1 -> $$1

        // TODO we probably want to have a smarter way of doing this
        auto conv = [&](std::string constr) {
            constr = std::regex_replace(constr, std::regex("\\{|\\}"), ""); // remove braces
            constr = std::regex_replace(constr, std::regex("rax"),  "a");
            constr = std::regex_replace(constr, std::regex("rbx"),  "b");
            constr = std::regex_replace(constr, std::regex("rcx"),  "c");
            constr = std::regex_replace(constr, std::regex("rdx"),  "d");
            constr = std::regex_replace(constr, std::regex("rsi"),  "S");
            constr = std::regex_replace(constr, std::regex("rdi"),  "D");
            // TODO more cases
            return constr;
        };

        auto oconstrs = ass->output_constraints();
        auto iconstrs = ass-> input_constraints();
        auto clobbers = ass->clobbers();

        bb->body.fmt("asm {}(\"{}\"\t\n", ass->has_sideeffects() ? "volatile " : "", s);
        bb->body.fmt(": ").rangei(oconstrs, ", ", [&](size_t i) { bb->body.fmt("\"{}\" ({})", conv(oconstrs[i]), outputs[i]); }).fmt(" /* outputs */\n");
        bb->body.fmt(": ").rangei(iconstrs, ", ", [&](size_t i) { bb->body.fmt("\"{}\" ({})", conv(iconstrs[i]),  inputs[i]); }).fmt(" /* inputs */\n");
        bb->body.fmt(": ").rangei(clobbers, ", ", [&](size_t i) { bb->body.fmt("\"{}\"",      conv(clobbers[i])            ); }).fmt(" /* clobbers */\b\n);\n");
        return "";
    } else if (auto select = def->isa<Select>()) {
        auto cond = emit_unsafe(select->cond());
        auto tval = emit_unsafe(select->tval());
        auto fval = emit_unsafe(select->fval());
        s.fmt("({} ? {} : {})", cond, tval, fval);
    } else if (auto global = def->isa<Global>()) {
        assert(!global->init()->isa_nom<Continuation>());
        if (global->is_mutable() && lang_ != Lang::C99)
            world().wdef(global, "{}: Global variable '{}' will not be synced with host", lang_as_string(lang_), global);

        auto converted_type = convert(global->alloced_type());

        std::string prefix = device_prefix();
        std::string suffix = "";

        if (lang_ == Lang::OpenCL && use_channels_) {
            std::replace(name.begin(), name.end(), '_', 'X'); // xilinx pipe name restriction
            suffix = " __attribute__((xcl_reqd_pipe_depth(32)))";
        }

        vars_decls_.fmt("{}{} g_{} {}", prefix, converted_type, name, suffix);
        if (global->init()->isa<Bottom>())
            vars_decls_.fmt("; // bottom\n");
        else
            vars_decls_.fmt(" = {};\n", emit_constant(global->init()));
        if (use_channels_)
            s.fmt("g_{}", name);
        else
            s.fmt("&g_{}", name);

        return s.str();
    } else if (def->isa<Bottom>()) {
        return emit_bottom(def->type());
    } else {
        THORIN_UNREACHABLE;
    }

    if (bb) {
        func_impls_.fmt("{} {};\n", convert(emitted_type), name);
        func_defs_.insert(def);
        bb->body.fmt("{} = {};\n", name, s.str());
        return name;
    } else
        return "(" + s.str() + ")";
}

std::string CCodeGen::emit_fun_head(Continuation* cont, bool is_proto) {
    StringStream s;

    // Emit function qualifiers
    auto config = cont->is_exported() && kernel_config_.count(cont)
        ? kernel_config_.find(cont)->second.get() : nullptr;
    if (cont->is_exported()) {
        auto config = kernel_config_.find(cont);
        switch (lang_) {
            default: break;
            case Lang::CUDA:
                s << "__global__ ";
                if (!is_proto && config != kernel_config_.end()) {
                    auto block = config->second->as<GPUKernelConfig>()->block_size();
                    if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0)
                        s.fmt("__launch_bounds__({} * {} * {}) ", std::get<0>(block), std::get<1>(block), std::get<2>(block));
                }
                break;
            case Lang::OpenCL:
                if (!is_proto && config != kernel_config_.end()) {
                    auto block = config->second->as<GPUKernelConfig>()->block_size();

                    // See "Intel FPGA SDK for OpenCL"
                    if (block == std::tuple(1, 1, 1)) {
                        s << guarded_statement(cl_dialect_guard(CLDialect::INTEL), [&](Stream& gs){
                            gs << "__attribute__((max_global_work_dim(0)))\n";
                            if (!has_concrete_params(cont)) {
                                gs << "__attribute__((autorun))\n";
                            }
                            gs << "#else\n" << "__attribute__((reqd_work_group_size(1, 1, 1)))\n";
                        });
                    } else
                        s.fmt("__attribute__((reqd_work_group_size({}, {}, {})))\n", std::get<0>(block), std::get<1>(block), std::get<2>(block));
                }
                s << "__kernel ";
                break;
        }
    } else if (lang_ == Lang::CUDA) {
        s << "__device__ ";
    } else if (!world().is_external(cont)) {
        s << "static ";
    }

    s.fmt("{} {}(",
        convert(mangle_return_type(cont->type()->return_param_type())),
        !world().is_external(cont) ? cont->unique_name() : cont->name());

    // Emit and store all first-order params
    bool needs_comma = false;
    for (size_t i = 0, n = cont->num_params(); i < n; ++i) {
        auto param = cont->param(i);
        if (lang_ == Lang::C99 && param->type()->isa<ReturnType>()) {
            defs_[param] = "&return_buf";
            continue;
        }
        if (!is_concrete(param)) {
            defs_[param] = {};
            continue;
        }
        if (needs_comma) s.fmt(", ");

        // TODO: This should go in favor of a prepare pass that rewrites the kernel parameters
        if (lang_ == Lang::OpenCL && cont->is_exported() && is_passed_via_buffer(param)) {
            // OpenCL structs are passed via buffer; the parameter is a pointer to this buffer
            s << "__global " << convert(param->type()) << "*";
            if (!is_proto) s.fmt(" {}_", param->unique_name());
        } else if (lang_ == Lang::HLS && cont->is_exported() && param->type()->isa<PtrType>() && param->type()->as<PtrType>()->pointee()->isa<ArrayType>()) {
            auto array_size = config->as<HLSKernelConfig>()->param_size(param);
            assert(array_size > 0);
            auto ptr_type = param->type()->as<PtrType>();
            auto elem_type = ptr_type->pointee();
            if (auto array_type = elem_type->isa<ArrayType>()){
                elem_type = array_type->elem_type();
            }
            if (interface == HlsInterface::HPC_STREAM) {
                s << "hls::stream<" << convert(elem_type) <<">*";
                if (!is_proto)
                    s << " " << param->unique_name();
            } else {
                s << convert(elem_type);
                if (!is_proto)
                    s << " " << param->unique_name();
                s << "[" << array_size << "]";
            }

        } else {
            std::string qualifier;
            if (cont->is_exported() && (lang_ == Lang::OpenCL || lang_ == Lang::CUDA) &&
                config && config->as<GPUKernelConfig>()->has_restrict() &&
                param->type()->isa<PtrType>())
            {
                qualifier = lang_ == Lang::CUDA ? " __restrict" : " restrict";
            }
            s.fmt("{}{}", convert(param->type()), qualifier);
            if (!is_proto) s.fmt(" {}", param->unique_name());
        }
        needs_comma = true;
    }
    s << ")";
    return s.str();
}

std::string CCodeGen::emit_fun_decl(Continuation* cont) {
    if (cont->cc() != CC::Device)
        func_decls_.fmt("{};\n", emit_fun_head(cont, true));
    return !world().is_external(cont) ? cont->unique_name() : cont->name();
}

Stream& CCodeGen::emit_debug_info(Stream& s, const Def* def) {
    if (debug_ && !def->loc().file.empty())
        return s.fmt("#line {} \"{}\"\n", def->loc().begin.row, def->loc().file);
    return s;
}

void CCodeGen::emit_c_int() {
    // Do not emit C interfaces for definitions that are not used
    thorin_.cleanup();

    for (auto def : world().defs()) {
        auto cont = def->isa_nom<Continuation>();
        if (!cont)
            continue;
        if (!cont->is_external())
            continue;
        if (cont->cc() != CC::C && cont->is_imported())
            continue;

        // Generate C types for structs used by imported or exported functions
        for (auto op : cont->type()->types()) {
            if (auto fn_type = op->isa<FnType>()) {
                // Convert the return types as well (those are embedded in return continuations)
                for (auto other_op : fn_type->types()) {
                    if (!other_op->isa<FnType>())
                        convert(other_op);
                }
            } else
                convert(op);
        }

        // Generate function prototypes only for exported functions
        if (!cont->is_exported())
            continue;
        assert(cont->is_returning());
        emit_fun_decl(cont);
    }

    size_t pos = world().name().find_last_of("\\/");
    pos = (pos == std::string::npos) ? 0 : pos + 1;
    auto guard = world().name().substr(pos) + ".h";
    auto file_name = world().name() + ".h";

    // Generate a valid include guard macro name
    if (!std::isalpha(guard[0]) && guard[0] != '_') guard.insert(guard.begin(), '_');
    transform(guard.begin(), guard.end(), guard.begin(), [] (char c) -> char {
        if (!std::isalnum(c)) return '_';
        return ::toupper(c);
    });
    guard[guard.length() - 2] = '_';

    stream_.fmt("/* {}: C Interface file generated by Thorin */\n", file_name);
    stream_.fmt("#ifndef {}\n", guard);
    stream_.fmt("#define {}\n\n", guard);
    stream_.fmt("#ifdef __cplusplus\n");
    stream_.fmt("extern \"C\" {{\n");
    stream_.fmt("#endif\n\n");

    stream_.fmt("#include <stdbool.h>\n"    // for the 'bool' type
                "#include <stdint.h>\n\n");   // for the fixed-width integer types

    stream_.fmt("typedef   int8_t  i8;\n"
                "typedef  uint8_t  u8;\n"
                "typedef  int16_t i16;\n"
                "typedef uint16_t u16;\n"
                "typedef  int32_t i32;\n"
                "typedef uint32_t u32;\n"
                "typedef  int64_t i64;\n"
                "typedef uint64_t u64;\n"
                "typedef    float f32;\n"
                "typedef   double f64;\n\n");

    if (!type_decls_.str().empty())
        stream_.fmt("{}\n", type_decls_.str());
    if (!func_decls_.str().empty())
        stream_.fmt("{}\n", func_decls_.str());
    if (!vars_decls_.str().empty())
        stream_.fmt("{}\n", vars_decls_.str());

    stream_.fmt("#ifdef __cplusplus\n");
    stream_.fmt("}}\n");
    stream_.fmt("#endif\n\n");
    stream_.fmt("#endif /* {} */\n", guard);
}

template <typename T, typename IsInfFn, typename IsNanFn>
std::string CCodeGen::emit_float(T t, IsInfFn is_inf, IsNanFn is_nan) {
    auto emit_nn = [&] (std::string def, std::string cuda, std::string opencl) {
        switch (lang_) {
            case Lang::CUDA:   return cuda;
            case Lang::OpenCL: return opencl;
            default:           return def;
        }
    };

    if (is_inf(t)) {
        if (std::is_same<T, half>::value) {
            return emit_nn("std::numeric_limits<half>::infinity()", "__short_as_half(0x7c00)", "as_half(0x7c00)");
        } else if (std::is_same<T, float>::value) {
            return emit_nn("std::numeric_limits<float>::infinity()", "__int_as_float(0x7f800000)", "as_float(0x7f800000)");
        } else {
            return emit_nn("std::numeric_limits<double>::infinity()", "__longlong_as_double(0x7ff0000000000000LL)", "as_double(0x7ff0000000000000LL)");
        }
    } else if (is_nan(t)) {
        if (std::is_same<T, half>::value) {
            return emit_nn("nan(\"\")", "__short_as_half(0x7fff)", "as_half(0x7fff)");
        } else if (std::is_same<T, float>::value) {
            return emit_nn("nan(\"\")", "__int_as_float(0x7fffffff)", "as_float(0x7fffffff)");
        } else {
            return emit_nn("nan(\"\")", "__longlong_as_double(0x7fffffffffffffffLL)", "as_double(0x7fffffffffffffffLL)");
        }
    }

    StringStream s;
    auto float_mode = lang_ == Lang::CUDA ? std::scientific : std::hexfloat;
    const char* suf = "", * pref = "";

    if (std::is_same<T, half>::value) {
        if (lang_ == Lang::CUDA) {
            pref = "__float2half(";
            suf  = ")";
        } else {
            suf = "h";
        }
    } else if (std::is_same<T, float>::value) {
        suf  = "f";
    }

    s.fmt("{}{}{}{}", float_mode, pref, t, suf);
    return s.str();
}

std::string CCodeGen::array_name(const DefiniteArrayType* array_type) {
    return "array_" + std::to_string(array_type->gid());
}

std::string CCodeGen::tuple_name(const TupleType* tuple_type) {
    return "tuple_" + std::to_string(tuple_type->gid());
}

std::string CCodeGen::fn_name(const FnType* fn_type) {
    return "fn_" + std::to_string(fn_type->gid());
}

std::string CCodeGen::closure_name(const ClosureType* fn_type) {
    return "closure_" + std::to_string(fn_type->gid());
}

std::string CCodeGen::return_name(const ReturnType* fn_type) {
    return "return_" + std::to_string(fn_type->gid());
}

//------------------------------------------------------------------------------

void CodeGen::emit_stream(std::ostream& stream) {
    Stream s(stream);
    CCodeGen(thorin(), kernel_config_, s, lang_, debug_, flags_).emit_module();
}

void emit_c_int(Thorin& thorin, Stream& stream) {
    std::string flags;
    CCodeGen(thorin, {}, stream, Lang::C99, false, flags).emit_c_int();
}

//------------------------------------------------------------------------------

}
