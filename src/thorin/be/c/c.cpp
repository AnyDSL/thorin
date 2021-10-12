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
#include <unordered_map>
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
    CCodeGen(World& world, const Cont2Config& kernel_config, Stream& stream, Lang lang, bool debug)
        : world_(world)
        , kernel_config_(kernel_config)
        , lang_(lang)
        , fn_mem_(world.fn_type({world.mem_type()}))
        , debug_(debug)
        , stream_(stream)
    {}

    World& world() const { return world_; }
    void emit_module();
    void emit_c_int();
    void emit_epilogue(Continuation*);

    std::string emit_bb(BB&, const Def*);
    std::string emit_constant(const Def*);
    std::string emit_bottom(const Type*);
    std::string emit_def(BB*, const Def*);
    void emit_access(Stream&, const Type*, const Def*, const std::string_view& = ".");
    bool is_valid(const std::string& s) { return !s.empty(); }
    std::string emit_fun_head(Continuation*, bool = false);
    std::string emit_fun_decl(Continuation*);
    std::string prepare(const Scope&);
    void prepare(Continuation*, const std::string&);
    void finalize(const Scope&);
    void finalize(Continuation*);

private:
    std::string convert(const Type*);
    std::string addr_space_prefix(AddrSpace);
    std::string constructor_prefix(const Type*);
    std::string device_prefix();
    Stream& emit_debug_info(Stream&, const Def*);

    template <typename T, typename IsInfFn, typename IsNanFn>
    std::string emit_float(T, IsInfFn, IsNanFn);

    std::string array_name(const DefiniteArrayType*);
    std::string tuple_name(const TupleType*);

    World& world_;
    const Cont2Config& kernel_config_;
    Lang lang_;
    const FnType* fn_mem_;
    bool use_math_ = false;
    bool use_fp_64_ = false;
    bool use_fp_16_ = false;
    bool use_channels_ = false;
    bool use_align_of_ = false;
    bool use_memcpy_ = false;
    bool use_malloc_ = false;
    bool debug_;

    Stream& stream_;
    StringStream func_impls_;
    StringStream func_decls_;
    StringStream type_decls_;
    /// Tracks defs that have been emitted as local variables of the current function
    DefSet func_defs_;

    std::ostringstream macro_xilinx_;
    std::ostringstream macro_intel_;

    /// emit_fun_head may want to add its own pragmas so we put this in a global
    std::string hls_pragmas_;
    std::unordered_map<const Continuation*, FuncMode> builtin_funcs_; // OpenCL builtin functions
};

static inline const std::string lang_as_string(Lang lang) {
    switch (lang) {
        default:
        case Lang::C99:    return "C99";
        case Lang::HLS:    return "HLS";
        case Lang::CUDA:   return "CUDA";
        case Lang::OpenCL: return "OpenCL";
    }
}

static inline bool is_string_type(const Type* type) {
    if (auto array = type->isa<DefiniteArrayType>())
        if (auto primtype = array->elem_type()->isa<PrimType>())
            if (primtype->primtype_tag() == PrimType_pu8)
                return true;
    return false;
}

// TODO I think we should have a full-blown channel type
inline bool is_channel_type(const StructType* struct_type) {
    return struct_type->name().str().find("channel") != std::string::npos;
}

/// Returns true when the param carries concrete data in the final generated code
inline bool is_concrete_param(const Param* param) { return !is_mem(param) && param->order() == 0 && !is_unit(param);}
inline bool has_concrete_params(Continuation* cont) {
    return std::any_of(cont->params().begin(), cont->params().end(), [](const Param* param) { return is_concrete_param(param); });
}

inline bool get_interface(HlsInterface &interface, HlsInterface &gmem) {
    const char* fpga_env = std::getenv("ANYDSL_FPGA");
    if (fpga_env != NULL) {
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

/*
 * convert
 */

std::string CCodeGen::convert(const Type* type) {
    if (auto res = types_.lookup(type)) return *res;

    StringStream s;
    std::string name;

    if (type == world().unit() || type->isa<MemType>() || type->isa<FrameType>())
        s << "void";
    else if (auto primtype = type->isa<PrimType>()) {
        switch (primtype->primtype_tag()) {
            case PrimType_bool:                     s << "bool";                      break;
            case PrimType_ps8:  case PrimType_qs8:  s << "char";                      break;
            case PrimType_pu8:  case PrimType_qu8:  s << "unsigned char";             break;
            case PrimType_ps16: case PrimType_qs16: s << "short";                     break;
            case PrimType_pu16: case PrimType_qu16: s << "unsigned short";            break;
            case PrimType_ps32: case PrimType_qs32: s << "int";                       break;
            case PrimType_pu32: case PrimType_qu32: s << "unsigned int";              break;
            case PrimType_ps64: case PrimType_qs64: s << "long";                      break;
            case PrimType_pu64: case PrimType_qu64: s << "unsigned long";             break;
            case PrimType_pf32: case PrimType_qf32: s << "float";                     break;
            case PrimType_pf16: case PrimType_qf16: s << "half";   use_fp_16_ = true; break;
            case PrimType_pf64: case PrimType_qf64: s << "double"; use_fp_64_ = true; break;
            default: THORIN_UNREACHABLE;
        }
        if (primtype->is_vector())
            s << primtype->length();
    } else if (auto array = type->isa<IndefiniteArrayType>()) {
        return types_[type] = convert(array->elem_type()); // IndefiniteArrayType always occurs within a pointer
    } else if (type->isa<FnType>()) {
        assert(false && "todo");
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
        s.rangei(tuple->ops(), "\n", [&](size_t i) { s.fmt("{} e{};", convert(tuple->op(i)), i); });
        s.fmt("\b\n}} {};\n", name);
    } else if (auto variant = type->isa<VariantType>()) {
        types_[variant] = name = variant->name().str();
        auto tag_type =
            variant->num_ops() < (UINT64_C(1) <<  8u) ? world_.type_qu8()  :
            variant->num_ops() < (UINT64_C(1) << 16u) ? world_.type_qu16() :
            variant->num_ops() < (UINT64_C(1) << 32u) ? world_.type_qu32() :
                                                        world_.type_qu64();
        s.fmt("typedef struct {{\t\n");

        // This is required because we have zero-sized types but C/C++ do not
        if (variant->has_payload()) {
            s.fmt("union {{\t\n");
            s.rangei(variant->ops(), "\n", [&] (size_t i) {
                if (is_type_unit(variant->op(i)))
                    s << "// ";
                s.fmt("{} {};", convert(variant->op(i)), variant->op_name(i));
            });
            s.fmt("\b\n}} data;\n");
        }

        s.fmt("{} tag;", convert(tag_type));
        s.fmt("\b\n}} {};\n", name);
    } else if (auto struct_type = type->isa<StructType>()) {
        name = struct_type->name().str();
        if (lang_ == Lang::OpenCL && use_channels_) {
            s.fmt("typedef {} {}_{};", convert(struct_type->op(0)), name, struct_type->gid());
            name = (struct_type->name().str() + "_" + std::to_string(type->gid()));
        } else if (is_channel_type(struct_type) && lang_ == Lang::HLS) {
            s.fmt("typedef {} {}_{};", convert(struct_type->op(0)), name, struct_type->gid());
            name = ("hls::stream<" + struct_type->name().str() + "_" + std::to_string(type->gid()) + ">");
        } else {
            s.fmt("typedef struct {{\t\n");
            s.rangei(struct_type->ops(), "\n", [&] (size_t i) { s.fmt("{} {};", convert(struct_type->op(i)), struct_type->op_name(i)); });
            s.fmt("\b\n}} {};\n", name);
        }
        if (struct_type->name().str().find("channel_") != std::string::npos)
            use_channels_ = true; // TODO is there a risk of missing this before we emit something for real ?
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
            if(use_channels_)
                return "PIPE ";
            else
                return "__constant ";
    }
}

/*
 * emit
 */

HlsInterface interface, gmem_config;
auto interface_status = get_interface(interface, gmem_config);

void CCodeGen::emit_module() {
    // TODO do something to make those ifdefs sane to work with -H
    if (lang_ == Lang::OpenCL)
        func_decls_ << "#ifndef __xilinx__" << "\n";

    // removing function prototypes from HLS synthesis
    if (lang_ == Lang::HLS)
        func_decls_ << "#ifndef __SYNTHESIS__\n";

    Continuation* hls_top = nullptr;
    Scope::for_each(world(), [&] (const Scope& scope) {
        if (scope.entry()->name() == "hls_top")
            hls_top = scope.entry();
        else
            emit_scope(scope);
    });
    if (hls_top)
        emit_scope(Scope(hls_top));

    if (lang_ == Lang::OpenCL)
        func_decls_ << "#endif /* __xilinx__ */"<< "\n";
    if (lang_ == Lang::HLS)
        func_decls_ << "#endif /* __SYNTHESIS__ */\n";

    if (lang_ == Lang::OpenCL) {
        if (use_channels_) {
            std::string write_channel_params = "(channel, val) ";
            std::string read_channel_params = "(val, channel) ";

            macro_xilinx_ << "#if defined(__xilinx__)" << "\n" << "#define PIPE pipe" << "\n";
            macro_intel_  << "\n" << "#elif defined(INTELFPGA_CL)" << "\n"
            <<"#pragma OPENCL EXTENSION cl_intel_channels : enable" << "\n" << "#define PIPE channel" << "\n";
            for (auto map : builtin_funcs_) {
                if (map.first->is_channel()) {
                    if (map.second == FuncMode::Write) {
                        macro_xilinx_ << "#define " << map.first->name() << write_channel_params << "write_pipe_block(channel, &val)" << "\n";
                        macro_intel_ << "#define "<< map.first->name() << write_channel_params << "write_channel_intel(channel, val)" << "\n";
                    } else if (map.second == FuncMode::Read) {
                        macro_xilinx_ << "#define " << map.first->name() << read_channel_params << "read_pipe_block(channel, &val)" << "\n";
                        macro_intel_  << "#define " << map.first->name() << read_channel_params << "val = read_channel_intel(channel)" << "\n";
                    }
                }
            }
            stream_ << macro_xilinx_.str() << macro_intel_.str();
            stream_ << "\n" << "#else" << "\n" << "#define PIPE pipe"<< "\n";
            stream_ << "\n" << "#endif" << "\n";

            if (use_fp_16_)
                stream_ << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" << "\n";
            if (use_fp_64_)
                stream_ << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << "\n";
            if (use_channels_ || use_fp_16_ || use_fp_64_)
                stream_ << "\n";
        }
    }

    if (lang_ == Lang::C99) {
        stream_.fmt("#include <stdbool.h>\n"); // for the 'bool' type
        if (use_align_of_)
            stream_.fmt("#include <stdalign.h>\n"); // for 'alignof'
        if (use_memcpy_)
            stream_.fmt("#include <string.h>\n"); // for 'memcpy'
        if (use_malloc_)
            stream_.fmt("#include <stdlib.h>\n"); // for 'malloc'
        if (use_math_)
            stream_.fmt("#include <math.h>\n"); // for 'cos'/'sin'/...
        stream_.fmt("\n");
    }

    if (lang_ == Lang::CUDA && use_fp_16_) {
        stream_.fmt("#include <cuda_fp16.h>\n\n");
        stream_.fmt("#if __CUDACC_VER_MAJOR__ > 8\n");
        stream_.fmt("#define half __half_raw\n");
        stream_.fmt("#endif\n\n");
    }
    if (lang_ == Lang::HLS)
        stream_ << "#include \"hls_stream.h\""<< "\n" << "#include \"hls_math.h\""<< "\n";

    if (lang_ == Lang::CUDA || lang_ == Lang::HLS) {
        stream_.fmt("extern \"C\" {{\n");
    }

    stream_ << type_decls_.str();
    stream_.endl() << func_decls_.str();

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

inline bool is_passed_via_buffer(const Param* param) {
    return param->type()->isa<DefiniteArrayType>()
        || param->type()->isa<StructType>()
        || param->type()->isa<TupleType>();
}

inline const Type* ret_type(const FnType* fn_type) {
    auto ret_fn_type = (*std::find_if(
        fn_type->ops().begin(), fn_type->ops().end(), [] (const Type* op) {
            return op->order() % 2 == 1;
        }))->as<FnType>();
    std::vector<const Type*> types;
    for (auto op : ret_fn_type->ops()) {
        if (op->isa<MemType>() || is_type_unit(op) || op->order() > 0) continue;
        types.push_back(op);
    }
    return fn_type->table().tuple_type(types);
}

static inline const Type* pointee_or_elem_type(const PtrType* ptr_type) {
    auto elem_type = ptr_type->as<PtrType>()->pointee();
    if (auto array_type = elem_type->isa<ArrayType>())
        elem_type = array_type->elem_type();
    return elem_type;
}

std::string CCodeGen::prepare(const Scope& scope) {
    auto cont = scope.entry();

    for (auto param : cont->params()) {
        defs_[param] = param->unique_name();
        if (lang_ == Lang::HLS && cont->is_exported() && param->type()->isa<PtrType>()) {
            auto elem_type = pointee_or_elem_type(param->type()->as<PtrType>());
            if (elem_type->isa<StructType>() || elem_type->isa<DefiniteArrayType>())
                hls_pragmas_ += "#pragma HLS data_pack variable=" + param->unique_name() + " struct_level\n";
        }
    }

    if (lang_ == Lang::HLS && cont->is_exported()) {
        if (cont->name() == "hls_top") {
            if (interface_status) {
                if (cont->num_params() > 2) {
                    size_t hls_gmem_index = 0;
                    for (auto param : cont->params()) {
                        if (!is_concrete_param(param))
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
                        hls_pragmas_ += "#pragma HLS INTERFACE ap_ctrl_none port = return\n";
                    else if (interface == HlsInterface::HPC)
                        hls_pragmas_ += "#pragma HLS INTERFACE ap_ctrl_chain port = return\n";
                }
            } else {
                interface = HlsInterface::None;
                world().WLOG("HLS accelerator generated with no interface");
            }
            hls_pragmas_ += "#pragma HLS top name = hls_top\n";
            if (use_channels_)
                hls_pragmas_ += "#pragma HLS DATAFLOW\n";
        } else if (use_channels_) {
            hls_pragmas_ += "#pragma HLS INLINE off\n";
        }
    }

    func_impls_.fmt("{} {{", emit_fun_head(cont));
    func_impls_.fmt("\t\n");

    if (!hls_pragmas_.empty())
        func_impls_.fmt("{}", hls_pragmas_);
    hls_pragmas_.clear();

    // Load OpenCL structs from buffers
    // TODO: See above
    for (auto param : cont->params()) {
        if (!is_concrete_param(param))
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
            if (!is_concrete_param(param)) {
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

inline std::string make_identifier(const std::string& str) {
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

inline std::string label_name(const Def* def) {
    return make_identifier(def->as_continuation()->unique_name());
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

void CCodeGen::emit_epilogue(Continuation* cont) {
    auto&& bb = cont2bb_[cont];
    emit_debug_info(bb.tail, cont->arg(0));

    if (cont->callee() == entry_->ret_param()) { // return
        std::vector<std::string> values;
        std::vector<const Type*> types;

        for (auto arg : cont->args()) {
            if (auto val = emit_unsafe(arg); !val.empty()) {
                values.emplace_back(val);
                types.emplace_back(arg->type());
            }
        }

        switch (values.size()) {
            case 0: bb.tail.fmt(lang_ == Lang::HLS ? "return void();" : "return;"); break;
            case 1: bb.tail.fmt("return {};", values[0]); break;
            default:
                auto tuple = convert(world().tuple_type(types));
                bb.tail.fmt("{} ret_val;\n", tuple);
                for (size_t i = 0, e = types.size(); i != e; ++i)
                    bb.tail.fmt("ret_val.e{} = {};\n", i, values[i]);
                bb.tail.fmt("return ret_val;");
                break;
        }
    } else if (cont->callee() == world().branch()) {
        auto c = emit(cont->arg(0));
        auto t = label_name(cont->arg(1));
        auto f = label_name(cont->arg(2));
        bb.tail.fmt("if ({}) goto {}; else goto {};", c, t, f);
    } else if (auto callee = cont->callee()->isa_continuation(); callee && callee->intrinsic() == Intrinsic::Match) {
        bb.tail.fmt("switch ({}) {{\t\n", emit(cont->arg(0)));

        for (size_t i = 2; i < cont->num_args(); i++) {
            auto arg = cont->arg(i)->as<Tuple>();
            bb.tail.fmt("case {}: goto {};\n", emit_constant(arg->op(0)), label_name(arg->op(1)));
        }

        bb.tail.fmt("default: goto {};", label_name(cont->arg(1)));
        bb.tail.fmt("\b\n}}");
    } else if (cont->callee()->isa<Bottom>()) {
        bb.tail.fmt("return;  // bottom: unreachable");
    } else if (auto callee = cont->callee()->isa_continuation(); callee && callee->is_basicblock()) { // ordinary jump
        assert(callee->num_params() == cont->num_args());
        for (size_t i = 0, size = callee->num_params(); i != size; ++i) {
            if (auto arg = emit_unsafe(cont->arg(i)); !arg.empty())
                bb.tail.fmt("p_{} = {};\n", callee->param(i)->unique_name(), arg);
        }
        bb.tail.fmt("goto {};", label_name(callee));
    } else if (auto callee = cont->callee()->isa_continuation(); callee && callee->is_intrinsic()) {
        if (callee->intrinsic() == Intrinsic::Reserve) {
            if (!cont->arg(1)->isa<PrimLit>())
                world().edef(cont->arg(1), "reserve_shared: couldn't extract memory size");

            auto ret_cont = cont->arg(2)->as_continuation();
            auto elem_type = ret_cont->param(1)->type()->as<PtrType>()->pointee()->as<ArrayType>()->elem_type();
            func_impls_.fmt("{}{} {}_reserved[{}];\n",
                addr_space_prefix(AddrSpace::Shared), convert(elem_type),
                cont->unique_name(), emit_constant(cont->arg(1)));
            if (lang_ == Lang::HLS) {
                func_impls_.fmt("#pragma HLS dependence variable={}_reserved inter false\n", cont->unique_name());
                func_impls_.fmt("#pragma HLS data_pack  variable={}_reserved\n", cont->unique_name());
            }
            bb.tail.fmt("p_{} = {}_reserved;\n", ret_cont->param(1)->unique_name(), cont->unique_name());
            bb.tail.fmt("goto {};", label_name(ret_cont));
        } else if (callee->intrinsic() == Intrinsic::Pipeline) {
            assert((lang_ == Lang::OpenCL || lang_ == Lang::HLS) && "pipelining not supported on this backend");

            std::string interval;
            if (cont->arg(1)->as<PrimLit>()->value().get_s32() != 0)
                interval = emit_constant(cont->arg(1));

            auto begin = emit(cont->arg(2));
            auto end   = emit(cont->arg(3));
            if (lang_ == Lang::OpenCL) {
                bb.tail << guarded_statement(cl_dialect_guard(CLDialect::INTEL), [&](Stream& s){
                    s.fmt("#pragma ii {}\n", !interval.empty() ? interval : "1");
                });
                bb.tail << guarded_statement(cl_dialect_guard(CLDialect::XILINX), [&](Stream& s){
                    s.fmt("__attribute__((xcl_pipeline_loop({})))\n", !interval.empty() ? interval : "1");
                });
            }
            bb.tail.fmt("for (int i{} = {}; i{} < {}; i{}++) {{\t\n",
                callee->gid(), begin, callee->gid(), end, callee->gid());
            if (lang_ == Lang::HLS) {
                bb.tail << "#pragma HLS PIPELINE";
                if (!interval.empty())
                    bb.tail.fmt(" II={}", interval);
                bb.tail.fmt("\n");
            }

            auto body = cont->arg(4)->as_continuation();
            bb.tail.fmt("p_{} = i{};\n", body->param(1)->unique_name(), callee->gid());
            bb.tail.fmt("goto {};\n", label_name(body));

            // Emit a label that can be used by the "pipeline_continue()" intrinsic.
            bb.tail.fmt("\b\n{}: continue;\n}}\n", label_name(cont->arg(6)));
            bb.tail.fmt("goto {};", label_name(cont->arg(5)));
        } else if (callee->intrinsic() == Intrinsic::PipelineContinue) {
            bb.tail.fmt("goto {};", label_name(callee));
        } else {
            THORIN_UNREACHABLE;
        }
    } else if (auto callee = cont->callee()->isa_continuation()) { // function/closure call
        auto ret_cont = (*std::find_if(cont->args().begin(), cont->args().end(), [] (const Def* arg) {
            return arg->isa_continuation();
        }))->as_continuation();

        std::vector<std::string> args;
        for (auto arg : cont->args()) {
            if (arg == ret_cont) continue;
            if (auto emitted_arg = emit_unsafe(arg); !emitted_arg.empty())
                args.emplace_back(emitted_arg);
        }

        // Do not store the result of `void` calls
        auto ret_type = thorin::c::ret_type(callee->type());
        if (!is_type_unit(ret_type))
            bb.tail.fmt("{} ret_val = ", convert(ret_type));

        bb.tail.fmt("{}({, });\n", emit(callee), args);

        // Pass the result to the phi nodes of the return continuation
        if (!is_type_unit(ret_type)) {
            size_t i = 0;
            for (auto param : ret_cont->params()) {
                if (!is_concrete_param(param))
                    continue;
                if (ret_type->isa<TupleType>())
                    bb.tail.fmt("p_{} = ret_val.e{};\n", param->unique_name(), i++);
                else
                    bb.tail.fmt("p_{} = ret_val;\n", param->unique_name());
            }
        }
        bb.tail.fmt("goto {};", label_name(ret_cont));
    } else {
        THORIN_UNREACHABLE;
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
        s.range(type->ops(), ", ", [&] (const Type* op) { s << emit_bottom(op); });
        s << " }";
        return s.str();
    } else if (auto variant_type = type->isa<VariantType>()) {
        if (variant_type->has_payload()) {
            auto non_unit = *std::find_if(variant_type->ops().begin(), variant_type->ops().end(),
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
        s.fmt("{}e{}", prefix, emit_constant(index));
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
    } else {
        THORIN_UNREACHABLE;
    }
}

static inline bool is_const_primop(const Def* def) {
    return def->isa<PrimOp>() && !def->has_dep(Dep::Param);
}

std::string CCodeGen::emit_bb(BB& bb, const Def* def) {
    return emit_def(&bb, def);
}

std::string CCodeGen::emit_constant(const Def* def) {
    return emit_def(nullptr, def);
}

/// If bb is nullptr, then we are emitting a constant, otherwise we emit the def as a local variable
std::string CCodeGen::emit_def(BB* bb, const Def* def) {
    StringStream s;
    auto name = def->unique_name();
    const Type* emitted_type = def->type();

    if (is_unit(def)) return "";
    else if (auto bin = def->isa<BinOp>()) {
        const char* op;
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
            case PrimType_ps64: case PrimType_qs64: return std::to_string(primlit->ps64_value());
            case PrimType_pu64: case PrimType_qu64: return std::to_string(primlit->pu64_value());
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
        if (bb->cont->name() == "hls_top")
            func_impls_ << "#pragma HLS STREAM variable = "<< name << " depth = 5" << "\n";
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
                func_impls_.fmt("{} {};\n", convert(tup->op(i)), name);
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
        s.fmt("({} ? {} : {})", name, cond, tval, fval);
    } else if (auto global = def->isa<Global>()) {
        assert(!global->init()->isa_continuation());
        if (global->is_mutable() && lang_ != Lang::C99)
            world().wdef(global, "{}: Global variable '{}' will not be synced with host", lang_as_string(lang_), global);

        std::string prefix = device_prefix();
        std::string suffix = "";

        if (lang_ == Lang::OpenCL && use_channels_) {
            std::replace(name.begin(), name.end(), '_', 'X'); // xilinx pipe name restriction
            suffix = " __attribute__((xcl_reqd_pipe_depth(32)))";
        }

        func_decls_.fmt("{}{} g_{} {}", prefix, convert(global->alloced_type()), name, suffix);
        if (global->init()->isa<Bottom>())
            func_decls_.fmt("; // bottom\n");
        else
            func_decls_.fmt(" = {};\n", emit_constant(global->init()));
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
                    auto [bx, by, bz] = block;

                    // See "Intel FPGA SDK for OpenCL"
                    if (block == std::tuple(1, 1, 1)) {
                        s << guarded_statement(cl_dialect_guard(CLDialect::INTEL), [&](Stream& gs){
                            gs << "__attribute__((max_global_work_dim(0)))\n";
                            if (!has_concrete_params(cont)) {
                                gs << "__attribute__((autorun))\n";
                            }
                        });
                    } else
                        s.fmt("__attribute__((reqd_work_group_size({}, {}, {})))\n", std::get<0>(block), std::get<1>(block), std::get<2>(block));
                }
                s << "__kernel ";
                break;
        }
    } else if (lang_ == Lang::CUDA) {
        s << "__device__ ";
    } else if (cont->is_internal()) {
        s << "static ";
    }

    s.fmt("{} {}(",
        convert(ret_type(cont->type())),
        cont->is_internal() ? cont->unique_name() : cont->name());

    // Emit and store all first-order params
    bool needs_comma = false;
    for (size_t i = 0, n = cont->num_params(); i < n; ++i) {
        auto param = cont->param(i);
        if (!is_concrete_param(param)) {
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
            if (elem_type->isa<StructType>() || elem_type->isa<DefiniteArrayType>()) {
                hls_pragmas_ += "#pragma HLS data_pack variable=" + param->unique_name() + " struct_level\n";
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
    return cont->is_internal() ? cont->unique_name() : cont->name();
}

Stream& CCodeGen::emit_debug_info(Stream& s, const Def* def) {
    if (debug_ && !def->loc().file.empty())
        return s.fmt("#line {} \"{}\"\n", def->loc().begin.row, def->loc().file);
    return s;
}

void CCodeGen::emit_c_int() {
    // Do not emit C interfaces for definitions that are not used
    world().cleanup();

    for (auto cont : world().continuations()) {
        if (!cont->is_external())
            continue;

        // Generate C types for structs used by imported or exported functions
        for (auto op : cont->type()->ops()) {
            if (auto fn_type = op->isa<FnType>()) {
                // Convert the return types as well (those are embedded in return continuations)
                for (auto other_op : fn_type->ops()) {
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

    if (!type_decls_.str().empty())
        stream_.fmt("{}\n", type_decls_.str());
    if (!func_decls_.str().empty())
        stream_.fmt("{}\n", func_decls_.str());

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

//------------------------------------------------------------------------------

void CodeGen::emit_stream(std::ostream& stream) {
    Stream s(stream);
    CCodeGen(world(), kernel_config_, s, lang_, debug_).emit_module();
}

void emit_c_int(World& world, Stream& stream) {
    CCodeGen(world, {}, stream, Lang::C99, false).emit_c_int();
}

//------------------------------------------------------------------------------

}
