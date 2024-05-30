#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/transform/hls_dataflow.h"
#include "thorin/be/emitter.h"
#include "thorin/util/stream.h"
#include "c.h"

#include <cctype>
#include <cmath>
#include <regex>
#include <sstream>
#include <type_traits>
#include <unordered_map> // TODO don't use std::unordered_*
#include <unordered_set>
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
    CCodeGen(World& world, const Cont2Config& kernel_config, Stream& stream, Stream& graph_stream, Lang lang, bool debug, std::string& flags)
        : world_(world)
        , kernel_config_(kernel_config)
        , lang_(lang)
        , fn_mem_(world.fn_type({world.mem_type()}))
        , debug_(debug)
        , flags_(flags)
        , stream_(stream)
        , graph_stream_(graph_stream)
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
    std::string emit_class(Continuation*);
    std::string emit_fun_head(Continuation*, bool = false);
    std::string emit_fun_decl(Continuation*);
    std::string prepare(const Scope&);
    void prepare(Continuation*, const std::string&);
    void graph_ctor_gen(const Continuations&);
    void finalize(const Scope&);
    void finalize(Continuation*);

private:
    std::string convert(const Type*, bool = false);
    std::string addr_space_prefix(AddrSpace);
    std::string constructor_prefix(const Type*);
    std::string prefix_type(const Param* param);
    std::string device_prefix();
    Stream& emit_debug_info(Stream&, const Def*);
    bool get_interface(HlsInterface &interface, HlsInterface &gmem);
    bool get_cgra_options();
    auto get_config(Continuation* cont);
    auto get_vector_size(Continuation* cont);
    auto is_accum_type(const Type* type);
    auto is_mask_type(const Type* type);
    bool is_scalar_kernel();
    bool is_cgra_vector_kernel();
    bool has_vect_arg(Continuation*);
    std::unique_ptr<ApiConfig> special_device_api(const Continuation*);

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

    //auto [interface_status, hls_top_scope, cgra_graph_scope] = std::make_tuple(false, false, false);
    std::string flags_;
    Stream& stream_;
    Stream graph_stream_;

    StringStream func_impls_;
    StringStream func_decls_;
    StringStream type_decls_;
    StringStream vars_decls_;
    StringStream graph_ctor_;

    /// Tracks defs that have been emitted as local variables of the current function
    DefSet func_defs_;

    std::ostringstream macro_xilinx_;
    std::ostringstream macro_intel_;
    struct { bool hls = false; bool cgra_graph = false; } top_scope;
    //TODO: debug var should enable top debug point and add names to plio ports
    struct { bool sim_data = false; bool debug = false; int32_t iteration = -1;} options;
    size_t vector_size_;
    ContinuationMap<FuncMode> builtin_funcs_; // OpenCL builtin functions
};

static inline const std::string lang_as_string(Lang lang) {
    switch (lang) {
        default:     THORIN_UNREACHABLE;
        case Lang::C99:    return "C99";
        case Lang::HLS:    return "HLS";
        case Lang::CGRA:   return "CGRA";
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


inline bool is_mmul_type(const StructType* struct_type) {
    return struct_type->name().str().find("mmul") != std::string::npos;
}

inline bool is_mmul_type(const Type* type) {
    return type->isa<StructType>()->name().str().find("mmul") != std::string::npos;
}


/// Returns true when the def carries concrete data in the final generated code
inline bool is_concrete(const Def* def) { return !is_mem(def) && def->order() == 0 && !is_unit(def);}
inline bool has_concrete_params(Continuation* cont) {
    return std::any_of(cont->params().begin(), cont->params().end(), [](const Param* param) { return is_concrete(param); });
}

auto CCodeGen::get_config(Continuation* cont) {
    return cont->is_exported() && kernel_config_.count(cont) ? kernel_config_.find(cont)->second.get() : nullptr;
}

bool CCodeGen::is_scalar_kernel() {
    assert(lang_ == Lang::CGRA && "is_scalar_kernel is available only for CGRA");
    return (vector_size_ == 0 || vector_size_ == 1);
}


bool CCodeGen::is_cgra_vector_kernel() {
    return (lang_ == Lang::CGRA && (vector_size_ > 1));
}

bool CCodeGen::has_vect_arg(Continuation* cont) {
    assert(lang_ == Lang::CGRA && "has_vect_param is available only for CGRA");
    return cont->starts_with("aie::") && cont->ends_with("_v");
}

auto CCodeGen::get_vector_size(Continuation* cont) {
    assert(lang_ == Lang::CGRA && "vector_size is available only for CGRA");
    if(auto config = get_config(cont)) {
        assert(config->isa<CGRAKernelConfig>() && "CGRAKernelConfig expected");
        return config->as<CGRAKernelConfig>()->vector_size();
    }
    assert(false && "kernel has no config");
}

static auto prepare_flag(std::string flag) {
    std::string flag_str = flag;
    if (!flag.empty()) {
        for (auto& ch : flag_str)
            ch = std::toupper(ch, std::locale());
    }
    std::istringstream options_stream(flag_str);
    return options_stream;
}


bool CCodeGen::get_cgra_options() {

    auto opts = prepare_flag(flags_);
    auto found = false;
    if (!(opts.str().empty())) {
        std::string token;

        auto matches = [&] (std::string str) {
            if (token.compare(str) == 0)
                found = true;
            else if(auto equal_sign_pos = token.find("="); equal_sign_pos != std::string::npos) {
                found = (token.substr(0, (equal_sign_pos)).compare(str) == 0);
            } else
                found = false;
            return found;
        };

        auto option_val = [&] (std::string token) {
            std::string val = token.substr(token.find("=") + 1);
            return std::stoi(val);
        };

        while (std::getline(opts, token, ',')) {

            if (matches("USE_SIM_DATA")) {
                options.sim_data = true;
                continue;
            } else if (matches("ITERATION")) {
                options.iteration = option_val(token);
                continue;
            } else {
                continue;
            }
        }

    }
    return found;
}

bool CCodeGen::get_interface(HlsInterface &interface, HlsInterface &gmem) {

    auto options = prepare_flag(flags_);
    if (!(options.str().empty())) {

        std::string token;
        gmem = HlsInterface::None;
        bool set_interface = false;

        while (std::getline(options, token, ',')) {
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
std::string CCodeGen::convert(const Type* type, bool templated) {
    if (auto res = types_.lookup(type))
           return *res;

    StringStream s;
    std::string name;

    if (type == world().unit() || type->isa<MemType>() || type->isa<FrameType>())
        s << "void";
    else if (auto primtype = type->isa<PrimType>()) {
        switch (primtype->primtype_tag()) {
            case PrimType_bool:                     s << "bool";                     break;
            case PrimType_ps8:  case PrimType_qs8:  s <<   "i8";                     break;
            case PrimType_pu8:  case PrimType_qu8:  s <<   "u8";                     break;
            case PrimType_ps16: case PrimType_qs16: s <<  "i16";                     break;
            case PrimType_pu16: case PrimType_qu16: s <<  "u16";                     break;
            case PrimType_ps32: case PrimType_qs32: s <<  "i32";                     break;
            case PrimType_pu32: case PrimType_qu32: s <<  "u32";                     break;
            case PrimType_ps64: case PrimType_qs64: s <<  "i64";                     break;
            case PrimType_pu64: case PrimType_qu64: s <<  "u64";                     break;
            case PrimType_pf16: case PrimType_qf16: s <<  "f16";  use_fp_16_ = true; break;
            case PrimType_pf32: case PrimType_qf32: s <<  "f32";                     break;
            case PrimType_pf64: case PrimType_qf64: s <<  "f64";  use_fp_64_ = true; break;
            default: THORIN_UNREACHABLE;
        }

        if (templated) {
            StringStream temp;
            temp << "<" << s.str();
            swap(s, temp);
            s << ">";
        }
        if (primtype->is_vector())
            s << primtype->length();
    } else if (auto array = type->isa<IndefiniteArrayType>()) {
        return types_[type] = convert(array->elem_type()); // IndefiniteArrayType always occurs within a pointer
    } else if (type->isa<FnType>()) {
        assert(false && "todo");
    } else if (auto ptr = type->isa<PtrType>()) {
        //TODO: add all corner cases
        // CUDA supports generic pointers, so there is no need to annotate them (moreover, annotating them triggers a bug in NVCC 11)
        if (templated) { s.fmt("<{}{}>*","", convert(ptr->pointee())); }
        else
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
        types_[struct_type] = name = struct_type->name().str();
        if ((lang_ == Lang::OpenCL || lang_ == Lang::HLS) && is_channel_type(struct_type))
            use_channels_ = true;
        if (lang_ == Lang::OpenCL && use_channels_) {
            s.fmt("typedef {} {}_{};\n", convert(struct_type->op(0)), name, struct_type->gid());
            name = (struct_type->name().str() + "_" + std::to_string(type->gid()));
        } else if (is_channel_type(struct_type) && lang_ == Lang::HLS) {
            s.fmt("typedef {} {}_{};\n", convert(struct_type->op(0)), name, struct_type->gid());
            name = ("hls::stream<" + name + "_" + std::to_string(type->gid()) + ">");
        } else if (is_channel_type(struct_type) && lang_ == Lang::CGRA) {
            s.fmt("typedef {} {}_{};\n", convert(struct_type->op(0)), name, struct_type->gid());
            graph_stream_.fmt("typedef {} {}_{};\n\n", convert(struct_type->op(0)), name, struct_type->gid());
            //name = ("<" + name + "_" + std::to_string(type->gid()) + ">");
            name = ( name + "_" + std::to_string(type->gid()));
        } else if (lang_ == Lang::CGRA && is_mmul_type(struct_type)) {
            s.fmt("//AIE mmul {} obj\n", convert(struct_type) );
        } else {
            s.fmt("typedef struct {{\t\n");
            s.rangei(struct_type->ops(), "\n", [&] (size_t i) { s.fmt("{} {};", convert(struct_type->op(i)), struct_type->op_name(i)); });
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
        case Lang::OpenCL: return use_channels_ ? "PIPE " : "__constant ";
        case Lang::CGRA:   return "alignas(aie::vector_decl_align) ";
    }
}

static inline bool is_passed_via_buffer(const Param* param) {
    return param->type()->isa<DefiniteArrayType>()
        || (param->type()->isa<StructType>() && (!is_channel_type(param->type()->isa<StructType>())))
        || param->type()->isa<TupleType>();
}

static inline bool is_passed_via_global_mem(const Param* param) {
    return (param->type()->isa<PtrType>() && param->type()->as<PtrType>()->pointee()->isa<ArrayType>());
}

HlsInterface interface, gmem_config;
//auto [interface_status, hls_top_scope, cgra_graph_scope] = std::make_tuple(false, false, false);
bool interface_status = false;

std::string CCodeGen::prefix_type(const Param* param) {
    auto cont = param->as<Param>()->continuation();

    std::string interface;
    switch (cont->get_interface()) {
        case Interface::Stream: case Interface::Free_running: case Interface::Cascade:
            interface = "stream"; //TODO: consider "cascade" which streaming only between cgra kernels
            break;
        case Interface::Window:
            interface = "window";
            break;
        case Interface::Buffer:
            interface = "buffer"; //TODO: for buffer we need to add adf:extents to define the BUFFER_SIZE, also we need iterator APIs
            break;
        case Interface::Circular_buffer:
            interface = "circular_buffer";
            break;
        case Interface::None:
            if (!cont->is_cgra_graph())
                world().WLOG("Interface is not known, using stream by default");
            interface = "stream";
            break;
        default:
            world().WLOG("Interface is not supported, using stream by default");
            interface = "stream";
            break;
    }

    std::string prefix;
    if (lang_ == Lang::CGRA && (is_channel_type(param->type()) || is_passed_via_global_mem(param))) {
        if(auto config = get_config(cont); config) {
            assert(config->isa<CGRAKernelConfig>() && "CGRAKernelConfig expected");

            prefix = cont->is_cgra_graph() ? "adf::" : "";
            auto param_mode = config->as<CGRAKernelConfig>()->param_mode(param);
            if (param_mode == ChannelMode::Undef)
                world().WLOG("Direction of {} is undefined", param->unique_name());
            else if (param_mode == ChannelMode::Read)
                prefix += "input_";
            else if (param_mode == ChannelMode::Write)
                prefix += "output_";
            else if (param_mode == ChannelMode::ReadWrite)
                prefix += "inout_";


            std::string type_name;
            if (cont->is_cgra_graph()) {
                std::string io_type;
                //type_name = (" <" + convert(param->type()->as<PtrType>()->pointee()) + ">");
                if (is_channel_type(param->type()))
                    io_type = "plio";
                else if (is_passed_via_global_mem(param)) {
                    io_type = "gmio";
                } else {
                    assert(false && "TODO: CGRA runtime parameter");
                };
                prefix += io_type;
            } else {
                type_name = convert(param->type(), true); //template type_name
                prefix += interface;
            }

            return prefix + type_name;
        }
    }
    // any other languages
    return prefix + convert(param->type());
}


void CCodeGen::graph_ctor_gen (const Continuations& graph_conts) {

    using ModeCounters = std::array<int32_t, to_underlying(ChannelMode::Count)>;
    using Cont2Index = ContinuationMap<ModeCounters>;
    using Dependence = std::pair<const Def*, const Def*>;

    if (!get_cgra_options())
        world().WLOG("No CGRA options are provided, using default options");

    auto get_param_mode = [&] (auto index, Continuation* cont) {
        if(auto config = get_config(cont)) {
            assert(config->isa<CGRAKernelConfig>() && "CGRAKernelConfig expected");
            auto mode = config->as<CGRAKernelConfig>()->param_mode(cont->param(index));
            return mode;
        }
        assert(false && "kernel has no config");
    };


    auto get_runtime_ratio = [&] (Continuation* cont) {
        if(auto config = get_config(cont)) {
            assert(config->isa<CGRAKernelConfig>() && "CGRAKernelConfig expected");
            return config->as<CGRAKernelConfig>()->runtime_ratio();
        }
        assert(false && "kernel has no config");
    };


    auto get_location = [&] (Continuation* cont) {
        if(auto config = get_config(cont)) {
            assert(config->isa<CGRAKernelConfig>() && "CGRAKernelConfig expected");
            return config->as<CGRAKernelConfig>()->location();
        }
        assert(false && "kernel has no config");
    };

    // Plio params are those that on one side ther are connected to the outside of the cgra_graph
    // they are defined in the cgra_graph continuation
    auto is_plio = [&] (const Def* def) {
        return def->as<Param>()->continuation()->is_cgra_graph();
    };

    auto node_name = [&] (const Def* def) {
        return def->as<Param>()->continuation()->is_cgra_graph() ? def->unique_name() : "k" + def->name();
    };

    auto get_node_names = [&] (const Dependence dependence) {
        auto start_node = node_name(dependence.first);
        auto end_node   = node_name(dependence.second);
        return std::make_pair(start_node, end_node);
    };

    auto get_node_indices = [&] (const Dependence dependence, Cont2Index& cont2index) {
        auto from_indices = cont2index[dependence.first->as<Param>()->continuation()];
        auto from = from_indices[to_underlying(ChannelMode::Write)];
        auto to_indices = cont2index[dependence.second->as<Param>()->continuation()];
        auto to = to_indices[to_underlying(ChannelMode::Read)];
        return std::make_pair(from, to);
    };

    auto get_io_mode_index = [] (const ModeCounters& mode_counters, ChannelMode mode) {
        return mode_counters[to_underlying(mode)];
    };

    auto get_edge_label = [&] (const Dependence& dependence) {
        return dependence.first->as<Param>()->continuation()->name() + "_"
            + dependence.second->as<Param>()->continuation()->name();
    };

    auto krl_node_name = [&] (auto& kernel) {
        return "k" + kernel->name();
    };

    auto get_direction_prefix = [&] (ChannelMode mode) {
        std::string direction;
        switch (mode) {
            case ChannelMode::Write:
                direction = "output";
                break;
            case ChannelMode::Read:
                direction = "input";
                break;
            case ChannelMode::ReadWrite:
                direction = "inout";
                break;
            default:
                world().WLOG("Direction of the parameter is undefined. Direct GMem access is not fully supported yet");
                break;
        }
        return direction;
    };

    auto get_connection_method = [&] (const Dependence& dependence) {

        auto connection_method = [&] (const Continuation* cont) {
            std::string s;
            switch (cont->get_interface()) {
                case Interface::Stream:
                    s = "<adf::stream>";
                    break;
                case Interface::Cascade:
                    s =  "<adf::cascade>";
                    break;
                case Interface::Window: {
                    auto window_size = cont->get_buf_size();
                    auto type = is_plio(dependence.first) ? dependence.second->type() : dependence.first->type();
                    s = "<adf::window<" + std::to_string(window_size) + " * sizeof(" + convert(type->as<PtrType>()->pointee()) + ")>" + ">";
                    }
                    break;
                case Interface::Free_running:
                    s = "<>";
                    break;
                case Interface::Circular_buffer: case Interface::Buffer:
                    s = " ";
                    break;
                case Interface::None: {
                    world().WLOG("Connection method could not be determined, using stream by default");
                    s = "<adf::stream>";
                    }
                    break;
                default:
                    assert(false && "No connection method is defined for the this interface");
            }
            return s;
        };


        static auto get_continuation = [&] (const Def* def) {
            return def->as<Param>()->continuation();
        };

        static auto get_interface = [&] (const Def* def) {
            return get_continuation(def)->get_interface();
        };

        auto is_buffer_interface = [] (Interface interface) {
            return interface == Interface::Buffer || interface == Interface::Circular_buffer;
        };


        // unless the interface is buffer, kernels with different interfaces are not supported.
        // a more proper solution would be param-wise interface definition which is not supported yet.
        auto [from, to] = dependence;
        auto [from_intf, to_intf] = std::make_pair(get_interface(from), get_interface(to));
        std::string method = "METHOD";
        if ((from_intf == to_intf) && from_intf != Interface::Cascade) {
            method = connection_method(get_continuation(from));
        } else if (is_plio(from) || is_plio(to)) {
            if (from_intf == Interface::None) {
                assert(to_intf != Interface::None && "CGRA PLIOs cannot directly connect to each other");
                method = connection_method(get_continuation(to));
            } else {
                method = connection_method(get_continuation(from));
            }

            if (from_intf == Interface::Cascade || to_intf == Interface::Cascade) {
                // special case for cascade. using stream instead of cascade
                method = "<adf::stream>";
            }
        } else if (is_buffer_interface(from_intf) || !is_buffer_interface(to_intf)) {
            // if intfs are different not plio and at least one of them is a buffer
            world().WLOG("TODO: buffer is not supported yet");
        } else { // TODO: cascade
            world().ELOG("Interface mismatch");
        }

        return method;
    };
        //TODO:
        // we can ovverride interface attr using kernel config so that we can kernels with different interfaces on the same code
        // for the moment for cascade we can check if the conts are not from cgra_graph then we can change the stream to cascade

  //  };

    auto set_mode_counters = [&] (const auto& kernel, const auto& mode, auto& mode_counters, Cont2Index& cont2index) {
        if (!cont2index.contains(kernel)) {
            mode_counters[to_underlying(mode)] = 0;
            cont2index.emplace(kernel, mode_counters);
        } else {
            auto mode_counters = cont2index[kernel];
            mode_counters[to_underlying(mode)]++;
            cont2index[kernel] = mode_counters;
         }
    };

    auto set_io_mode_counters = [] (const auto& mode, auto& mode_counters) {
            mode_counters[to_underlying(mode)]++;
    };


    auto bit_width = [&] (const Type* type) {
        StringStream s;
        assert ((type != world().unit() || !(type->isa<MemType>()) || !(type->isa<FrameType>())) && "Only primary types allowed.");
        if (auto primtype = type->isa<PrimType>()) {
            switch (primtype->primtype_tag()) {
                case PrimType_bool:                                                             s << "1" ;  break;
                case PrimType_ps8 : case PrimType_qs8 : case PrimType_pu8 : case PrimType_qu8 : s << "8" ;  break;
                case PrimType_ps16: case PrimType_qs16: case PrimType_pu16: case PrimType_qu16: s << "16";  break;
                case PrimType_ps32: case PrimType_qs32: case PrimType_pu32: case PrimType_qu32: s << "32";  break;
                case PrimType_ps64: case PrimType_qs64: case PrimType_pu64: case PrimType_qu64: s << "64";  break;
                case PrimType_pf16: case PrimType_qf16:                                         s << "16";  break;
                case PrimType_pf32: case PrimType_qf32:                                         s << "32";  break;
                case PrimType_pf64: case PrimType_qf64:                                         s << "64";  break;
                default: THORIN_UNREACHABLE;
            }
        } else {
            world().ELOG("Type is not supported");
        }
        return s.str();
    };

    StringStream node_impls_;
    StringStream edge_impls_;
    StringStream configs_;

    node_impls_.indent(2);
    edge_impls_.indent(2);
    configs_.indent(2);

    DefSet visited_defs; // edges are emitted only if none of the corresponding nodes (dependence) has already been visited
    DefSet visited_conts;
    const auto counter_init = -1;

    for (auto cont : graph_conts) {
        if (cont->intrinsic() == Intrinsic::EndScope) continue;

        auto entry = cont->is_cgra_graph() ? cont: nullptr;

        Dependence dependence; //<From, To>
        auto adjust_operands = [&] (Dependence& dependence) {
            std::swap(dependence.first, dependence.second);
        };

        // emitting the connections between cgra_graph top and all other kernels
        if (cont->isa_nom<Continuation>() && (!cont->body()->empty()) && cont->has_body()) {

            if (cont == entry) {

                auto simulated_data = options.sim_data;
                ModeCounters io_counters; // only for naming simulation data files
                io_counters.fill(0);

                for (auto param : cont->params()) {
                    if (is_concrete(param) || is_channel_type(param->type())) {
                        param->dump();
                        if (visited_defs.empty() || visited_defs.count(param) == 0) {
                            // Note that nodes of the same edge have the same names in IR (args of Fns) but different names in C
                            visited_defs.emplace(param);

                            auto mode = get_param_mode(param->index(), cont);
                            set_io_mode_counters(mode, io_counters);
                            auto direc_prefix = get_direction_prefix(mode);
                            auto op_type = param->type();// TODO: Dummy value for GMem direct access.
                            if (auto ptr_type = param->type()->isa<PtrType>()) { // if not then it is a runtime parameter
                                if(auto struct_type = ptr_type->pointee()->isa<StructType>())
                                    op_type = struct_type->op(0);
                            }

                            auto io_index = get_io_mode_index(io_counters, mode);

                            node_impls_.fmt("{} = adf::{}_plio::create(adf::plio_{}_bits{});\n",
                                    param->unique_name(), direc_prefix, bit_width(op_type),
                                    simulated_data ? (", \"" + direc_prefix + "_" + std::to_string(io_index) + ".txt\"") : (""));

                        }

                        dependence.first = param;
                        for (size_t i = 0; i < graph_conts.size() - 1; ++i) {
                            Cont2Index cont2index;
                            ModeCounters cont_mode_counters, mode_counters;
                            // check if it works in the last version of aie compiler otherwise we need to remove any array index for plio/gmem ports and use them only if they are literally arrays of ports
                            cont_mode_counters.fill(0);
                            // mode indices of cgra_graph cont can actually resemble indices for arrays of plio/gmem ports
                            // but for now we assume that each param (port) is a scalar type (single port), therefore, we assign
                            // a zero index to each param, like param1.[0]
                            //auto mode = get_param_mode(kernel_param->index(), kernel);
                            //cont_mode_counters[to_underlying(mode)] = 0;
                            cont2index[cont] = cont_mode_counters;
                            //ModeCounters mode_counters;
                            mode_counters.fill(counter_init);
                            for (size_t arg_index = 0; const auto& arg : graph_conts[i]->body()->args()) {
                                if (is_concrete(arg) || is_channel_type(arg->type())) {
                                    auto graph_callee = graph_conts[i]->body()->callee();
                                    if (auto kernel = graph_callee->isa_nom<Continuation>()) {
                                        assert(kernel->name() == graph_callee->name());
                                        auto kernel_param = kernel->param(arg_index);
                                        //auto mode = get_param_mode(arg_index, kernel);
                                        auto mode = get_param_mode(kernel_param->index(), kernel);

                                        // The problem is some params are counted several times
                                        // After each param counters should be reset
                                        // or use a visitig def list to filter those params that have already been visited
                                        set_mode_counters(kernel, mode, mode_counters, cont2index);

                                        if (param == arg) { // edge found
                                            dependence.second = kernel_param;

                                            if (mode == ChannelMode::Write) {
                                                adjust_operands(dependence);
                                            }

                                            if (visited_conts.empty() || visited_conts.count(graph_callee) == 0) {
                                                visited_conts.emplace(graph_callee);
                                                node_impls_.fmt("{} = adf::kernel::create({});\n", krl_node_name(graph_callee), graph_callee->name());
                                            }
                                            //note: at the moment the interface type is an attribute of the continuation not the param
                                            // therefore, we cannot have a contiunation having different interfaces on their params.
                                            // we can overcome this by adding an attribute to the param class

                                            auto [start_node, end_node] = get_node_names(dependence);
                                            auto [start_index, end_index] = get_node_indices(dependence, cont2index);
                                            auto edge_label = get_edge_label(dependence);
                                            auto method = get_connection_method(dependence);
                                            edge_impls_.fmt("adf::connect{} {}({}.out[{}], {}.in[{}]);\n", method, edge_label, start_node, start_index, end_node, end_index);

                                        }

                                    }


                                }

                                arg_index++;
                            }

                        }

                    }
                }
            }


            // emiting all other connections (those among callees without cgra_graph continuation)
            Cont2Index cur_cont2index;
            ModeCounters cur_mode_counters;
            cur_mode_counters.fill(counter_init);
            for (size_t cur_arg_index = 0; const auto& cur_arg : cont->body()->args()) {
                if (is_concrete(cur_arg)) {
                    if (visited_defs.empty() || visited_defs.count(cur_arg) == 0) {
                        // Note that nodes of the same edge have the same names in IR (args of Fns) but different names in C
                        visited_defs.emplace(cur_arg);
                        auto graph_callee = cont->body()->callee();
                        if (visited_conts.empty() || visited_conts.count(graph_callee) == 0) {
                            visited_conts.emplace(graph_callee);
                            node_impls_.fmt("{} = adf::kernel::create({});\n", krl_node_name(graph_callee), graph_callee->name());
                        }

                        if (auto kernel = graph_callee->isa_nom<Continuation>()) {
                            assert(kernel->name() == graph_callee->name());
                            auto kernel_param = kernel->param(cur_arg_index);
                            auto mode = get_param_mode(kernel_param->index(), kernel);
                            set_mode_counters(kernel, mode, cur_mode_counters, cur_cont2index);
                            dependence.first = kernel_param;
                        }

                        Cont2Index cont2index;
                        ModeCounters mode_counters;
                        mode_counters.fill(counter_init);
                        for (size_t i = 0; i < graph_conts.size() - 1 ; ++i) {
                            auto cur_app = cont->body();
                            auto app = graph_conts[i]->body();
                            for (size_t arg_index = 0; const auto& arg : app->args()) {

                                if (cur_app->callee() == app->callee())
                                    continue;

                                if (is_concrete(arg) || is_channel_type(arg->type())) {
                                    auto graph_callee = graph_conts[i]->body()->callee();
                                    if (auto kernel = graph_callee->isa_nom<Continuation>()) {
                                        assert(kernel->name() == graph_callee->name());
                                        auto kernel_param = kernel->param(arg_index);
                                        //auto mode = get_param_mode(arg_index, kernel);
                                        auto mode = get_param_mode(kernel_param->index(), kernel);
                                        set_mode_counters(kernel, mode, mode_counters, cont2index);
                                        if (cur_arg == arg) { // edge found
                                            dependence.second = kernel_param;
                                            if (mode == ChannelMode::Write) {
                                                adjust_operands(dependence);
                                            }
                                            // We can safely merge the two maps because there won't be any similar keys(continuations)
                                            cont2index.insert(cur_cont2index.begin(), cur_cont2index.end());
                                            auto [start_index, end_index] = get_node_indices(dependence, cont2index);
                                            auto [start_node, end_node] = get_node_names(dependence);
                                            auto edge_label = get_edge_label(dependence);
                                            auto method = get_connection_method(dependence);
                                            edge_impls_.fmt("adf::connect{} {}({}.out[{}], {}.in[{}]);\n",method , edge_label, start_node, start_index, end_node, end_index);
                                        }
                                    }
                                }
                                arg_index++;
                            }
                        }
                    }
                }
                cur_arg_index++;
            }

            auto source_ext = thorin::c::CodeGen(world(), kernel_config_, lang_, debug_, flags_).file_ext();
            if (cont->has_body() && cont->body()->callee()->isa_nom<Continuation>()) {
                // TODO: return is_a<cont> in if and use it in the body
                // TODO: soure(kernels) = addr
                auto callee = cont->body()->callee();
                configs_.fmt( "adf::runtime<ratio>({}) = {};\n", krl_node_name(callee), get_runtime_ratio(callee->as_nom<Continuation>()));
                auto [loc_x, loc_y] = get_location(callee->as_nom<Continuation>());
                if (loc_x >= 0 && loc_y >= 0)
                    configs_.fmt( "adf::location<adf::kernel>({}) = adf::tile({}, {});\n", krl_node_name(callee), loc_x, loc_y);

                configs_.fmt( "adf::source({}) = \"{}{}\";\n", krl_node_name(callee), world().name(), source_ext);
            }
        }
    }

    graph_ctor_ <<"// Nodes\n\t\t" << node_impls_.str() << "// Edges\n\t\t" << edge_impls_.str() << "// Constrains and Configurations\n\t\t" << configs_.str();

    return;
}


/*
 * emit
 */

void CCodeGen::emit_module() {

    if (lang_ == Lang::CGRA) {
        graph_stream_.fmt("#include <adf.h>\n"
                "#include <aie_api/aie_adf.hpp>\n"
                "#include <aie_api/utils.hpp>\n"
                "#include <iostream>\n"
                "#include <string>\n"
                "#include <fstream>\n");

        graph_stream_.fmt("\n"
                        "typedef   int8_t  i8;\n"
                        "typedef  uint8_t  u8;\n"
                        "typedef  int16_t i16;\n"
                        "typedef uint16_t u16;\n"
                        "typedef  int32_t i32;\n"
                        "typedef uint32_t u32;\n"
                        "typedef  int64_t i64;\n" // only for scalar cores
                        "typedef uint64_t u64;\n" // only for scalar cores
                        "typedef    float f32;\n"
                        "typedef   double f64;\n"
                        "\n");
    }

    Continuation* top_module = nullptr;
    interface_status = get_interface(interface, gmem_config);

    Scope::for_each(world(), [&] (const Scope& scope) {
            auto entry = scope.entry();

            if (entry->is_hls_top() || entry->is_cgra_graph()) {
                top_module = entry;

                if (entry->is_cgra_graph()) {
                    graph_ctor_gen(schedule(scope));
                    // Trying out graph gen
                    for (auto graph_conts = schedule(scope); auto cont : graph_conts) {

                    if (cont->intrinsic() == Intrinsic::EndScope) continue;

                    if (cont == entry) {
                        std::cout << "___CGRA___\n";
                        for (auto param : entry->params()) {
                            if (is_concrete(param) || is_channel_type(param->type())) {
                                param->dump();
                            }
                        }
                        std::cout << "___CGRA_ END__\n";
                    }

                    if (cont->isa_nom<Continuation>() && !cont->body()->empty()) {
                        std::cout <<"UP callee "; cont->body()->callee()->dump();
                        if (cont->has_body()) {
                            if (auto counted = cont->body()->callee()->isa_nom<Continuation>()){
                                std::cout <<"UP CONT ";
                                counted->dump();
                                //params to find direction using arg index
                                for (auto param : counted->params()) {
                                    if (is_concrete(param)) {
                                        std::cout << "graph_param "; param->dump();
                                    }
                                }
                            }
                        }

                        //args to find edges of the graph
                        for (size_t arg_index = 0; auto arg : cont->body()->args()) {
                            if (is_concrete(arg) || is_channel_type(arg->type())) {
                                arg->dump();
                            }
                            arg_index++;
                        }
                    }
                    }
            }
        } else
            emit_scope(scope);
    });

    if (top_module) {
        if (top_module->is_hls_top()) {
            top_scope.hls    = true;
            //cgra_graph_scope = false;
        } else if (top_module->is_cgra_graph())
            top_scope.cgra_graph = true;
            //hls_top_scope    = false;
        emit_scope(Scope(top_module));
    }


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

        stream_.fmt(    "\n"
                        "typedef   char  i8;\n"
                        "typedef  uchar  u8;\n"
                        "typedef  short i16;\n"
                        "typedef ushort u16;\n"
                        "typedef    int i32;\n"
                        "typedef   uint u32;\n"
                        "typedef   long i64;\n"
                        "typedef  ulong u64;\n");
        if (use_fp_16_)
            stream_.fmt("typedef   half f16;\n");
        stream_.fmt(    "typedef  float f32;\n");
        if (use_fp_64_)
            stream_.fmt("typedef double f64;\n");
    }

    stream_.endl();

    if (lang_ == Lang::C99) {
        stream_.fmt(    "#include <stdbool.h>\n"    // for the 'bool' type
                        "#include <stdint.h>\n");   // for the fixed-width integer types
        if (use_align_of_)
            stream_.fmt("#include <stdalign.h>\n"); // for 'alignof'
        if (use_memcpy_)
            stream_.fmt("#include <string.h>\n");   // for 'memcpy'
        if (use_malloc_)
            stream_.fmt("#include <stdlib.h>\n");   // for 'malloc'
        if (use_math_)
            stream_.fmt("#include <math.h>\n");     // for 'cos'/'sin'/...
    }

    if (lang_ == Lang::HLS) {
        stream_.fmt("#include <hls_stream.h>\n"
                    "#include <hls_math.h>\n");
        if (use_fp_16_)
            stream_.fmt("#include <hls_half.h>\n");
    }

    if (lang_ == Lang::CGRA) {
        stream_.fmt("#include <adf.h>\n"
                    "#include <aie_api/aie_adf.hpp>\n"
                    "#include <aie_api/utils.hpp>\n"
                    "#include <aie_api/operators.hpp>\n"
                    "using namespace aie::operators;\n");
    }

    if (lang_ == Lang::C99 || lang_ == Lang::HLS || lang_ == Lang::CGRA) {
        stream_.fmt(    "\n"
                        "typedef   int8_t  i8;\n"
                        "typedef  uint8_t  u8;\n"
                        "typedef  int16_t i16;\n"
                        "typedef uint16_t u16;\n"
                        "typedef  int32_t i32;\n"
                        "typedef uint32_t u32;\n"
                        "typedef  {} i64;\n"
                        "typedef  {} u64;\n"
                        "typedef    float f32;\n"
                        "typedef   double f64;\n"
                        "\n", (is_cgra_vector_kernel()) ? "acc64" : "int64_t",
                              (is_cgra_vector_kernel()) ? "acc80" : "uint64_t");

         if (use_fp_16_ && lang_ == Lang::HLS)
            stream_.fmt("typedef     half f16;\n");
    }

    if (lang_ == Lang::CUDA) {
        if (use_fp_16_)
            stream_.fmt("#include <cuda_fp16.h>\n\n");
        stream_.fmt(    "typedef               char  i8;\n"
                        "typedef      unsigned char  u8;\n"
                        "typedef              short i16;\n"
                        "typedef     unsigned short u16;\n"
                        "typedef                int i32;\n"
                        "typedef       unsigned int u32;\n"
                        "typedef          long long i64;\n"
                        "typedef unsigned long long u64;\n"
                        "\n");
        if (use_fp_16_)
            stream_.fmt("#if __CUDACC_VER_MAJOR__ <= 8\n"
                        "typedef               half f16;\n"
                        "#else\n"
                        "typedef         __half_raw f16;\n"
                        "#endif\n");
        stream_.fmt(    "typedef              float f32;\n"
                        "typedef             double f64;\n"
                        "\n");
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
    //graph_stream_ << "TEST";

    if (lang_ == Lang::CUDA || lang_ == Lang::HLS)
        stream_.fmt("}} /* extern \"C\" */\n");
}


static inline const Type* ret_type(const FnType* fn_type) {
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



//A function that define a table structure for cgra device intrinsic holding template parameters' information
//specialize cgra device APIs
// A device API that has any kinds of template parameters should be configured in this table

std::unique_ptr<ApiConfig> CCodeGen::special_device_api(const Continuation* cont) {
    // how many template parameters are there in the real api?
    // and which indexes are type parameters?
    // and from which arg or param those type parameters are coming from?
    //using Cont2ApiConfig = ContinuationMap<std::vector<std::variant<const Def*, const Type*>>>; // used for CGRA API generation
    //using ContName2ApiConfig = std::map<std::string, ApiConfig>; // used for CGRA API generation
    auto body = cont->body();
    auto callee = body->callee();
    auto name = callee->name();
    std::cout << "DEBUG: Callee name: " << name << "\n";

    auto api_config = std::make_unique<ApiConfig>();

    std::vector<std::pair<size_t, const Type*>> empty_vect{};
    auto no_type = TempTypeParams{ empty_vect };
    auto ret_type = thorin::c::ret_type(callee->as_nom<Continuation>()->type());
    auto type_of_arg = [&] (const size_t index) -> const Type* { return cont->body()->arg(index)->type();};

    auto sliding_mul_config = [&] () -> ApiConfig {
        const auto temp_params_size = 6;
        TempTypeParams temp_type_params = { { 4, type_of_arg(5) }, { 5, type_of_arg(7) } };
        ApiConfig api_config {temp_params_size, temp_type_params};
        return {api_config};
    };

    auto sliding_mac_config = [&] () -> ApiConfig {
        const auto temp_params_size = 6;
        TempTypeParams temp_type_params = { { 4, type_of_arg(6) }, { 5, type_of_arg(8) } };
        ApiConfig api_config {temp_params_size, temp_type_params};
        return {api_config};
    };

    // 1) AIE APIs with tepmlate type and non-type parameters
    if      (name == "aie::broadcast")       *api_config = {2, TempTypeParams{ { 0, ret_type } } };// <type,const>
    else if (name == "aie::zeros")           *api_config = {2, TempTypeParams{ { 0, ret_type } } };
    else if (name == "aie::vector_cast")     *api_config = {1, TempTypeParams{ { 0, ret_type } } };
    else if (name == "aie::mmul")            *api_config = {5, TempTypeParams{ { 3, type_of_arg(1) }, { 4, type_of_arg(2) } } };
    else if (name == "aie::sliding_mul_ops") *api_config = {7, TempTypeParams{ { 5, type_of_arg(3) }, { 6, type_of_arg(4) } } };

    // 2) AIE APIs without any template type parameters
    else if (name == "aie::load_v")          *api_config = {1, no_type };
    else if (name == "aie::store_v")         *api_config = {1, no_type };
    else if (name == "window_readincr_v")    *api_config = {1, no_type };
    else if (name == "readincr_v")           *api_config = {1, no_type };
    else if (name == "writeincr_v")          *api_config = {1, no_type };
    else if (name == "aie::accumulate")      *api_config = {1, no_type };
    else if (name == "aie::sliding_mul")             *api_config = {2, no_type };
    else if (name == "aie::sliding_mac")             *api_config = {2, no_type };
    else if (name == "aie::sliding_mul_sym")         *api_config = {2, no_type };
    else if (name == "aie::sliding_mac_sym")         *api_config = {2, no_type };
    else if (name == "aie::sliding_mul_antisym")     *api_config = {2, no_type };
    else if (name == "aie::sliding_mac_antisym")     *api_config = {2, no_type };
    else if (name == "aie::sliding_mul_sym_uct")     *api_config = {2, no_type };
    else if (name == "aie::sliding_mac_sym_uct")     *api_config = {2, no_type };
    else if (name == "aie::sliding_mul_antisym_uct") *api_config = {2, no_type };
    else if (name == "aie::sliding_mac_antisym_uct") *api_config = {2, no_type };
    // 3) AIE class template APIs (no template type parameters but have a composite type (like struct) as fn parameter or they are member of aie::vector class)
    else if (name == "aie::mmul::mul")           *api_config = {0, type_of_arg(1) };
    else if (name == "aie::mmul::mac")           *api_config = {0, type_of_arg(1) };
    else if (name == "aie::vector::insert")      *api_config = {0, type_of_arg(1) };
    else if (name == "aie::vector::extract")     *api_config = {1, type_of_arg(2) };
    else if (name == "aie::vector::extract<32>") *api_config = {0, type_of_arg(1) };
    // 4) AIE struct template static APIs
    else if (name == "aie::sliding_mul_xy_ops::mul")        *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_xy_ops::mac")        *api_config = sliding_mac_config();
    else if (name == "aie::sliding_mul_xy_ops::mul_common") *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_xy_ops::negmul")     *api_config = sliding_mul_config();

    else if (name == "aie::sliding_mul_x_ops::mul")        *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_x_ops::mac")        *api_config = sliding_mac_config();
    else if (name == "aie::sliding_mul_x_ops::mul_common") *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_x_ops::negmul")     *api_config = sliding_mul_config();

    else if (name == "aie::sliding_mul_y_ops::mul")        *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_y_ops::mac")        *api_config = sliding_mac_config();
    else if (name == "aie::sliding_mul_y_ops::mul_common") *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_y_ops::negmul")     *api_config = sliding_mul_config();

    else if (name == "aie::sliding_mul_sym_x_ops::mac_antisym") *api_config = sliding_mac_config();
    else if (name == "aie::sliding_mul_sym_x_ops::mac_sym")     *api_config = sliding_mac_config();
    else if (name == "aie::sliding_mul_sym_x_ops::mul_antisyn") *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_sym_x_ops::mul_common")  *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_sym_x_ops::mul_sym")     *api_config = sliding_mul_config();

    else if (name == "aie::sliding_mul_sym_y_ops::mac_antisym") *api_config = sliding_mac_config();
    else if (name == "aie::sliding_mul_sym_y_ops::mac_sym")     *api_config = sliding_mac_config();
    else if (name == "aie::sliding_mul_sym_y_ops::mul_antisyn") *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_sym_y_ops::mul_common")  *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_sym_y_ops::mul_sym")     *api_config = sliding_mul_config();

    else if (name == "aie::sliding_mul_sym_xy_ops::mac_antisym") *api_config = sliding_mac_config();
    else if (name == "aie::sliding_mul_sym_xy_ops::mac_sym")     *api_config = sliding_mac_config();
    else if (name == "aie::sliding_mul_sym_xy_ops::mul_antisyn") *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_sym_xy_ops::mul_common")  *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_sym_xy_ops::mul_sym")     *api_config = sliding_mul_config();

    else if (name == "aie::sliding_mul_sym_uct_ops::mac_antisym_uct") *api_config = sliding_mac_config();
    else if (name == "aie::sliding_mul_sym_uct_ops::mac_sym_uct")     *api_config = sliding_mac_config();
    else if (name == "aie::sliding_mul_sym_uct_ops::mul_antisyn_uct") *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_sym_uct_ops::mul_common_uct")  *api_config = sliding_mul_config();
    else if (name == "aie::sliding_mul_sym_uct_ops::mul_sym_uct")     *api_config = sliding_mul_config();

    // 5) AIE commands
    else if (name == "aie::set_saturation(aie::saturation_mode::saturate)")  *api_config = {0, no_type };
    else if (name == "aie::set_saturation(aie::saturation_mode::none)")      *api_config = {0, no_type };
    else if (name == "aie::set_saturation(aie::saturation_mode::truncate)")  *api_config = {0, no_type };
    else if (name == "aie::set_saturation(aie::saturation_mode::symmetric)") *api_config = {0, no_type };
    else if (name == "aie::set_rounding(aie::rounding_mode::ceil)")          *api_config = {0, no_type };
    else if (name == "aie::set_rounding(aie::rounding_mode::floor)")         *api_config = {0, no_type };
    //TODO: fft, and new way to implement aie::mmul
    // make aie::mmul<M, K, N, T, T>() like a struct type and pass MKN as members of the struct but make different types of struct
    // in code make an object of this new type. (like a static global)
    // make a new device intrinsic like channel that gets this new object and two matrices. but generates the code like obj.mul(block_a, block_b);
    // then we need to do the same for aie::mmul::mac to get something like obj.mac(block_a, block_b);
    // for the current purpose it is only useful if we want to do convolution via mmul.
    // Also we need to implment a specific fn for mmul cast to vector: auto test_ = block_c.template to_vector<T>();
    //TODO: Alos aie::sliding_mul_xy_ops as struct type? basically all class template device APIs
    else return nullptr;

    return api_config;
}

auto CCodeGen::is_accum_type(const Type* type) {
    assert((lang_ == Lang::CGRA) && "Only CGRA is supported");
    auto is_accum = false;
    if (vector_size_ > 1) {
        if (auto primtype = type->isa<PrimType>()) {
            switch (primtype->primtype_tag()) {
                case PrimType_ps64: case PrimType_qs64:
                case PrimType_pu64: case PrimType_qu64:
                    is_accum = true;
                    break;
                default:
                    break;
            }
        }
    }
    return is_accum;
}

auto CCodeGen::is_mask_type(const Type* type) {
    assert((lang_ == Lang::CGRA) && "Only CGRA is supported");
    auto is_mask = false;
    if (vector_size_ > 1) {
        if (auto primtype = type->isa<PrimType>()) {
            auto type_tag = primtype->primtype_tag();
            is_mask = (type_tag == PrimType_bool);
        }
    }
    return is_mask;
}


std::string CCodeGen::prepare(const Scope& scope) {
    auto cont = scope.entry();

    if (lang_ == Lang::CGRA && !cont->is_cgra_graph() && cont->is_exported())
        vector_size_ = get_vector_size(cont);

    //TODO: for interface attr. 1) check codegen use, cont (seems not working)
    //2) use old2new to find old cont. then probably  "using conts" then check for interface attr on that cont and then set for all new apps
    //3) adding interface as config
    StringStream hls_pragmas_;

    for (auto param : cont->params()) {
        defs_[param] = param->unique_name();
        if (lang_ == Lang::HLS && cont->is_exported() && param->type()->isa<PtrType>()) {
            auto elem_type = pointee_or_elem_type(param->type()->as<PtrType>());
            if ((elem_type->isa<StructType>() || elem_type->isa<DefiniteArrayType>()) && !top_scope.hls)
                hls_pragmas_.fmt("#pragma HLS data_pack variable={} struct_level\n", param->unique_name());
        }
    }

    if (top_scope.cgra_graph && (lang_ == Lang::CGRA)) {
        //func_impls_.fmt("{}", emit_class(cont));
        graph_stream_.fmt("\n{}\t\n", emit_class(cont));
    } else {
        if (lang_ == Lang::CGRA)
            graph_stream_.fmt("{};\n", emit_fun_head(cont));
        func_impls_.fmt("{} {{", emit_fun_head(cont));
    }
    func_impls_.fmt("\t\n");

    if (lang_ == Lang::HLS && cont->is_exported()) {
        if (top_scope.hls) {
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

                            func_impls_ << "#pragma HLS STABLE variable = " << param->unique_name() << "\n";

                        } else if(is_channel_type(param->type())) {
                            func_impls_ << "#pragma HLS INTERFACE axis port = " << param->unique_name() << "\n";

                        } else {

                            if (interface == HlsInterface::SOC)
                                func_impls_ << "#pragma HLS INTERFACE s_axilite port = " << param->unique_name() << "\n";
                            else if (interface == HlsInterface::HPC)
                                func_impls_ << "#pragma HLS INTERFACE s_axilite port = " << param->unique_name() << " bundle = control" << "\n";
                        }

                    }

                    // return protocol
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
            std::string param_type_str;
            //TODO: assign 1 for scalar core
            //if (lang_ == Lang::CGRA && cont->is_exported())
            param_type_str = convert(param->type());
            if (lang_ == Lang::CGRA) {

                // TODO: can be merged with adjust vectore size lambda, so that we can make it less costly
                // in that context this analysis just needs to return zero if the param is not a loop index
                auto isa_pipline_index = [&] () {
                    for (auto use: cont->uses()) {
                        if (auto app = use->isa<App>(); app && app->callee()->isa_nom<Continuation>()) {
                            auto callee = app->callee()->as_nom<Continuation>();
                            if (callee->is_intrinsic())  {
                                if (callee->intrinsic() == Intrinsic::Pipeline)  {
                                    auto loop_body = app->arg(4);
                                    auto loop_index = loop_body->as<Continuation>()->param(1);
                                    if (param == loop_index)
                                        return true;
                                    }
                                }
                            }
                        }
                    return false;
                    };


                //TODO: we need two more analysis to consider the uses of pointers and branches.
                if (auto type = param->type(); (vector_size_ > 1) && (!type->isa<PtrType>())
                        && (cont->body()->callee() != world().branch()) ) {
                    if ((vector_size_ > 1) && !isa_pipline_index()) {


                        std::string reg_type;
                        if (is_accum_type(type)) {
                            reg_type = "aie::accum";
                        } else if (is_mask_type(type)) {
                            reg_type = "aie::mask";
                        } else {
                            reg_type = "aie::vector";
                        }

                        //TODO: This lambda should be implemented as an analysis thorin pass that sets a new attribute for the continuations
                        // Basicaly this new pass marks those continuations that have free variables belonging to the set of irregular_apis
                        // later on in the c-backend we can check for this attribute and adjust the vector size of params accordingly
                        // TODO:: all the the defs that use irregular defs also should be modifed
                        auto adjust_vector_size = [&] () {

                            using DeviceApiSet = std::unordered_set<std::string>;
                            auto new_vector_size = vector_size_;
                            DeviceApiSet irregular_apis = { "aie::vector::extract", "aie::store_v", "readincr_v", "window_readincr_v",
                                "aie::load_v", "aie::sliding_"/*all sliding APIs*/};

                            for (auto use: cont->uses()) {
                                if (auto app = use->isa<App>(); app && app->callee()->isa_nom<Continuation>()) {
                                    auto callee = app->callee()->as_nom<Continuation>();
                                    if (callee->cc() == CC::Device) {
                                        auto name = callee->name();
                                        if (name.find("aie::sliding_") != std::string::npos)
                                            name = "aie::sliding_";
                                        if (irregular_apis.count(name)) {
                                            if (app->num_args() > 1) {
                                                // The first arg of all irregular APIs is the lane size
                                                if (auto primtype = app->arg(1)->type()->isa<PrimType>()) {
                                                    if (primtype->primtype_tag() == PrimType_pu32) {
                                                        new_vector_size = app->arg(1)->as<PrimLit>()->value().get_u32();
                                                    } else {
                                                        world().WLOG("Lane size in {} must be an unsigned integer value to be effective", name);
                                                    }
                                                }


                                            }
                                        }

                                    }
                                }
                            }
                            return new_vector_size;
                        };

                        param_type_str = is_mask_type(type) ? (reg_type + "<" + std::to_string(vector_size_)+ ">") :
                            (reg_type + "<" + convert(param->type()) + ", " + std::to_string(adjust_vector_size())  + ">");
                    }
                }
            }
            func_impls_.fmt("{}   {};\n", param_type_str, param->unique_name());
            func_impls_.fmt("{} p_{};\n", param_type_str, param->unique_name());
            bb.head.fmt("{} = p_{};\n", param->unique_name(), param->unique_name());
            defs_[param] = param->unique_name();
        }
    }

}

static inline auto get_middle_token (const Continuation* cont , const std::string& delimiter = "::") {
    auto s = cont->name();
    size_t startPos = s.find_first_of(delimiter);
    if (startPos == std::string::npos) return std::string();
    startPos += delimiter.length();
    size_t endPos = s.find_last_of(delimiter);
    if (endPos == std::string::npos) return std::string();
    endPos -= 1;
    return s.substr(startPos, endPos - startPos);
};


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


inline std::string cgra_obj_name() { return "cgra_dataflow"; }

auto cgra_testbench(int32_t iteration) {
    StringStream s;
    // TODO: Check testbenchcode for non mem allocation
    // TODO: we probably need to remove interface pragmas for hls xdma interface when connecting to cgra
    // test HPC interface maybe it already does it.  for QDMA we need to add template args
    // set channel params of cgra graph to axis (both for xdma and qdma), in qdma mem i/o is also axis and it is stream
    // TODO: it seems we need 32bits for stream and 64 or 128 bits for window when creating virtual ports
    //if (options.iteration > 0) { }
    //TODO: Parallel Streams Access (aie_stream_resource_in::a, aie_stream_resource_out::a) on read and write stream Fns
    s << "#if defined(__AIESIM__) || defined(__X86SIM__)\n";
    s << "int main(void) {\n"
            << "\t" << cgra_obj_name() << ".init();\n"
            << "\t"<< cgra_obj_name() << ".run(" << iteration << ");\n"
            << "\tstd::cout << \"Graph executed " << iteration << " times\" << std::endl;\n"
            << "\t" << cgra_obj_name() << ".end();\n"
            << "\tstd::cout << \"Graph ended.\" << std::endl;\n"
            << "\treturn 0;\n"
    << "}\n";
    s << "#endif\n";
   return s.str();
}


void CCodeGen::finalize(const Scope& scope) {
    for (auto& def : func_defs_) {
        assert(defs_.contains(def) && "sanity check, should have been emitted if it's here");
        defs_.erase(def);
    }
    func_defs_.clear();
    if (top_scope.cgra_graph && (lang_ == Lang::CGRA)) {
        //func_impls_.fmt( "\b}};\n\n{} {};\n\n", scope.entry()->name(), cgra_obj_name());
        //func_impls_.fmt("{}", cgra_testbench());
        graph_stream_.fmt( "{} {};\n\n", scope.entry()->name(), cgra_obj_name());
        graph_stream_.fmt("{}", cgra_testbench(options.iteration));
    } else
        func_impls_.fmt("}}\n\n");
}

void CCodeGen::finalize(Continuation* cont) {
    //TODO: make a graph_gen function that gets conts from finalize conts
    auto&& bb = cont2bb_[cont];
    //if (cont->is_cgra_graph()) {

  //      std::cout << "Finalize all scope conts ";
  //      cont->dump();
  //  if (top_scope.cgra_graph) {
  //      std::cout << "Finalize\n";
  //      std::cout <<"continuation: "; cont->dump();
  //      //cont->body()->dump();
  //      std::cout <<"callee "; cont->body()->callee()->dump();
  //      for (auto arg : cont->body()->args()) {
  //        //  if (arg->isa_nom<Continuation>()) {
  //        //      arg->as<Continuation>()->body()->callee()->dump();
  //        //  }

  //      if (is_concrete(arg))
  //          arg->dump();
  //      }
  //  }
    if (cont != entry_)
        if (!top_scope.cgra_graph)
            func_impls_.fmt("{}: \t", label_name(cont));

   // if (cont->is_cgra_graph() && (lang_ == Lang::CGRA)){
   //     //bb.body.indent(2);
   //     //func_impls_ <<std::setw(5);
   //     bb.body.fmt("public:\n");
   // }
    if (top_scope.cgra_graph && lang_ == Lang::CGRA) {

    //bb.body.indent(2);
    bb.tail.indent(2);
        //func_impls_ <<std::setw(5);
        if (cont->is_cgra_graph()) {
            bb.body.fmt("public:\n{}() {{\n", cont->name());
            bb.tail.fmt("{}\b}};", graph_ctor_.str());
            //func_impls_.fmt("\n{}{}{}\n", bb.head.str(), bb.body.str(), bb.tail.str());
            graph_stream_.fmt("\n{}", bb.body.str());
            //graph_stream_.fmt("{}", graph_ctor_.str());
            graph_stream_.fmt("{}\b\n}};\n\n", bb.tail.str());
        }
    }
    else
        func_impls_.fmt("{{\t\n{}{}{}\b\n}}\b\n", bb.head.str(), bb.body.str(), bb.tail.str());
}

//void CCodeGen::finalize(Continuation* cont) {
//    auto&& bb = cont2bb_[cont];
//    if (cont != entry_)
//        func_impls_.fmt("{}: \t", label_name(cont));
//    func_impls_.fmt("{{\t\n{}{}{}\b\n}}\b\n", bb.head.str(), bb.body.str(), bb.tail.str());
//}


void CCodeGen::emit_epilogue(Continuation* cont) {
    auto&& bb = cont2bb_[cont];
    assert(cont->has_body());
    auto body = cont->body();
    emit_debug_info(bb.tail, body->arg(0));


    if ((lang_ == Lang::OpenCL || (lang_ == Lang::HLS && top_scope.hls)) && (cont->is_exported()))
        emit_fun_decl(cont);


    if (body->callee() == entry_->ret_param()) { // return
        std::vector<std::string> values;
        std::vector<const Type*> types;

        for (auto arg : body->args()) {
            if (auto val = emit_unsafe(arg); !val.empty()) {
                values.emplace_back(val);
                types.emplace_back(arg->type());
            }
        }

        switch (values.size()) {
            case 0: if (top_scope.cgra_graph && lang_ == Lang::CGRA) break;
                    bb.tail.fmt(lang_ == Lang::HLS ? "return void();" : "return;"); break;
            case 1: bb.tail.fmt("return {};", values[0]); break;
            default:
                auto tuple = convert(world().tuple_type(types));
                bb.tail.fmt("{} ret_val;\n", tuple);
                for (size_t i = 0, e = types.size(); i != e; ++i)
                    bb.tail.fmt("ret_val.e{} = {};\n", i, values[i]);
                bb.tail.fmt("return ret_val;");
                break;
        }
    } else if (body->callee() == world().branch()) {
        auto c = emit(body->arg(0));
        auto t = label_name(body->arg(1));
        auto f = label_name(body->arg(2));
        bb.tail.fmt("if ({}) goto {}; else goto {};", c, t, f);
    } else if (auto callee = body->callee()->as_nom<Continuation>(); callee && callee->intrinsic() == Intrinsic::Match) {
        bb.tail.fmt("switch ({}) {{\t\n", emit(body->arg(0)));

        for (size_t i = 2; i < body->num_args(); i++) {
            auto arg = body->arg(i)->as<Tuple>();
            bb.tail.fmt("case {}: goto {};\n", emit_constant(arg->op(0)), label_name(arg->op(1)));
        }

        bb.tail.fmt("default: goto {};", label_name(body->arg(1)));
        bb.tail.fmt("\b\n}}");
    } else if (body->callee()->isa<Bottom>()) {
        bb.tail.fmt("return;  // bottom: unreachable");
    } else if (auto callee = body->callee()->isa_nom<Continuation>(); callee && callee->is_basicblock()) { // ordinary jump
        assert(callee->num_params() == body->num_args());
        for (size_t i = 0, size = callee->num_params(); i != size; ++i) {
            if (auto arg = emit_unsafe(body->arg(i)); !arg.empty())
                bb.tail.fmt("p_{} = {};\n", callee->param(i)->unique_name(), arg);
        }
        bb.tail.fmt("goto {};", label_name(callee));
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
            if (lang_ == Lang::HLS && !top_scope.hls) {
                func_impls_.fmt("#pragma HLS dependence variable={}_reserved inter false\n", cont->unique_name());
                func_impls_.fmt("#pragma HLS data_pack  variable={}_reserved\n", cont->unique_name());
                func_impls_<< "#if defined( __VITIS_HLS__ )\n   __attribute__((packed))\n  #endif\n";
            }
            bb.tail.fmt("p_{} = {}_reserved;\n", ret_cont->param(1)->unique_name(), cont->unique_name());
            bb.tail.fmt("goto {};", label_name(ret_cont));
        } else if (callee->intrinsic() == Intrinsic::Pipeline) {
            assert((lang_ == Lang::OpenCL || lang_ == Lang::HLS || lang_ == Lang::CGRA) && "pipelining not supported on this backend");

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
            if (lang_ == Lang::CGRA) {
                bb.tail.fmt("for (i{} = {}; i{} < {}; i{}++)\nchess_prepare_for_pipelining {{\t\n",
                    callee->gid(), begin, callee->gid(), end, callee->gid());
                //bb.tail << "{\t\n";
            } else
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
            bb.tail.fmt("\b{}: continue;\n}}\n", label_name(body->arg(6)));
            bb.tail.fmt("goto {};", label_name(body->arg(5)));
        } else if (callee->intrinsic() == Intrinsic::PipelineContinue) {
            emit_unsafe(body->arg(0));
            bb.tail.fmt("goto {};", label_name(callee));
        } else {
            THORIN_UNREACHABLE;
        }
    } else if (auto callee = body->callee()->isa_nom<Continuation>()) { // function/closure call
        auto ret_cont = (*std::find_if(body->args().begin(), body->args().end(), [] (const Def* arg) {
            return arg->isa_nom<Continuation>();
        }))->as_nom<Continuation>();
        if (top_scope.cgra_graph && lang_ == Lang::CGRA) {
            if (cont->is_cgra_graph()) {
                //func_impls_ << "private:\n";
                graph_stream_ << "private:\n";
            }
            // cgra kernel obj definitions start with the letter 'k'
            //func_impls_<< "\tadf::kernel " << "k" << emit(callee) << ";\n";
            graph_stream_ <<"\tadf::kernel " << "k" << emit(callee) << ";\n";

        }

        std::vector<std::string> args;
        for (auto arg : body->args()) {
            if (arg == ret_cont) continue;
            if (auto emitted_arg = emit_unsafe(arg); !emitted_arg.empty())
                args.emplace_back(emitted_arg);
        }

   // if (cgra_graph_scope && lang_ == Lang::CGRA)
   //     if (cont->is_cgra_graph())
   //         bb.tail.fmt("private:\n");
   //     bb.tail.fmt("adf::kernel {};\n",emit(body->callee()->as_nom<Continuation>()));

        size_t num_params = ret_cont->num_params();
        size_t n = 0;
        Array<const Param*> values(num_params);
        Array<const Type*> types(num_params);
        for (auto param : ret_cont->params()) {
            if (!is_mem(param) && !is_unit(param)) {
                values[n] = param;
                types[n] = param->type();
                n++;
            }
        }

        const Param* channel_read_result = n == 1 ? values[0] : nullptr;

        bool channel_transaction = false, no_function_call = false;

        auto name = (callee->is_exported() || callee->empty()) ? callee->name() : callee->unique_name();
        if (lang_ == Lang::OpenCL && use_channels_ && callee->is_channel()) {
            auto [usage, _] = builtin_funcs_.emplace(callee, FuncMode::Read);

            if (name.find("write") != std::string::npos) {
                usage->second = FuncMode::Write;
            } else if (name.find("read") != std::string::npos) {
                usage->second = FuncMode::Read;
                assert(channel_read_result != nullptr);
                args.emplace(args.begin(), emit(channel_read_result));
            } else THORIN_UNREACHABLE;
            channel_transaction = true;
        } else if (lang_ == Lang::HLS && callee->is_channel()) {
            int i = 0;
            for (auto arg : body->args()) {
                if (!is_concrete(arg)) continue;
                if (i == 0)
                    bb.tail.fmt("*{}", emit(arg));
                if (i == 1) {
                    if (name.find("write_channel") != std::string::npos) {
                        bb.tail.fmt(" << {};\n", emit(arg));
                    } else THORIN_UNREACHABLE;
                }
                if (name.find("read_channel") != std::string::npos) {
                    bb.tail.fmt(" >> {};\n", emit(channel_read_result));
                }
                i++;
            }
            no_function_call = true;
            //TODO: Check it
            channel_transaction = true;
        } else if (lang_ == Lang::CGRA) {
            if (callee->is_channel()) {

                //TODO: Adapt the placeholders for ADF APIs and for differetn interfaces, start with Stream interface
                //TODO: Simplify it
                for (size_t i = 0; auto arg : body->args()) {
                    args.size();
                    if (!is_concrete(arg)) continue;
                    const Def* channel_def;
                    if (i == 0)
                        channel_def = arg; //channel
                    if (i == 1) {
                        if (name.find("write_channel") != std::string::npos) {
                            switch (cont->get_interface()) {
                                case Interface::Stream: case Interface::Free_running: case Interface::Cascade:
                                    bb.tail.fmt("writeincr({}, {});\n", emit(channel_def), emit(arg));
                                    break;
                                case Interface::Window:
                                    bb.tail.fmt("window_writeincr({}, {});\n", emit(channel_def), emit(arg));
                                    break;
                                default:
                                    world().WLOG("Interface not determined or not supported yet. Fall back on STREAM");
                                    bb.tail.fmt("writeincr({}, {});\n", emit(channel_def), emit(arg));
                            }
                        } else THORIN_UNREACHABLE;
                    }

                    auto config_read_api = [&] (std::string interface_prefix) {
                        bb.tail.fmt("{} = {}readincr{}({});\n",
                                emit(channel_read_result),
                                interface_prefix,
                                is_scalar_kernel() ? "" : "_v<" + std::to_string(vector_size_) + ">",
                                emit(channel_def));
                    };

                    if (name.find("read_channel") != std::string::npos) {
                        switch (cont->get_interface()) {
                            case Interface::Stream: case Interface::Free_running: case Interface::Cascade:
                                config_read_api("");
                                break;
                            case Interface::Window:
                                config_read_api("window_");
                                break;
                            default:
                                world().WLOG("Interface not determined or not supported yet. Fall back on STREAM");
                                config_read_api("");
                        }
                    }
                    ++i;
                }
                no_function_call = true;
                channel_transaction = true;
            }
        }

//TODO:: use a mode bool var for monitoring read/write changes to emit chess_scheduler

        // Do not store the result of `void` calls
        auto ret_type = thorin::c::ret_type(callee->type());
        if (!is_type_unit(ret_type) && !channel_transaction) {
            if (lang_ == Lang::CGRA)
                bb.tail.fmt("{} = ", emit(values[0]));
            else
                bb.tail.fmt("{} ret_val = ", convert(ret_type));
        }

        if (!no_function_call) { // rest of the calls
                                 //CGRA graph DEBUG point
            if (!top_scope.cgra_graph) {
                if (lang_ != Lang::CGRA) {
                    bb.tail.fmt("{}({, });\n", emit(callee), args);
                } else { // if it is a cgra graph kernel

                    std::vector<std::string> template_args, fun_args;
                    std::vector<std::string>::iterator args_split_point;
                    auto api_config = special_device_api(cont);
                    const Type* composite_type = nullptr;
                    if (api_config) {

                        auto [num_templ_params, type_params] = *api_config;
                        TempTypeParams temp_type_params;
                        if (std::holds_alternative<TempTypeParams>(type_params))
                            temp_type_params = get<TempTypeParams>(type_params);
                        else if (std::holds_alternative<const Type*>(type_params))
                            composite_type = get<const Type*>(type_params);
                        // if empty to fix the bug (tempTypeParam variant but empty)

                        if (temp_type_params.empty() && (args.size() > num_templ_params))
                            args_split_point = args.begin() + num_templ_params;
                        else
                            args_split_point = args.begin() + num_templ_params - temp_type_params.size();

                        template_args.insert(template_args.begin(), args.begin(), args_split_point);
                        fun_args.insert(fun_args.begin(), args_split_point, args.end());

                        if (!temp_type_params.empty()) {
                            // augmenting template args with the type params
                            for (auto[templ_param_index, type_of_arg] : temp_type_params) {
                                template_args.insert(template_args.begin() + templ_param_index, convert(type_of_arg));
                            }
                        }
                    }

                    // helper lambdas for device APIs
                    auto get_method = [] (const Continuation* cont) {
                        auto s = cont->name();
                        return s.substr(s.find_last_of("::") + 1);
                    };

                    auto get_struct = [] (const Continuation* cont) {
                        return get_middle_token(cont, "::");
                    };

                    auto check_membership= [&] (const Continuation* cont, const std::string& class_name) {
                        return get_struct(cont) == class_name;
                    };

                    auto shift_left = [] (std::vector<std::string> vec, size_t shift_by) {
                        if (shift_by >= vec.size()) {
                            vec.clear();
                        } else {
                            for (size_t i = shift_by; i < vec.size(); ++i) {
                                vec[i - shift_by] = vec[i];
                            }
                            vec.resize(vec.size() - shift_by);
                        return vec;
                        }
                    };

                    auto cont_is_command = [&] () {
                        // if it is a specialized intrinsic but without any args then it is a command API
                        return  api_config && template_args.empty() && fun_args.empty();
                    };

                    auto cont_is_class_obj_method = [&] () {
                        return ((vector_size_ > 1) && ((composite_type) && (composite_type->isa<StructType>())) ||
                                (check_membership(callee, "vector") && (template_args.size() == 0)));
                    };

                    auto cont_is_class_templ_obj_method = [&] () {
                        return ((vector_size_ > 1) && ((composite_type) && (composite_type->isa<StructType>())) ||
                                (check_membership(callee, "vector") && (template_args.size() > 0)));
                    };
 
                    auto cont_is_fun_template = [&] () {
                        return (template_args.size() > 0); // not enough to check if it is a template
                    };

                    auto cont_is_struct_templ_method = [&] () {
                        return ((template_args.size() > 1 ) && (!get_struct(callee).empty()));
                    };

                    // emit the device API
                    if (cont_is_command()) {
                        bb.tail.fmt("{};\n", emit(callee));
                    } else if(cont_is_class_obj_method()) {
                        auto composite_arg = args[0];
                        bb.tail.fmt("{}.{}({, });\n", composite_arg , get_method(callee), shift_left(args, 1) );
                    } else if(cont_is_class_templ_obj_method()) {
                        auto composite_arg = args[1];
                        bb.tail.fmt("{}.{}<{}>({, });\n", composite_arg , get_method(callee), template_args, shift_left(args, 1 + template_args.size()));
                    } else if (template_args.empty()) {
                        bb.tail.fmt("{}({, });\n", emit(callee), args);
                    } else if (cont_is_fun_template()) {
                        bb.tail.fmt("{}<{, }>({, });\n", emit(callee), template_args, fun_args);
                    } else if (cont_is_struct_templ_method()) {
                        bb.tail.fmt("{}<{, }>::{}({, });\n", get_struct(callee), template_args, get_method(callee), fun_args);
                    }
                    else THORIN_UNREACHABLE;
                }
            }
        }

        // Pass the result to the phi nodes of the return continuation
        if (!is_type_unit(ret_type)) {
            size_t i = 0;
            for (auto param : ret_cont->params()) {
                if (!is_concrete(param))
                    continue;
                // TODO: tuple type bypass should be handled
                if (ret_type->isa<TupleType>())
                    bb.tail.fmt("p_{} = ret_val.e{};\n", param->unique_name(), i++);
                else if ((lang_ == Lang::OpenCL && use_channels_) || (lang_ == Lang::HLS) || (lang_ == Lang::CGRA))
                    bb.tail.fmt("p_{} = {};\n", emit(channel_read_result), param->unique_name());
                else
                    bb.tail.fmt("p_{} = ret_val;\n", param->unique_name());
            }
        }
        // TODO:: simplify this logic
        if (!cont->is_cgra_graph() && lang_ == Lang::CGRA && !top_scope.cgra_graph)
            bb.tail.fmt("goto {};", label_name(ret_cont));
        if (!cont->is_hls_top() && lang_ == Lang::HLS && !top_scope.hls)
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
    return def->isa_structural() && !def->has_dep(Dep::Param);
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
            // First we check about CGRA requirements
            // TODO: make a function for this
            if (lang_ == Lang::CGRA && is_mmul_type(def->type())) {
                auto struct_type = def->type()->as<StructType>();
                auto type_of_mmul = [&] (const Type* type) {
                    auto s = convert(type);
                    return s.substr(s.find_last_of('_') + 1);
                };

                auto mmul_size = [&] () {
                    std::string m, n, k;
                    for (int i = 0; const auto& op : def->ops()) {
                        auto op_name = struct_type->op_name(i++).c_str();

                        if (strlen(op_name) > 1) continue;

                        switch(*op_name) {
                            case 'M': case 'm' : m = emit_unsafe(op); break;
                            case 'N': case 'n' : n = emit_unsafe(op); break;
                            case 'K': case 'k' : k = emit_unsafe(op); break;
                            default: m = '4'; n = '2'; k = '4';
                        }

                    }
                    return std::make_tuple(m, n, k);
                };
                auto[m ,n, k] = mmul_size();
                func_impls_.fmt("aie::mmul<{}, {}, {}, {}, {}> {};\n",
                        m, n, k, type_of_mmul(def->type()), type_of_mmul(def->type()), name);
                func_defs_.insert(def);
            } else {
                func_impls_.fmt("{} {};\n", convert(def->type()), name);
                func_defs_.insert(def);
                for (size_t i = 0, n = def->num_ops(); i < n; ++i) {
                    auto op = emit_unsafe(def->op(i));
                    bb->body << name;
                    emit_access(bb->body, def->type(), world().literal(thorin::pu64{i}));
                    bb->body.fmt(" = {};\n", op);
                }
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
        //TODO: check slot condition, as we need slots for CGRA kernels but not for cgra_graph
        if (!top_scope.cgra_graph ) {
            func_impls_.fmt("{} {}_slot;\n", t, name);
            func_impls_.fmt("{}* {} = &{}_slot;\n", t, name, name);
        }
        func_defs_.insert(def);
        if (top_scope.hls)
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
        auto emitted_type_str = convert(emitted_type) ;

        auto is_pipeline_body = [&] (Continuation* cont) {
            auto pred_cont = cont->as_nom<Continuation>()->preds().back();
            auto pred_callee = pred_cont->body()->callee();
            return pred_callee->isa_nom<Continuation>()->intrinsic() == Intrinsic::Pipeline;
        };


        if (is_cgra_vector_kernel() && (!emitted_type->isa<PtrType>()) && (!is_mask_type(emitted_type))) {
            // This condition is to avoid the vectorization of the pipeline body but only for window interface since there is no
            // loop pipelining in stream interface
            if (!is_pipeline_body(bb->cont) || (!def->isa<Load>() && !def->isa<BinOp>() && !def->isa<AggOp>())) {
                std::string reg_type = is_accum_type(emitted_type) ? "aie::accum" : "aie::vector";
                emitted_type_str = reg_type + "<" + emitted_type_str + ", " + std::to_string(vector_size_) + ">";
            }
        }
        func_impls_.fmt("{} {};\n", emitted_type_str, name);
        func_defs_.insert(def);
        bb->body.fmt("{} = {};\n", name, s.str());
        return name;
    } else
        return "(" + s.str() + ")";
}

std::string CCodeGen::emit_class(Continuation* cont) {
    assert(cont->is_cgra_graph() && "Class generation is only for CGRA");
    StringStream s;

    s.fmt("class {} : public adf::graph {{\t\npublic:", cont->name());

    // skipping non-concrete params
    for (size_t i = 0, n = cont->num_params(); i < n; ++i) {
        auto param = cont->param(i);
        if (!is_concrete(param)) {
            defs_[param] = {};
            continue;
        }

        s.fmt("\n");
        // Emit and store all first-order params
        if (cont->is_exported() && is_passed_via_buffer(param)) {

            // OpenCL structs are passed via buffer; the parameter is a pointer to this buffer
            s << convert(param->type()) << "*";
            s.fmt(" {}_", param->unique_name());
            //TODO:: The following if blocks can be simplified as the type of param (channel or gmem) is checked in the prefix_type function
        } else if (cont->is_exported() && is_passed_via_global_mem(param)) {
            //    auto param_mode = config->as<CGRAKernelConfig>()->param_mode(param);
            //    if (param_mode != ChannelMode::Undef )
            //        std::cout << "param_mode found \n";
            //assert(param_mode);
            auto ptr_type = param->type()->as<PtrType>();
            auto elem_type = ptr_type->pointee();
            if (auto array_type = elem_type->isa<ArrayType>()){
                elem_type = array_type->elem_type();
            }
            // global memory.
            s << prefix_type(param) << " " << param->unique_name() << ";";
        } else {
            s.fmt("{} {};", prefix_type(param), param->unique_name());
        }

    }

    // interface for cgra_graph module (class) should be always None
    //auto intr = cont->get_interface();
    return s.str();
}

std::string CCodeGen::emit_fun_head(Continuation* cont, bool is_proto) {
    StringStream s;

    // Emit function qualifiers
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
            case Lang::CGRA:
                std::cout << "C.CPP" <<std::endl;
                if (cont->get_interface() == Interface::None) {
                    std::cout << "Graph Interface None" <<std::endl;
                } else if (cont->get_interface() == Interface::Stream) {
                    std::cout << "C.cpp Stream" <<std::endl;
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
        convert(ret_type(cont->type())),
        !world().is_external(cont) ? cont->unique_name() : cont->name());

    // Emit and store all first-order params
    bool needs_comma = false;
    for (size_t i = 0, n = cont->num_params(); i < n; ++i) {
        auto param = cont->param(i);
        if (!is_concrete(param)) {
            defs_[param] = {};
            continue;
        }
        if (needs_comma) s.fmt(", ");

        auto config = get_config(cont);
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
            if (lang_ == Lang::CGRA && !cont->is_cgra_graph())
                s.fmt("{}{}", prefix_type(param), qualifier);
            else
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
    world().cleanup();

    for (auto def : world().defs()) {
        auto cont = def->isa_nom<Continuation>();
        if (!cont)
            continue;
        if (!cont->is_external())
            continue;
        if (cont->cc() != CC::C && cont->is_imported())
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

//------------------------------------------------------------------------------
void CodeGen::emit_stream(std::ostream& stream) {
    Stream s0(stream);
    Stream s1 = {};
    CCodeGen(world(), kernel_config_, s0, s1,  lang_, debug_, flags_).emit_module();
}

void CodeGen::emit_stream(std::ostream& stream0, std::ostream& stream1) {
    if (lang_ != Lang::CGRA)
        world().WLOG("This backend does not support multiple streams");
    Stream s0(stream0);
    Stream s1(stream1);
    CCodeGen CCodeGen_obj(world(), kernel_config_, s0, s1, lang_, debug_, flags_);
    CCodeGen_obj.emit_module();
}

void emit_c_int(World& world, Stream& stream) {
    std::string flags;
    Stream s {};
    CCodeGen(world, {}, stream, s, Lang::C99, false, flags).emit_c_int();
}

//------------------------------------------------------------------------------

}
