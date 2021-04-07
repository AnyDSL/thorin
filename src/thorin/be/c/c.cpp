#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/be/emitter.h"
#include "thorin/util/stream.h"
#include "c.h"

#include <cmath>
#include <sstream>
#include <type_traits>
#include <cctype>
#include <variant>

namespace thorin::c {

struct BB {
    BB() = default;

    StringStream head;
    StringStream body;
    StringStream tail;

    friend void swap(BB& a, BB& b) {
        using std::swap;
        swap(a.head, b.head);
        swap(a.body, b.body);
        swap(a.tail, b.tail);
    }
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
    Stream& emit_debug_info(Stream&, const Def*);

    template <typename T, typename IsInfFn, typename IsNanFn>
    std::string emit_float(T, IsInfFn, IsNanFn);

    const std::string var_name(const Def*);
    const std::string get_lang() const;

    std::string type_name(const Type*);
    std::string array_name(const DefiniteArrayType*);
    std::string tuple_name(const TupleType*);

    World& world_;
    const Cont2Config& kernel_config_;
    Lang lang_;
    const FnType* fn_mem_;
    bool use_fp_64_ = false;
    bool use_fp_16_ = false;
    bool use_channels_ = false;
    bool use_align_of_ = false;
    bool use_memcpy_ = false;
    bool use_malloc_ = false;
    bool debug_;
    int primop_counter = 0;

    Stream& stream_;
    StringStream func_impls_;
    StringStream func_decls_;
    StringStream type_decls_;

    friend class CEmit;
};

static inline bool is_string_type(const Type* type) {
    if (auto array = type->isa<DefiniteArrayType>())
        if (auto primtype = array->elem_type()->isa<PrimType>())
            if (primtype->primtype_tag() == PrimType_pu8)
                return true;
    return false;
}

static std::string handle_string_character(char c) {
    switch (c) {
        case '\a': return "\\a";
        case '\b': return "\\b";
        case '\f': return "\\f";
        case '\n': return "\\n";
        case '\r': return "\\r";
        case '\t': return "\\t";
        case '\v': return "\\v";
        default:   return std::string(1, c);
    }
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
        if (lang_ == Lang::OpenCL) {
            switch (ptr->addr_space()) {
                default:
                case AddrSpace::Generic:                  break;
                case AddrSpace::Global: s << "__global "; break;
                case AddrSpace::Shared: s << "__local ";  break;
            }
        }
        s.fmt("{}*", convert(ptr->pointee()));
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
        name = type_name(variant);
        types_[variant] = name;
        auto tag_type =
            variant->num_ops() < (UINT64_C(1) <<  8u) ? world_.type_qu8()  :
            variant->num_ops() < (UINT64_C(1) << 16u) ? world_.type_qu16() :
            variant->num_ops() < (UINT64_C(1) << 32u) ? world_.type_qu32() :
                                                        world_.type_qu64();
        s.fmt("typedef struct {{\t\n");

        // This is required because we have zero-sized types but C/C++ do not
        if (!std::all_of(variant->ops().begin(), variant->ops().end(), is_type_unit)) {
            s.fmt("union {{\t\n");
            s.rangei(variant->ops(), "\n", [&] (size_t i) {
                if (is_type_unit(variant->op(i)))
                    s << "// ";
                s.fmt("{} {};", convert(variant->op(i)), variant->op_name(i));
            });
            s.fmt("\b\n}} data;\n");
        }

        s.fmt("{} tag;", convert(tag_type));
        s.fmt("\b\n}} {};", name);
    } else if (auto struct_type = type->isa<StructType>()) {
        name = type_name(struct_type);
        types_[struct_type] = name;
        s.fmt("typedef struct {{\t\n");
        s.rangei(struct_type->ops(), "\n", [&](size_t i) { s.fmt("{} {};", convert(struct_type->op(i)), struct_type->op_name(i)); });
        s.fmt("\b\n}} {};\n", name);
        if (struct_type->name().str().find("channel_") != std::string::npos) use_channels_ = true;
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

/*
 * emit
 */

void CCodeGen::emit_module() {
    if (lang_ == Lang::CUDA) {
        for (auto x : std::array {'x', 'y', 'z'}) {
            func_decls_.fmt("__device__ inline int threadIdx_{}() { return threadIdx.{}; }\n", x, x);
            func_decls_.fmt("__device__ inline int blockIdx_{}() { return blockIdx.{}; }\n", x, x);
            func_decls_.fmt("__device__ inline int blockDim_{}() { return blockDim.{}; }\n", x, x);
            func_decls_.fmt("__device__ inline int gridDim_{}() { return gridDim.{}; }\n", x, x);
        }
    }

    Scope::for_each(world(), [&] (const Scope& scope) { emit_scope(scope); });

    if (lang_ == Lang::OpenCL) {
        if (use_channels_)
            stream_.fmt("#pragma OPENCL EXTENSION cl_intel_channels : enable\n");
        if (use_fp_16_)
            stream_.fmt("#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");
        if (use_fp_64_)
            stream_.fmt("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
        if (use_channels_ || use_fp_16_ || use_fp_64_)
            stream_.endl();
    }

    if (lang_ == Lang::C99) {
        stream_.fmt("#include <stdbool.h>\n"); // for the 'bool' type
        if (use_align_of_)
            stream_.fmt("#include <stdalign.h>\n"); // for 'alignof'
        if (use_memcpy_)
            stream_.fmt("#include <string.h>\n"); // for 'memcpy'
        if (use_malloc_)
            stream_.fmt("#include <stdlib.h>\n"); // for 'malloc'
        stream_.fmt("\n");
    }

    if (lang_ == Lang::CUDA && use_fp_16_) {
        stream_.fmt("#include <cuda_fp16.h>\n\n");
        stream_.fmt("#if __CUDACC_VER_MAJOR__ > 8\n");
        stream_.fmt("#define half __half_raw\n");
        stream_.fmt("#endif\n\n");
    }

    if (lang_ == Lang::CUDA || lang_ == Lang::HLS) {
        stream_.fmt("extern \"C\" {{\t\n");
    }

    stream_ << type_decls_.str();
    stream_.endl() << func_decls_.str();
    stream_.endl() << func_impls_.str();

    if (lang_ == Lang::CUDA || lang_ == Lang::HLS)
        stream_.fmt("}} /* extern \"C\" */\n");
}

inline bool is_passed_via_buffer(const Param* param) {
    return
        param->type()->isa<DefiniteArrayType>() ||
        param->type()->isa<StructType>() ||
        param->type()->isa<TupleType>();
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

    // Place parameters in map and gather HLS pragmas
    std::string hls_pragmas;
    for (auto param : cont->params()) {
        defs_[param] = param->unique_name();
        if (lang_ == Lang::HLS && cont->is_exported() && param->type()->isa<PtrType>()) {
            auto elem_type = pointee_or_elem_type(param->type()->as<PtrType>());
            if (elem_type->isa<StructType>() || elem_type->isa<DefiniteArrayType>())
                hls_pragmas += "#pragma HLS data_pack variable=" + param->unique_name() + " struct_level\n";
        }
    }

    func_impls_.fmt("{} {{", emit_fun_head(cont));
    if (!hls_pragmas.empty())
        func_impls_.fmt("\n{}", hls_pragmas);
    func_impls_.fmt("\t\n");

    // Load OpenCL structs from buffers
    // TODO: See above
    for (auto param : cont->params()) {
        if (is_mem(param) || is_unit(param) || param->order() > 0)
            continue;
        if (lang_ == Lang::OpenCL && cont->is_exported() && is_passed_via_buffer(param))
            func_impls_.fmt("{} {} = *{}_;\n", convert(param->type()), param->unique_name(), param->unique_name());
    }
    return {};
}

void CCodeGen::prepare(Continuation* cont, const std::string&) {
    auto& bb = cont2bb_[cont];
    bb.head.indent(2);
    bb.body.indent(2);
    bb.tail.indent(2);
    // The parameters of the entry continuation have already been emitted.
    if (cont != entry_) {
        for (auto param : cont->params()) {
            if (is_mem(param) || is_unit(param) || param->order() > 0) {
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

void CCodeGen::finalize(const Scope&) {
    func_impls_.fmt("}}\n\n");
}

void CCodeGen::finalize(Continuation* cont) {
    auto&& bb = cont2bb_[cont];
    if (cont != entry_)
        func_impls_.fmt("{}: \t", cont->unique_name());
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
            case 0: bb.tail.fmt("return;");               break;
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
        auto t = cont->arg(1)->unique_name();
        auto f = cont->arg(2)->unique_name();
        bb.tail.fmt("if ({}) goto {}; else goto {};", c, t, f);
    } else if (auto callee = cont->callee()->isa_continuation(); callee && callee->intrinsic() == Intrinsic::Match) {
        bb.tail.fmt("switch ({}) {{\t\n", emit(cont->arg(0)));

        for (size_t i = 2; i < cont->num_args(); i++) {
            auto arg = cont->arg(i)->as<Tuple>();
            bb.tail.fmt("case {}: goto {};\n", emit_constant(arg->op(0)), arg->op(1)->unique_name());
        }

        bb.tail.fmt("default: goto {};", cont->arg(1)->unique_name());
        bb.tail.fmt("\b\n}}");
    } else if (cont->callee()->isa<Bottom>()) {
        bb.tail.fmt("return;  // bottom: unreachable");
    } else if (auto callee = cont->callee()->isa_continuation(); callee && callee->is_basicblock()) { // ordinary jump
        assert(callee->num_params() == cont->num_args());
        for (size_t i = 0, size = callee->num_params(); i != size; ++i) {
            if (auto arg = emit_unsafe(cont->arg(i)); !arg.empty())
                bb.tail.fmt("p_{} = {};\n", callee->param(i)->unique_name(), arg);
        }
        bb.tail.fmt("goto {};", callee->unique_name());
    } else if (auto callee = cont->callee()->isa_continuation(); callee && callee->is_intrinsic()) {
#if 0
        if (callee->intrinsic() == Intrinsic::Reserve) {
            if (!cont->arg(1)->isa<PrimLit>())
                world().EDEF(cont->arg(1), "reserve_shared: couldn't extract memory size");

            switch (lang_) {
                case Lang::CUDA:   bb.tail.fmt("__shared__ "); break;
                case Lang::OpenCL: bb.tail.fmt("__local ");    break;
                default:                                        break;
            }

            auto cont = cont->arg(2)->as_cont();
            auto elem_type = cont->param(1)->type()->as<PtrType>()->pointee()->as<ArrayType>()->elem_type();
            auto name = "reserver_" + cont->param(1)->unique_name();
            convert(bb.tail, elem_type) << " " << name << "[";
            emit(cont->arg(1)) << "];" << endl;
            // store_phi:
            bb.tail << "p" << cont->param(1)->unique_name() << " = " << name << ";";
            if (lang_ == Lang::HLS)
                bb.tail << endl
                            << "#pragma HLS dependence variable=" << name << " inter false" << endl
                            << "#pragma HLS data_pack  variable=" << name;
        } else if (callee->intrinsic() == Intrinsic::Pipeline) {
            assert((lang_ == Lang::OpenCL || lang_ == Lang::HLS) && "pipelining not supported on this backend");
            // cast to cont to get unique name of "for index"
            auto body = cont->arg(4)->as_cont();
            if (lang_ == Lang::OpenCL) {
                if (cont->arg(1)->as<PrimLit>()->value().get_s32() !=0) {
                    bb.tail << "#pragma ii ";
                    emit(cont->arg(1)) << endl;
                } else {
                    bb.tail << "#pragma ii 1"<< endl;
                }
            }
            bb.tail << "for (i" << callee->gid() << " = ";
            emit(cont->arg(2));
            bb.tail << "; i" << callee->gid() << " < ";
            emit(cont->arg(3)) <<"; i" << callee->gid() << "++) {"<< up << endl;
            if (lang_ == Lang::HLS) {
                if (cont->arg(1)->as<PrimLit>()->value().get_s32() != 0) {
                    bb.tail << "#pragma HLS PIPELINE II=";
                    emit(cont->arg(1)) << endl;
                } else {
                    bb.tail << "#pragma HLS PIPELINE"<< endl;
                }
            }
            // emit body and "for index" as the "body parameter"
            bb.tail << "p" << body->param(1)->unique_name() << " = i"<< callee->gid()<< ";" << endl;
            emit(body);
            // emit "continue" with according label used for goto
            bb.tail << down << endl << "l" << cont->arg(6)->gid() << ": continue;" << endl << "}" << endl;
            if (cont->arg(5) == ret_param)
                bb.tail << "return;" << endl;
            else
                emit(cont->arg(5));
        } else if (callee->intrinsic() == Intrinsic::PipelineContinue) {
            bb.tail << "goto l" << callee->gid() << ";" << endl;
        } else {
            THORIN_UNREACHABLE;
        }
#endif
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
                if (is_mem(param) || is_unit(param) || param->order() > 0)
                    continue;
                if (ret_type->isa<TupleType>())
                    bb.tail.fmt("p_{} = ret_val.e{};\n", param->unique_name(), i++);
                else
                    bb.tail.fmt("p_{} = ret_val;\n", param->unique_name());
            }
        }
        bb.tail.fmt("goto {};", ret_cont->unique_name());
    } else {
        THORIN_UNREACHABLE;
    }
}

std::string CCodeGen::emit_constant(const Def* def) {
    if (def->isa<Aggregate>() || def->isa<DefiniteArray>()) {
        auto is_agg = def->isa<Aggregate>();
        StringStream s;
        s.fmt(is_agg ? "{{\t\n" : "{{ {{ ");
        s.range(def->ops(), is_agg ? ",\n" : ", ", [&] (const Def* op) { s << emit_constant(op); });
        s.fmt(is_agg ? "\b\n}}" : "}} }}");
        return s.str();
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
        THORIN_UNREACHABLE;
    } else {
        // This is not a constant
        THORIN_UNREACHABLE;
    }
}

void CCodeGen::emit_access(Stream& s, const Type* agg_type, const Def* index, const std::string_view& prefix) {
    if (agg_type->isa<DefiniteArrayType>()) {
        s.fmt("{}e[{}]", prefix, emit(index));
    } else if (agg_type->isa<IndefiniteArrayType>()) {
        s.fmt("[{}]", emit(index));
    } else if (agg_type->isa<TupleType>()) {
        s.fmt("{}e{}", prefix, emit(index));
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

std::string CCodeGen::emit_bb(BB& bb, const Def* def) {
    auto name = var_name(def);

    if (auto bin = def->isa<BinOp>()) {
        auto a = emit(bin->lhs());
        auto b = emit(bin->rhs());

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
                case ArithOp_add: op ="+";  break;
                case ArithOp_sub: op ="-";  break;
                case ArithOp_mul: op ="*";  break;
                case ArithOp_div: op ="/";  break;
                case ArithOp_rem: op ="%";  break;
                case ArithOp_and: op ="&";  break;
                case ArithOp_or:  op ="|";  break;
                case ArithOp_xor: op ="^";  break;
                case ArithOp_shl: op ="<<"; break;
                case ArithOp_shr: op =">>"; break;
            }
        }

        func_impls_.fmt("{} {};\n", convert(bin->type()), name);
        bb.body.fmt("{} = {} {} {};\n", name, a, op, b);
    } else if (auto conv = def->isa<ConvOp>()) {
        auto s_type = conv->from()->type();
        auto d_type = conv->type();
        auto s_ptr = s_type->isa<PtrType>();
        auto d_ptr = d_type->isa<PtrType>();
        auto src = emit(conv->from());

        auto s_t = convert(s_type);
        auto d_t = convert(d_type);

        func_impls_.fmt("{} {};\n", d_t, name);
        if (s_ptr && s_ptr->pointee()->isa<DefiniteArrayType>() &&
            d_ptr && d_ptr->pointee()->isa<IndefiniteArrayType>() &&
            s_ptr->pointee()->as<ArrayType>()->elem_type() ==
            d_ptr->pointee()->as<ArrayType>()->elem_type()) {
            bb.body.fmt("{} = {}->e;\n", name, src);
        } else if (s_ptr && d_ptr && s_ptr->addr_space() == d_ptr->addr_space()) {
            bb.body.fmt("{} = ({}) {};\n", name, d_t, src);
        } else if (conv->isa<Cast>()) {
            auto s_prim = s_type->as<PrimType>();
            auto d_prim = d_type->as<PrimType>();

            if (lang_ == Lang::CUDA && s_prim && (s_prim->primtype_tag() == PrimType_pf16 || s_prim->primtype_tag() == PrimType_qf16)) {
                bb.body.fmt("{} = __half2float({});\n", name, src);
            } else if (lang_ == Lang::CUDA && d_prim && (d_prim->primtype_tag() == PrimType_pf16 || d_prim->primtype_tag() == PrimType_qf16)) {
                bb.body.fmt("{} = __float2half({});\n", name, src);
            } else {
                bb.body.fmt("{} = ({}) {};\n", name, d_t, src);
            }
        } else if (conv->isa<Bitcast>()) {
            if (lang_ == Lang::OpenCL) {
                // OpenCL explicitly supports type punning via unions (6.4.4.1)
                bb.body.fmt("union {{\t\n");
                bb.body.fmt("{} src;\n",   s_t);
                bb.body.fmt("{} dst;\b\n", d_t);
                bb.body.fmt("}} {}_u;\n", name);
                bb.body.fmt("{}_u.src = {};\n", name, src);
                bb.body.fmt("{} = {}_u.dst;\n", name, name);
            } else {
                bb.body.fmt("memcpy(&{}, &{}, sizeof({}));\n", name, src, name);
                use_memcpy_ = true;
            }
        }
    } else if (auto align_of = def->isa<AlignOf>()) {
        if (lang_ == Lang::C99 || lang_ == Lang::OpenCL) {
            world().wdef(def, "alignof() is only available in C11");
            use_align_of_ = true;
        }
        return "alignof(" + convert(align_of->of()) + ")";
    } else if (auto size_of = def->isa<SizeOf>()) {
        return "sizeof(" + convert(size_of->of()) + ")";
    } else if (def->isa<IndefiniteArray>()) {
        func_impls_.fmt("{} {}; // indefinite array: bottom\n", convert(def->type()), name);
    } else if (def->isa<Aggregate>() || def->isa<DefiniteArray>()) {
        func_impls_.fmt("{} {};\n", convert(def->type()), name);
        for (size_t i = 0, n = def->num_ops(); i < n; ++i) {
            auto op = emit(def->op(i));
            bb.body << name;
            emit_access(bb.body, def->type(), world().literal(thorin::pu64{i}));
            bb.body.fmt(" = {};\n", op);
        }
    } else if (auto agg_op = def->isa<AggOp>()) {
        if (auto agg = emit_unsafe(agg_op->agg()); !agg.empty()) {
            emit(agg_op->index());
            if (auto extract = def->isa<Extract>()) {
                if (is_mem(extract) || extract->type()->isa<FrameType>())
                    return "";
                func_impls_.fmt("{} {};\n", convert(extract->type()), name);
                bb.body.fmt("{} = {}", name, agg);
                if (!extract->agg()->isa<MemOp>() && !extract->agg()->isa<Assembly>())
                    emit_access(bb.body, extract->agg()->type(), extract->index());
                bb.body.fmt(";\n");
            } else if (auto insert = def->isa<Insert>()) {
                if (auto value = emit_unsafe(insert->value()); !value.empty()) {
                    func_impls_.fmt("{} {}\n;", convert(insert->type()), name);
                    bb.body.fmt("{} = {};\n", name, agg);
                    bb.body.fmt("{}", name);
                    emit_access(bb.body, insert->agg()->type(), insert->index());
                    bb.body.fmt(" = {}\n;", value);
                }
            }
        } else {
            return "";
        }
    } else if (auto primlit = def->isa<PrimLit>()) {
        return emit_constant(primlit);
    } else if (auto variant = def->isa<Variant>()) {
        func_impls_.fmt("{} {};\n", convert(variant->type()), name);
        if (auto value = emit_unsafe(variant->value()); !value.empty())
            bb.body.fmt("{}.data.{} = {};\n", name, variant->type()->as<VariantType>()->op_name(variant->index()), value);
        bb.body.fmt("{}.tag = {};\n", name, variant->index());
    } else if (auto variant_index = def->isa<VariantIndex>()) {
        func_impls_.fmt("{} {};\n", convert(variant_index->type()), name);
        bb.body.fmt("{} = {}.tag;\n", name, emit(variant_index->op(0)));
    } else if (auto variant_extract = def->isa<VariantExtract>()) {
        func_impls_.fmt("{} {};\n", convert(variant_extract->type()), name);
        auto variant_type = variant_extract->value()->type()->as<VariantType>();
        bb.body.fmt("{} = {}.data.{};\n", name, emit(variant_extract->value()), variant_type->op_name(variant_extract->index()));
    } else if (auto bottom = def->isa<Bottom>()) {
        func_impls_.fmt("{} {}; // bottom\n", convert(bottom->type()), name);
    } else if (auto load = def->isa<Load>()) {
        emit_unsafe(load->mem());
        auto ptr = emit(load->ptr());
        func_impls_.fmt("{} {};\n", convert(load->out_val()->type()), name);
        bb.body.fmt("{} = *{};\n", name, ptr);
    } else if (auto store = def->isa<Store>()) {
        // TODO: IndefiniteArray should be removed
        if (store->val()->isa<IndefiniteArray>())
            return "";
        emit_unsafe(store->mem());
        bb.body.fmt("*{} = {};\n", emit(store->ptr()), emit(store->val()));
        return "";
    } else if (auto slot = def->isa<Slot>()) {
        emit_unsafe(slot->frame());
        auto t = convert(slot->alloced_type());
        func_impls_.fmt("{} {}_slot;\n", t, name);
        func_impls_.fmt("{}* {} = &{}_slot;\n", t, name, name);
    } else if (auto alloc = def->isa<Alloc>()) {
        use_malloc_ = true;
        emit_unsafe(alloc->mem());
        auto t = convert(alloc->alloced_type());
        func_impls_.fmt("{}* {};\n", t, name);

        if (auto array = alloc->alloced_type()->isa<IndefiniteArrayType>()) {
            auto extra = emit(alloc->extra());
            bb.body.fmt("{} = malloc(sizeof({}) * {});\n", name, t, extra);
        } else {
            bb.body.fmt("{} = malloc(sizeof({}));\n", name, t);
        }
    } else if (auto enter = def->isa<Enter>()) {
        return emit_unsafe(enter->mem());
    } else if (auto lea = def->isa<LEA>()) {
        auto ptr = emit(lea->ptr());
        auto index = emit(lea->index());
        func_impls_.fmt("{} {};\n", convert(lea->type()), name);
        bb.body.fmt("{} = &{}", name, ptr);
        emit_access(bb.body, lea->ptr_pointee(), lea->index(), "->");
        bb.body.fmt(";\n");
    } else if (auto assembly = def->isa<Assembly>()) {
#if 0
        size_t out_size = assembly->type()->num_ops() - 1;
        Array<std::string> outputs(out_size, std::string(""));
        for (auto use : assembly->uses()) {
            auto extract = use->as<Extract>();
            size_t index = primlit_value<unsigned>(extract->index());
            if (index == 0)
                continue;   // skip the mem

            assert(outputs[index - 1] == "" && "Each use must belong to a unique index.");
            auto name = var_name(extract);
            outputs[index - 1] = name;
            convert(func_impls_, assembly->type()->op(index)) << " " << name << ";" << endl;
        }
        // some outputs that were originally there might have been pruned because
        // they are not used but we still need them as operands for the asm
        // statement so we need to generate them here
        for (size_t i = 0; i < out_size; ++i) {
            if (outputs[i] == "") {
                auto name = var_name(assembly) + "_" + std::to_string(i + 1);
                convert(func_impls_, assembly->type()->op(i + 1)) << " " << name << ";" << endl;
                outputs[i] = name;
            }
        }

        // emit temporaries
        for (auto op : assembly->ops())
            emit_temporaries(op);

        func_impls_ << "asm ";
        if (assembly->has_sideeffects())
            func_impls_ << "volatile ";
        if (assembly->is_alignstack() || assembly->is_inteldialect())
            WDEF(assembly, "stack alignment and inteldialect flags unsupported for C output");
        func_impls_ << "(\"";
        for (auto c : assembly->asm_template())
            func_impls_ << handle_string_character(c);
        func_impls_ << "\"";

        // emit the outputs
        const char* separator = " : ";
        const auto& output_constraints = assembly->output_constraints();
        for (size_t i = 0; i < output_constraints.size(); ++i) {
            func_impls_ << separator << "\"" << output_constraints[i] << "\"("
                << outputs[i] << ")";
            separator = ", ";
        }

        // emit the inputs
        separator = output_constraints.empty() ? " :: " : " : ";
        auto input_constraints = assembly->input_constraints();
        for (size_t i = 0; i < input_constraints.size(); ++i) {
            func_impls_ << separator << "\"" << input_constraints[i] << "\"(";
            emit(assembly->op(i + 1)) << ")";
            separator = ", ";
        }

        // emit the clobbers
        separator = input_constraints.empty() ? output_constraints.empty() ? " ::: " : " :: " : " : ";
        for (auto clob : assembly->clobbers()) {
            func_impls_ << separator << "\"" << clob << "\"";
            separator = ", ";
        }
        return func_impls_ << ");";
#endif
    } else if (auto global = def->isa<Global>()) {
        assert(!global->init()->isa_continuation());
        if (global->is_mutable() && lang_ != Lang::C99)
            world().wdef(global, "{}: Global variable '{}' will not be synced with host", get_lang(), global);

        std::string prefix;
        switch (lang_) {
            default:                                   break;
            case Lang::CUDA:   prefix = "__device__ "; break;
            case Lang::OpenCL: prefix = "__constant "; break;
        }

        func_decls_.fmt("{}{} {}_slot", prefix, convert(global->alloced_type()), name);
        if (global->init()->isa<Bottom>())
            func_decls_.fmt("; // bottom\n");
        else
            func_decls_.fmt(" = {};\n", emit_constant(global->init()));
        func_decls_.fmt("{}{} {} = &{}_slot;\n", prefix, convert(global->type()), name, name);
    } else if (auto select = def->isa<Select>()) {
        auto cond = emit(select->cond());
        auto tval = emit(select->tval());
        auto fval = emit(select->fval());
        func_impls_.fmt("{} {}\n;", convert(select->type()), name);
        bb.body.fmt("{} {} = {} ? {} : {};", name, cond, tval, fval);
    } else {
        THORIN_UNREACHABLE;
    }

    return name;
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
                        s.fmt("__launch_bounds__ ({} * {} * {})", std::get<0>(block), std::get<1>(block), std::get<2>(block));
                }
                break;
            case Lang::OpenCL:
                s << "__kernel ";
                if (!is_proto && config != kernel_config_.end()) {
                    auto block = config->second->as<GPUKernelConfig>()->block_size();
                    if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0)
                        s.fmt("__attribute__((reqd_work_group_size({} * {} * {})))", std::get<0>(block), std::get<1>(block), std::get<2>(block));
                }
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
        if (is_mem(param) || is_unit(param) || param->order() > 0) {
            defs_[param] = {};
            continue;
        }
        if (needs_comma) s.fmt(", ");

        // TODO: This should go in favor of a prepare pass that rewrites the kernel parameters
        if (lang_ == Lang::OpenCL && cont->is_exported() && is_passed_via_buffer(param)) {
            // OpenCL structs are passed via buffer; the parameter is a pointer to this buffer
            s << "__global " << convert(param->type()) << "* " << param->unique_name() << "_";
        } else if (lang_ == Lang::HLS && cont->is_exported() && param->type()->isa<PtrType>()) {
            auto array_size = config->as<HLSKernelConfig>()->param_size(param);
            assert(array_size > 0);
            s.fmt("{} {}[{}]", convert(pointee_or_elem_type(param->type()->as<PtrType>())), param->unique_name(), array_size);
        } else {
            std::string qualifier;
            if (cont->is_exported() && (lang_ == Lang::OpenCL || lang_ == Lang::CUDA) &&
                config && config->as<GPUKernelConfig>()->has_restrict() &&
                param->type()->isa<PtrType>())
            {
                qualifier = lang_ == Lang::CUDA ? " __restrict" : " restrict";
            }
            s << convert(param->type()) << qualifier << " " << param->unique_name();
        }
        needs_comma = true;
    }
    s << ")";
    return s.str();
}

std::string CCodeGen::emit_fun_decl(Continuation* cont) {
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
        if (!cont->is_imported() && !cont->is_exported())
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

static inline bool is_const_primop(const Def* def) {
    return def->isa<PrimOp>() && !def->has_dep(Dep::Param);
}

// TODO do we need this?
const std::string CCodeGen::var_name(const Def* def) {
    if (is_const_primop(def))
        return def->unique_name() + "_" + std::to_string(primop_counter++);
    else
        return def->unique_name();
}

const std::string CCodeGen::get_lang() const {
    switch (lang_) {
        default:
        case Lang::C99:    return "C99";
        case Lang::HLS:    return "HLS";
        case Lang::CUDA:   return "CUDA";
        case Lang::OpenCL: return "OpenCL";
    }
}

inline std::string make_identifier(const std::string& str) {
    auto copy = str;
    std::transform(copy.begin(), copy.end(), copy.begin(), [] (auto c) {
        if (c == ' ') return '_';
        if (c == '*') return 'p';
        return c;
    });
    return copy;
}

// TODO do we need this?
std::string CCodeGen::type_name(const Type* type) {
    if (type->is_nominal())
        return type->as<NominalType>()->name().str();
    return make_identifier(convert(type)); // TODO especially this invocation of convert looks scary
}

std::string CCodeGen::array_name(const DefiniteArrayType* array_type) {
    return "array_" + std::to_string(array_type->dim()) + "_" + type_name(array_type->elem_type());
}

std::string CCodeGen::tuple_name(const TupleType* tuple_type) {
    std::string name = "tuple";
    for (auto op : tuple_type->ops())
        name += "_" + type_name(op);
    return name;
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
