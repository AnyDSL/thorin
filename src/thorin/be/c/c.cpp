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
    bool is_valid(const std::string& s) { return !s.empty(); }
    std::string emit_fun_decl(Continuation*);
    std::string prepare(const Scope&);
    void prepare(Continuation*, const std::string&);
    void finalize(const Scope&);
    void finalize(Continuation*);

private:
    std::string convert(const Type*);
    Stream& emit_debug_info(Stream&, const Def*);

    Stream& emit_aggop_defs(const Def*);
    Stream& emit_aggop_decl(const Type*);
    Stream& emit_addr_space(Stream&, const Type*);
    Stream& emit_string(const Global*);
    Stream& emit_temporaries(const Def*);

    template <typename T, typename IsInfFn, typename IsNanFn>
    std::string emit_float(T, IsInfFn, IsNanFn);

    const std::string var_name(const Def*);
    const std::string get_lang() const;
    bool is_texture_type(const Type*);

    std::string type_name(const Type*);
    std::string array_name(const DefiniteArrayType*);
    std::string tuple_name(const TupleType*);

    World& world_;
    const Cont2Config& kernel_config_;
    Lang lang_;
    const FnType* fn_mem_;
    bool use_64_ = false;
    bool use_16_ = false;
    bool use_channels_ = false;
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

    if (type->isa<MemType>())        s << "<MemType>";
    else if (type->isa<FrameType>()) s <<  "<FrameType>";
    else if( type == world().unit()) s <<  "void";
    else if (auto primtype = type->isa<PrimType>()) {
        switch (primtype->primtype_tag()) {
            case PrimType_bool:                     s << "bool";                   break;
            case PrimType_ps8:  case PrimType_qs8:  s << "char";                   break;
            case PrimType_pu8:  case PrimType_qu8:  s << "unsigned char";          break;
            case PrimType_ps16: case PrimType_qs16: s << "short";                  break;
            case PrimType_pu16: case PrimType_qu16: s << "unsigned short";         break;
            case PrimType_ps32: case PrimType_qs32: s << "int";                    break;
            case PrimType_pu32: case PrimType_qu32: s << "unsigned int";           break;
            case PrimType_ps64: case PrimType_qs64: s << "long";                   break;
            case PrimType_pu64: case PrimType_qu64: s << "unsigned long";          break;
            case PrimType_pf32: case PrimType_qf32: s << "float";                  break;
            case PrimType_pf16: case PrimType_qf16: s << "half";   use_16_ = true; break;
            case PrimType_pf64: case PrimType_qf64: s << "double"; use_64_ = true; break;
            default: THORIN_UNREACHABLE;
        }
        if (primtype->is_vector())
            s << primtype->length();
    } else if (auto array = type->isa<IndefiniteArrayType>()) {
        return types_[type] = convert(array->elem_type()); // IndefiniteArrayType always occurs within a pointer
    } else if (type->isa<FnType>()) {
        assert(false && "todo");
    } else if (auto ptr = type->isa<PtrType>()) {
        s.fmt("{}*", convert(ptr->pointee()));
    } else if (auto array = type->isa<DefiniteArrayType>()) {
        name = array_name(array);
        auto elem_type = convert(array->elem_type());
        s.fmt("typedef struct {{\t\n{} e[{}];\b\n}} {};", elem_type, array->dim(), name);
    } else if (auto tuple = type->isa<TupleType>()) {
        name = tuple_name(tuple);
        s.fmt("typedef struct {{\t\n");
        s.rangei(tuple->ops(), "\n", [&](size_t i) { s.fmt("{} e{};", convert(tuple->op(i)), i); });
        s.fmt("\b\n}} {};\n", name);
    } else if (auto variant = type->isa<VariantType>()) {
        name = variant->name();
        auto tag_type =
            variant->num_ops() < (UINT64_C(1) <<  8u) ? world_.type_qu8()  :
            variant->num_ops() < (UINT64_C(1) << 16u) ? world_.type_qu16() :
            variant->num_ops() < (UINT64_C(1) << 32u) ? world_.type_qu32() :
                                                        world_.type_qu64();
        s.fmt("typedef struct {{\t\n");

        // This is required because we have zero-sized types but C/C++ do not
        if (!std::all_of(variant->ops().begin(), variant->ops().end(), is_type_unit)) {
            s.fmt("union {{\t\n");
            s.rangei(variant->ops(), "\n", [&](size_t i) {
                if (!is_type_unit(variant->op(i)))
                    s.fmt("{} {};", convert(variant->op(i)), variant->op_name(i));
            });
            s.fmt("\b\n}} data;");
        }

        s.fmt("{} tag;\n", convert(tag_type));
        s.fmt("\b\n}} {};", name);
    } else if (auto struct_type = type->isa<StructType>()) {
        name = type_name(struct_type);
        types_[struct_type] = name;
        s.fmt("typedef struct {{\t\n");
        size_t i = 0;
        s.range(struct_type->ops(), ";\n", [&](const Type* t) { s.fmt("{} e{}", convert(t), struct_type->op_name(i)); });
        s.fmt("\b\n}} {};", name);
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

#if 0
    // emit all globals
    for (auto primop : world().primops()) {
        if (auto global = primop->isa<Global>()) {
            if (is_string_type(global->init()->type()))
                emit_string(global);
            else {
                emit_aggop_decl(global->type());
                emit(global).endl();
            }
        }
    }
#endif

    Scope::for_each(world(), [&] (const Scope& scope) { emit_scope(scope); });

    if (lang_ == Lang::OpenCL) {
        if (use_channels_)
            stream_.fmt("#pragma OPENCL EXTENSION cl_intel_channels : enable\n");
        if (use_16_)
            stream_.fmt("#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");
        if (use_64_)
            stream_.fmt("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
        if (use_channels_ || use_16_ || use_64_)
            stream_.endl();
    }

    if (lang_ == Lang::CUDA && use_16_) {
        stream_.fmt("#include <cuda_fp16.h>\n\n");
        stream_.fmt("#if __CUDACC_VER_MAJOR__ > 8\n");
        stream_.fmt("#define half __half_raw\n");
        stream_.fmt("#endif\n\n");
    }

    if (lang_ == Lang::CUDA || lang_ == Lang::HLS) {
        stream_.fmt("extern \"C\" {{\t\n");
    }

    stream_ << type_decls_.str();
    stream_ << func_decls_.str();
    stream_ << func_impls_.str();

    if (lang_ == Lang::CUDA || lang_ == Lang::HLS)
        stream_.fmt("}} /* extern \"C\" */\n");
}

std::string CCodeGen::prepare(const Scope& scope) {
    StringStream s;

    auto cont = scope.entry();
    auto ret_param_type = cont->ret_param()->type()->as<FnType>();
    auto name = (cont->is_exported() || cont->empty()) ? cont->name() : cont->unique_name();

    // Convert the return type to a tuple if several values are returned
    auto ret_type = ret_param_type->num_ops() > 2
        ? world_.tuple_type(ret_param_type->ops().skip_front())
        : ret_param_type->ops().back();

    // Emit function qualifiers
    if (cont->is_exported()) {
        auto config = kernel_config_.find(cont);
        switch (lang_) {
            default: break;
            case Lang::CUDA:
                s << "__global__ ";
                if (config != kernel_config_.end()) {
                    auto block = config->second->as<GPUKernelConfig>()->block_size();
                    if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0)
                        s << "__launch_bounds__ (" << std::get<0>(block) << " * " << std::get<1>(block) << " * " << std::get<2>(block) << ") ";
                }
                break;
            case Lang::OpenCL:
                s << "__kernel ";
                if (config != kernel_config_.end()) {
                    auto block = config->second->as<GPUKernelConfig>()->block_size();
                    if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0)
                        s << "__attribute__((reqd_work_group_size(" << std::get<0>(block) << ", " << std::get<1>(block) << ", " << std::get<2>(block) << "))) ";
                }
                break;
        }
    } else if (lang_ == Lang::CUDA) {
        s << "__device__ ";
    } else {
        s << "static ";
    }

    s << convert(ret_type) << " " << name << "(";
    return s.str();
}

inline bool passed_via_buffer(const Param* param) {
    return
        param->type()->isa<DefiniteArrayType>() ||
        param->type()->isa<StructType>() ||
        param->type()->isa<TupleType>();
}

void CCodeGen::prepare(Continuation* cont, const std::string& func_prefix) {
    auto config = cont->is_exported() && kernel_config_.count(cont)
        ? kernel_config_.find(cont)->second.get() : nullptr;

    StringStream s;
    std::string hls_pragmas;

    // emit and store all first-order params
    bool needs_comma = false;
    for (size_t i = 0, n = cont->num_params(); i < n; ++i) {
        auto param = cont->param(i);
        if (is_mem(param) || is_unit(param)) {
            defs_[param] = "";
            continue;
        }
        if (param->order() == 0) {
            if (needs_comma) { s.fmt(", "); needs_comma = false; }

            // TODO
#if 0
            if (is_texture_type(param->type())) {
                auto pointee = convert(param->type()->as<PtrType>()->pointee());
                type_decls_.fmt("texture<{}, cudaTextureType1D, cudaReadModeElementType> {};\n", pointee, param->name());
                // TODO
                //insert(param, param->name());
                // skip arrays bound to texture memory
                continue;
            }
#endif

            if (lang_ == Lang::OpenCL && cont->is_exported() && passed_via_buffer(param)) {
                // OpenCL structs are passed via buffer; the parameter is a pointer to this buffer
                s << "__global " << convert(param->type()) << "* " << param->unique_name() << "_";
            } else if (lang_ == Lang::HLS && cont->is_exported() && param->type()->isa<PtrType>()) {
                // HLS requires to annotate the array size at compile-time
                auto array_size = config->as<HLSKernelConfig>()->param_size(param);
                auto ptr_type = param->type()->as<PtrType>();
                auto elem_type = ptr_type->pointee();
                if (auto array_type = elem_type->isa<ArrayType>())
                    elem_type = array_type->elem_type();
                assert(array_size > 0);
                s << convert(elem_type) << " " << param->unique_name() << "[" << array_size << "]";
                if (elem_type->isa<StructType>() || elem_type->isa<DefiniteArrayType>())
                    hls_pragmas += "#pragma HLS data_pack variable=" + param->unique_name() + " struct_level\n";
            } else {
                std::string qualifier;
                if (cont->is_exported() && (lang_ == Lang::OpenCL || lang_ == Lang::CUDA) &&
                    config && config->as<GPUKernelConfig>()->has_restrict() &&
                    param->type()->isa<PtrType>())
                {
                    qualifier = lang_ == Lang::CUDA ? " __restrict" : " restrict";
                }
#if 0
                emit_addr_space(func_decls_, param->type());
                emit_addr_space(func_impls_, param->type());
#endif
                s << convert(param->type()) << qualifier << " " << param->unique_name();
            }
            defs_[param] = param->unique_name();
            needs_comma = true;
        }
    }
    s.fmt(")");
    func_impls_.fmt("{}{} {{\t\n", func_prefix, s.str());
    func_decls_.fmt("{}{};\n", func_prefix, s.str());

    if (!hls_pragmas.empty())
        func_impls_.fmt("\b\n{}\t\n", hls_pragmas);

    // Load OpenCL structs from buffers
    for (auto param : cont->params()) {
        if (is_mem(param) || is_unit(param))
            continue;
        if (param->order() == 0) {
            if (lang_ == Lang::OpenCL && cont->is_exported() && passed_via_buffer(param))
                func_impls_.fmt("{} {} = *{}_;\n", convert(param->type()), param->unique_name(), param->unique_name());
        }
    }
}

void CCodeGen::finalize(const Scope&) {
    func_impls_.fmt("\b\n}}");
}

void CCodeGen::finalize(Continuation* cont) {
    auto&& bb = cont2bb_[cont];
    if (!bb->head.str().empty()) func_impls_.fmt("{}\n", bb->head.str());
    if (!bb->body.str().empty()) func_impls_.fmt("{}\n", bb->body.str());
    func_impls_ << bb->tail.str();
}

void CCodeGen::emit_epilogue(Continuation* cont) {
    auto&& bb = cont2bb_[cont];
    emit_debug_info(bb->tail, cont->arg(0));

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
            case 0: bb->tail.fmt("return;");               break;
            case 1: bb->tail.fmt("return {};", values[0]); break;
            default:
                auto tuple = convert(world().tuple_type(types));
                bb->tail.fmt("{} _result;\n", tuple);

                for (size_t i = 0, e = types.size(); i != e; ++i)
                    bb->tail.fmt("_result.e{} = {};\n", i, values[i]);
                bb->tail.fmt("return _result;");
                break;
        }
    } else if (cont->callee() == world().branch()) {
        auto c = emit(cont->arg(0));
        auto t = emit(cont->arg(1));
        auto f = emit(cont->arg(2));
        bb->tail.fmt("if ({}) goto {}; else goto {};", c, t, f);
    } else if (auto callee = cont->callee()->isa_continuation(); callee && callee->intrinsic() == Intrinsic::Match) {
        bb->tail.fmt("switch ({}) {{\t\n", emit(cont->arg(0)));

        for (size_t i = 2; i < cont->num_args(); i++) {
            auto arg = cont->arg(i)->as<Tuple>();
            auto value = emit(arg->op(0));
            auto label = emit(arg->op(1));
            bb->tail.fmt("case {}: goto {};\n", value, label);
        }

        bb->tail.fmt("default: goto {};\n", emit(cont->arg(1)));
        bb->tail.fmt("\b\n}}");
    } else if (cont->callee()->isa<Bottom>()) {
        bb->tail.fmt("return;  // bottom: unreachable");
    } else if (auto callee = cont->callee()->isa_continuation(); callee && callee->is_basicblock()) { // ordinary jump
#if 0
        auto store_phi = [&] (const Def* param, const Def* arg) {
            bb->tail << "p" << param->unique_name() << " = ";
            emit(arg) << ";";
        };

        auto callee = cont->callee()->as_cont();
        if (callee->is_basicblock()) {   // ordinary jump
            assert(callee->num_params()==cont->num_args());
            for (size_t i = 0, size = callee->num_params(); i != size; ++i)
                if (!is_mem(callee->param(i)) && !is_unit(callee->param(i))) {
                    store_phi(callee->param(i), cont->arg(i));
                    bb->tail << endl;
                }
            emit(callee);
        } else {
#endif
        // TODO phi
        bb->tail.fmt("goto {}", emit(callee));
    } else if (auto callee = cont->callee()->isa_continuation(); callee && callee->is_intrinsic()) {
#if 0
        if (callee->intrinsic() == Intrinsic::Reserve) {
            if (!cont->arg(1)->isa<PrimLit>())
                world().EDEF(cont->arg(1), "reserve_shared: couldn't extract memory size");

            switch (lang_) {
                case Lang::CUDA:   bb->tail.fmt("__shared__ "); break;
                case Lang::OpenCL: bb->tail.fmt("__local ");    break;
                default:                                        break;
            }

            auto cont = cont->arg(2)->as_cont();
            auto elem_type = cont->param(1)->type()->as<PtrType>()->pointee()->as<ArrayType>()->elem_type();
            auto name = "reserver_" + cont->param(1)->unique_name();
            convert(bb->tail, elem_type) << " " << name << "[";
            emit(cont->arg(1)) << "];" << endl;
            // store_phi:
            bb->tail << "p" << cont->param(1)->unique_name() << " = " << name << ";";
            if (lang_ == Lang::HLS)
                bb->tail << endl
                            << "#pragma HLS dependence variable=" << name << " inter false" << endl
                            << "#pragma HLS data_pack  variable=" << name;
        } else if (callee->intrinsic() == Intrinsic::Pipeline) {
            assert((lang_ == Lang::OpenCL || lang_ == Lang::HLS) && "pipelining not supported on this backend");
            // cast to cont to get unique name of "for index"
            auto body = cont->arg(4)->as_cont();
            if (lang_ == Lang::OpenCL) {
                if (cont->arg(1)->as<PrimLit>()->value().get_s32() !=0) {
                    bb->tail << "#pragma ii ";
                    emit(cont->arg(1)) << endl;
                } else {
                    bb->tail << "#pragma ii 1"<< endl;
                }
            }
            bb->tail << "for (i" << callee->gid() << " = ";
            emit(cont->arg(2));
            bb->tail << "; i" << callee->gid() << " < ";
            emit(cont->arg(3)) <<"; i" << callee->gid() << "++) {"<< up << endl;
            if (lang_ == Lang::HLS) {
                if (cont->arg(1)->as<PrimLit>()->value().get_s32() != 0) {
                    bb->tail << "#pragma HLS PIPELINE II=";
                    emit(cont->arg(1)) << endl;
                } else {
                    bb->tail << "#pragma HLS PIPELINE"<< endl;
                }
            }
            // emit body and "for index" as the "body parameter"
            bb->tail << "p" << body->param(1)->unique_name() << " = i"<< callee->gid()<< ";" << endl;
            emit(body);
            // emit "continue" with according label used for goto
            bb->tail << down << endl << "l" << cont->arg(6)->gid() << ": continue;" << endl << "}" << endl;
            if (cont->arg(5) == ret_param)
                bb->tail << "return;" << endl;
            else
                emit(cont->arg(5));
        } else if (callee->intrinsic() == Intrinsic::PipelineContinue) {
            bb->tail << "goto l" << callee->gid() << ";" << endl;
        } else {
            THORIN_UNREACHABLE;
        }
#endif
    } else { // function/closure call
#if 0
        auto emit_call = [&] (const Param* param = nullptr) {
            auto name = (callee->is_exported() || callee->empty()) ? callee->name() : callee->unique_name();
            if (param)
                emit(param) << " = ";
            bb->tail << name << "(";
            // emit all first-order args
            size_t i = 0;
            for (auto arg : cont->args()) {
                if (arg->order() == 0 && !(is_mem(arg) || is_unit(arg))) {
                    if (i++ > 0)
                        bb->tail << ", ";
                    emit(arg);
                }
            }
            bb->tail << ");";
            if (param) {
                bb->tail << endl;
                store_phi(param, param);
            }
        };

        const Def* ret_arg = 0;
        for (auto arg : cont->args()) {
            if (arg->order() != 0) {
                assert(!ret_arg);
                ret_arg = arg;
            }
        }

        // must be call + cont --- call + return has been removed by codegen_prepare
        auto succ = ret_arg->as_cont();
        size_t num_params = succ->num_params();

        size_t n = 0;
        Array<const Param*> values(num_params);
        Array<const Type*> types(num_params);
        for (auto param : succ->params()) {
            if (!is_mem(param) && !is_unit(param)) {
                values[n] = param;
                types[n] = param->type();
                n++;
            }
        }

        if (n == 0)
            emit_call();
        else if (n == 1)
            emit_call(values[0]);
        else {
            types.shrink(n);
            auto ret_type = world_.tuple_type(types);
            auto ret_tuple_name = "ret_tuple" + std::to_string(cont->gid());
            emit_aggop_decl(ret_type);
            convert(bb->tail, ret_type) << " " << ret_tuple_name << ";" << endl << ret_tuple_name << " = ";
            emit_call();

            // store arguments to phi node
            for (size_t i = 0; i != n; ++i)
                bb->tail << endl << "p" << values[i]->unique_name() << " = " << ret_tuple_name << ".e" << i << ";";
        }
#endif
    }
}

std::string CCodeGen::emit_bb(BB& bb, const Def* def) {
    //if (auto continuation = def->isa<Continuation>())
        //return func_impls_.fmt("goto l{};", continuation->gid());

    auto t = convert(def->type());
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

        bb.body.fmt("{} {} = {} {} {};", t, name, a, op, b);
    } else if (auto conv = def->isa<ConvOp>()) {
#if 0
        emit_aggop_defs(conv->from());
        auto src_type = conv->from()->type();
        auto dst_type = conv->type();
        auto src_ptr = src_type->isa<PtrType>();
        auto dst_ptr = dst_type->isa<PtrType>();

        // string handling: bitcast [n*pu8]* -> [pu8]*
        if (conv->from()->isa<Global>() && is_string_type(conv->from()->as<Global>()->init()->type())) {
            if (dst_ptr && dst_ptr->pointee()->isa<IndefiniteArrayType>()) {
                func_impls_ << "// skipped string bitcast: ";
                emit(conv->from());
                insert(def, get_name(conv->from()));
                return func_impls_;
            }
        }

        emit_addr_space(func_impls_, dst_type);
        convert(func_impls_, dst_type) << " " << def_name << ";" << endl;

        if (src_ptr && dst_ptr && src_ptr->addr_space() == dst_ptr->addr_space()) {
            func_impls_ << def_name << " = (";
            emit_addr_space(func_impls_, dst_type);
            convert(func_impls_, dst_type) << ")";
            emit(conv->from()) << ";";
            insert(def, def_name);
            return func_impls_;
        }

        if (conv->isa<Cast>()) {
            func_impls_ << def_name << " = ";

            auto from = src_type->as<PrimType>();
            auto to   = dst_type->as<PrimType>();

            if (lang_ == Lang::CUDA && from && (from->primtype_tag() == PrimType_pf16 || from->primtype_tag() == PrimType_qf16)) {
                func_impls_ << "(";
                convert(func_impls_, dst_type) << ") __half2float(";
                emit(conv->from()) << ");";
            } else if (lang_ == Lang::CUDA && to && (to->primtype_tag() == PrimType_pf16 || to->primtype_tag() == PrimType_qf16)) {
                func_impls_ << "__float2half((float)";
                emit(conv->from()) << ");";
            } else {
                func_impls_ << "(";
                emit_addr_space(func_impls_, dst_type);
                convert(func_impls_, dst_type) << ")";
                emit(conv->from()) << ";";
            }
        }

        if (conv->isa<Bitcast>()) {
            func_impls_ << "union { ";
            emit_addr_space(func_impls_, dst_type);
            convert(func_impls_, dst_type) << " dst; ";
            emit_addr_space(func_impls_, src_type);
            convert(func_impls_, src_type) << " src; ";
            func_impls_ << "} u" << def_name << ";" << endl;
            func_impls_ << "u" << def_name << ".src = ";
            emit(conv->from()) << ";" << endl;
            func_impls_ << def_name << " = u" << def_name << ".dst;";
        }

        insert(def, def_name);
        return func_impls_;
#endif
    } else if (auto align_of = def->isa<AlignOf>()) {
#if 0
        func_impls_ << "alignof(";
        convert(func_impls_, align_of->of()) << ")";
        return func_impls_;
#endif
    } else if (auto size_of = def->isa<SizeOf>()) {
#if 0
        func_impls_ << "sizeof(";
        convert(func_impls_, size_of->of()) << ")";
        return func_impls_;
#endif
    } else if (auto array = def->isa<DefiniteArray>()) { // DefArray is mapped to a struct
#if 0
        emit_aggop_decl(def->type());
        // emit definitions of inlined elements
        for (auto op : array->ops())
            emit_aggop_defs(op);

        convert(func_impls_, array->type()) << " " << def_name << ";" << endl << "{" << up << endl;
        convert(func_impls_, array->type()) << " " << def_name << "_tmp = { { ";
        for (size_t i = 0, e = array->num_ops(); i != e; ++i)
            emit(array->op(i)) << ", ";
        func_impls_ << "} };" << endl;
        func_impls_ << def_name << " = " << def_name << "_tmp;" << down << endl << "}";
        insert(def, def_name);
        return func_impls_;
#endif
    } else if (auto agg = def->isa<Aggregate>()) {
#if 0
        emit_aggop_decl(def->type());
        assert(def->isa<Tuple>() || def->isa<StructAgg>());
        // emit definitions of inlined elements
        for (auto op : agg->ops())
            emit_aggop_defs(op);

        convert(func_impls_, agg->type()) << " " << def_name << ";" << endl << "{" << up<< endl;
        convert(func_impls_, agg->type()) << " " << def_name << "_tmp = { " << up;
        for (size_t i = 0, e = agg->ops().size(); i != e; ++i) {
            func_impls_ << endl;
            emit(agg->op(i)) << ",";
        }
        func_impls_ << down << endl << "};" << endl;
        func_impls_ << def_name << " = " << def_name << "_tmp;" << down << endl << "}";
        insert(def, def_name);
        return func_impls_;
#endif
    } else if (auto aggop = def->isa<AggOp>()) {
#if 0
        emit_aggop_defs(aggop->agg());

        auto emit_access = [&] (const Def* def, const Def* index) -> Stream& {
            if (def->type()->isa<ArrayType>()) {
                func_impls_ << ".e[";
                emit(index) << "]";
            } else if (def->type()->isa<TupleType>()) {
                func_impls_ << ".e";
                emit(index);
            } else if (def->type()->isa<StructType>()) {
                func_impls_ << "." << def->type()->as<StructType>()->op_name(primlit_value<size_t>(index));
            } else if (def->type()->isa<VectorType>()) {
                if (is_primlit(index, 0))
                    func_impls_ << ".x";
                else if (is_primlit(index, 1))
                    func_impls_ << ".y";
                else if (is_primlit(index, 2))
                    func_impls_ << ".z";
                else if (is_primlit(index, 3))
                    func_impls_ << ".w";
                else {
                    func_impls_ << ".s";
                    emit(index);
                }
            } else {
                THORIN_UNREACHABLE;
            }
            return func_impls_;
        };

        if (auto extract = aggop->isa<Extract>()) {
            if (is_mem(extract) || extract->type()->isa<FrameType>())
                return func_impls_;
            if (!extract->agg()->isa<Assembly>()) { // extract is a nop for inline assembly
                convert(func_impls_, aggop->type()) << " " << def_name << ";" << endl;
                func_impls_ << def_name << " = ";
                if (auto memop = extract->agg()->isa<MemOp>())
                    emit(memop) << ";";
                else {
                    emit(aggop->agg());
                    emit_access(aggop->agg(), aggop->index()) << ";";
                }
            }
            insert(def, def_name);
            return func_impls_;
        }

        auto ins = def->as<Insert>();
        convert(func_impls_, aggop->type()) << " " << def_name << ";" << endl;
        func_impls_ << def_name << " = ";
        emit(ins->agg()) << ";" << endl;
        func_impls_ << def_name;
        emit_access(def, ins->index()) << " = ";
        emit(ins->value()) << ";";
        insert(def, def_name);
        return func_impls_;
#endif
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
    } else if (auto variant = def->isa<Variant>()) {
#if 0
        convert(func_impls_, variant->type()) << " " << def_name << ";" << endl;
        func_impls_ << "{" << up << endl;
        convert(func_impls_, variant->type()) << " " << def_name << "_tmp;" << endl;
        if (!is_type_unit(variant->op(0)->type())) {
            auto variant_type = variant->type()->as<VariantType>();
            func_impls_ << def_name << "_tmp.data." << variant_type->op_name(variant->index()) << " = ";
            emit(variant->op(0)) << ";" << endl;
        }
        func_impls_
            << def_name << "_tmp.tag = " << variant->index() << ";" << endl
            << def_name << " = " << def_name << "_tmp;" << down << endl
            << "}";
        insert(def, def_name);
        return func_impls_;
#endif
    } else if (auto variant_index = def->isa<VariantIndex>()) {
#if 0
        convert(func_impls_, variant_index->type()) << " " << def_name << ";" << endl;
        func_impls_ << def_name << " = ";
        emit(variant_index->op(0)) << ".tag" << ";";
        insert(def, def_name);
        return func_impls_;
#endif
    } else if (auto variant_extract = def->isa<VariantExtract>()) {
#if 0
        convert(func_impls_, variant_extract->type()) << " " << def_name << ";" << endl;
        func_impls_ << def_name << " = ";
        auto variant_type = variant_extract->value()->type()->as<VariantType>();
        emit(variant_extract->op(0)) << ".data." << variant_type->op_name(variant_extract->index()) << ";";
        insert(def, def_name);
        return func_impls_;
#endif
    } else if (auto bottom = def->isa<Bottom>()) {
#if 0
        emit_addr_space(func_impls_, bottom->type());
        convert(func_impls_, bottom->type()) << " " << def_name << "; // bottom";
        insert(def, def_name);
        return func_impls_;
#endif
    } else if (auto load = def->isa<Load>()) {
#if 0
        convert(func_impls_, load->out_val()->type()) << " " << def_name << ";" << endl;
        func_impls_ << def_name << " = ";
        // handle texture fetches
        if (!is_texture_type(load->ptr()->type()))
            func_impls_ << "*";
        emit(load->ptr()) << ";";

        insert(def, def_name);
        return func_impls_;
#endif
    } else if (auto store = def->isa<Store>()) {
#if 0
        emit_aggop_defs(store->val()) << "*";
        emit(store->ptr()) << " = ";
        emit(store->val()) << ";";

        insert(def, def_name);
        return func_impls_;
#endif
    } else if (auto slot = def->isa<Slot>()) {
#if 0
        convert(func_impls_, slot->alloced_type()) << " " << def_name << "_slot;" << endl;
        convert(func_impls_, slot->alloced_type()) << "* " << def_name << ";" << endl;
        func_impls_ << def_name << " = &" << def_name << "_slot;";
        insert(def, def_name);
        return func_impls_;
#endif
    } else if (auto enter = def->isa<Enter>()) {
        return emit_unsafe(enter->mem());
    } else if (auto lea = def->isa<LEA>()) {
#if 0
        emit_aggop_defs(lea->ptr());
        emit_aggop_defs(lea->index());
        if (is_texture_type(lea->type())) { // handle texture fetches
            convert(func_impls_, lea->ptr_pointee()) << " " << def_name << ";" << endl;
            func_impls_ << def_name << " = tex1Dfetch(";
            emit(lea->ptr()) << ", ";
            emit(lea->index()) << ");";
        } else if (lea->ptr_pointee()->isa<TupleType>()) {
            convert(func_impls_, lea->type()) << " " << def_name << ";" << endl;
            func_impls_ << def_name << " = &";
            emit(lea->ptr()) << "->e";
            emit(lea->index()) << ";";
        } else if (lea->ptr_pointee()->isa<StructType>()) {
            convert(func_impls_, lea->type()) << " " << def_name << ";" << endl;
            func_impls_ << def_name << " = &";
            emit(lea->ptr()) << "->";
            func_impls_ << lea->ptr_pointee()->isa<StructType>()->op_name(primlit_value<size_t>(lea->index())) << ";";
        } else if (lea->ptr_pointee()->isa<DefiniteArrayType>()) {
            convert(func_impls_, lea->type()) << " " << def_name << ";" << endl;
            func_impls_ << def_name << " = &";
            emit(lea->ptr()) << "->e[";
            emit(lea->index()) << "];";
        } else {
            emit_addr_space(func_impls_, lea->ptr()->type());
            convert(func_impls_, lea->type()) << " " << def_name << ";" << endl;
            func_impls_ << def_name << " = ";
            emit(lea->ptr()) << " + ";
            emit(lea->index()) << ";";
        }

        insert(def, def_name);
        return func_impls_;
#endif
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
#if 0
        assert(!global->init()->isa_continuation() && "no global init continuation supported");

        if (global->is_mutable())
            WDEF(global, "{}: Global variable '{}' will not be synced with host", get_lang(), global);

        switch (lang_) {
            default:                                        break;
            case Lang::CUDA:   func_impls_ << "__device__ "; break;
            case Lang::OpenCL: func_impls_ << "__constant "; break;
        }
        bool bottom = global->init()->isa<Bottom>();
        if (!bottom)
            emit(global->init()) << endl;
        convert(func_impls_, global->alloced_type()) << " " << def_name << "_slot";
        if (bottom) {
            func_impls_ << "; // bottom";
        } else {
            func_impls_ << " = ";
            emit(global->init()) << ";";
        }
        func_impls_ << endl;

        switch (lang_) {
            default:                                        break;
            case Lang::CUDA:   func_impls_ << "__device__ "; break;
            case Lang::OpenCL: func_impls_ << "__constant "; break;
        }
        convert(func_impls_, global->alloced_type()) << " *" << def_name << " = &" << def_name << "_slot;";

        insert(def, def_name);
        return func_impls_;
#endif
    } else if (auto select = def->isa<Select>()) {
        auto cond = emit(select->cond());
        auto tval = emit(select->tval());
        auto fval = emit(select->fval());
        bb.body.fmt("{} {} = {} ? {} : {};", t, name, cond, tval, fval);
    } else {
        //THORIN_UNREACHABLE;
    }

    return name;
}

std::string CCodeGen::emit_fun_decl(Continuation*) {
    return "TODO";
}

Stream& CCodeGen::emit_debug_info(Stream& s, const Def* def) {
    if (debug_ && !def->loc().file.empty())
        return s.fmt("#line {} \"{}\"\n", def->loc().begin.row, def->loc().file);
    return s;
}

Stream& CCodeGen::emit_addr_space(Stream& s, const Type* type) {
    if (auto ptr = type->isa<PtrType>()) {
        if (lang_ == Lang::OpenCL) {
            switch (ptr->addr_space()) {
                default:
                case AddrSpace::Generic:                  break;
                case AddrSpace::Global: s << "__global "; break;
                case AddrSpace::Shared: s << "__local ";  break;
            }
        }
    }

    return s;
}

Stream& CCodeGen::emit_string(const Global* global) {
    if (auto str_array = global->init()->isa<DefiniteArray>()) {
        if (str_array->ops().back()->as<PrimLit>()->pu8_value() == pu8(0)) {
            if (auto primtype = str_array->elem_type()->isa<PrimType>()) {
                if (primtype->primtype_tag() == PrimType_pu8) {
                    std::string str = "\"";
                    for (auto op : str_array->ops().skip_back())
                        str += handle_string_character(op->as<PrimLit>()->pu8_value());
                    str += '"';
                    // TODO
                    //insert(global, str);
                }
            }
        }
    }

    return type_decls_;
}

Stream& CCodeGen::emit_aggop_defs(const Def*) {
#if 0
    if (lookup(def) || is_unit(def))
        return func_impls_;

    // look for nested array
    if (auto array = def->isa<DefiniteArray>()) {
        for (auto op : array->ops())
            emit_aggop_defs(op);
        if (lookup(def))
            return func_impls_;
        emit(array).endl();
    }

    // look for nested struct
    if (auto agg = def->isa<Aggregate>()) {
        for (auto op : agg->ops())
            emit_aggop_defs(op);
        if (lookup(def))
            return func_impls_;
        emit(agg).endl();
    }

    // look for nested variants
    if (auto variant = def->isa<Variant>()) {
        for (auto op : variant->ops())
            emit_aggop_defs(op);
        if (lookup(def))
            return func_impls_;
        emit(variant).endl();
    }

    // emit declarations for bottom - required for nested data structures
    if (def->isa<Bottom>())
        emit(def).endl();

    return func_impls_;
#endif
    THORIN_UNREACHABLE;
}

Stream& CCodeGen::emit_aggop_decl(const Type*) {
#if 0
    if (lookup(type) || type == world().unit())
        return type_decls_;

    if (auto ptr = type->isa<PtrType>())
        emit_aggop_decl(ptr->pointee());

    if (auto array = type->isa<IndefiniteArrayType>())
        emit_aggop_decl(array->elem_type());

    if (auto fn = type->isa<FnType>())
        for (auto type : fn->ops())
            emit_aggop_decl(type);

    // look for nested array
    if (auto array = type->isa<DefiniteArrayType>()) {
        emit_aggop_decl(array->elem_type());
        convert(type_decls_, array).endl();
        insert(type, array_name(array));
    }

    // look for nested tuple
    if (auto tuple = type->isa<TupleType>()) {
        for (auto op : tuple->ops())
            emit_aggop_decl(op);
        convert(type_decls_, tuple).endl();
        insert(type, tuple_name(tuple));
    }

    // look for nested struct
    if (auto struct_type = type->isa<StructType>()) {
        for (auto op : struct_type->ops())
            emit_aggop_decl(op);
        convert(type_decls_, struct_type).endl();
        insert(type, struct_type->name().str());
    }

    // look for nested variants
    if (auto variant = type->isa<VariantType>()) {
        for (auto op : variant->ops())
            emit_aggop_decl(op);
        convert(type_decls_, variant).endl();
        insert(type, variant->name().str());
    }

    return type_decls_;
#endif
    THORIN_UNREACHABLE;
}

Stream& CCodeGen::emit_temporaries(const Def*) {
#if 0
    // emit definitions of inlined elements, skip match
    if (!def->isa<PrimOp>() || !is_from_match(def->as<PrimOp>()))
        emit_aggop_defs(def);

    // emit temporaries for arguments
    if (def->order() >= 1 || is_mem(def) || is_unit(def) || lookup(def) || def->isa<PrimLit>())
        return func_impls_;

    return emit(def).endl();
#endif
    THORIN_UNREACHABLE;
}

void CCodeGen::emit_c_int() {
#if 0
    // Do not emit C interfaces for definitions that are not used
    world().cleanup();

    for (auto cont : world().continuations()) {
        if (!cont->is_imported() && !cont->is_exported())
            continue;

        assert(cont->is_returning());

        // retrieve return param
        const Param* ret_param = nullptr;
        for (auto param : cont->params()) {
            if (param->order() != 0) {
                assert(!ret_param);
                ret_param = param;
            }
        }
        assert(ret_param);

        auto ret_param_fn_type = ret_param->type()->as<FnType>();
        auto ret_type = ret_param_fn_type->num_ops() > 2 ? world_.tuple_type(ret_param_fn_type->ops().skip_front()) : ret_param_fn_type->ops().back();
        if (cont->is_imported()) {
            // only emit types
            emit_aggop_decl(ret_type);
            for (auto param : cont->params()) {
                if (is_mem(param) || is_unit(param) || param->order() != 0)
                    continue;
                emit_aggop_decl(param->type());
            }
            continue;
        }

        // emit function declaration
        emit_aggop_decl(ret_type);
        convert(func_decls_, ret_type) << " " << cont->name() << "(";
        size_t i = 0;

        // emit and store all first-order params
        for (auto param : cont->params()) {
            if (is_mem(param) || is_unit(param))
                continue;
            if (param->order() == 0) {
                emit_aggop_decl(param->type());
                if (i++ > 0)
                    func_decls_ << ", ";

                convert(func_decls_, param->type());
                insert(param, param->unique_name());
            }
        }
        func_decls_.fmt(");\n");
    }

    size_t pos = world().name().find_last_of("\\/");
    pos = (pos == std::string::npos) ? 0 : pos + 1;
    auto guard = world().name().substr(pos) + ".h";
    auto name = world().name() + ".h";

    // Generate a valid include guard macro name
    if (!std::isalpha(guard[0]) && guard[0] != '_') guard.insert(guard.begin(), '_');
    transform(guard.begin(), guard.end(), guard.begin(), [] (char c) -> char {
        if (!std::isalnum(c)) return '_';
        return ::toupper(c);
    });
    guard[guard.length() - 2] = '_';

    stream_.fmt("/* {}: Artic interface file generated by thorin */\n", name);
    stream_.fmt("#ifndef {}\n", guard);
    stream_.fmt("#define {}\n\n", guard);
    stream_.fmt("#ifdef __cplusplus\n");
    stream_.fmt("extern \"C\" {\n");
    stream_.fmt("#endif\n\n");

    if (!type_decls_.str().empty())
        stream_ << type_decls_.str() << endl;
    if (!func_decls_.str().empty())
        stream_ << func_decls_.str() << endl;

    stream_.fmt("#ifdef __cplusplus\n");
    stream_.fmt("}\n");
    stream_.fmt("#endif\n\n");
    stream_.fmt("#endif /* {} */\n", guard);
#endif
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

bool CCodeGen::is_texture_type(const Type* type) {
    if (auto ptr = type->isa<PtrType>()) {
        if (ptr->addr_space()==AddrSpace::Texture) {
            assert(lang_ == Lang::CUDA && "Textures currently only supported in CUDA");
            return true;
        }
    }
    return false;
}

inline std::string make_identifier(const std::string& str) {
    auto copy = str;
    std::transform(copy.begin(), copy.end(), copy.begin(), [] (auto c) { return c == ' ' ? '_' : c; });
    return copy;
}

std::string CCodeGen::type_name(const Type* /*type*/) {
#if 0
    std::stringstream os;
    convert(stream_, type);
    return make_identifier(std::string(os.str()));
#endif
    return "";
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
