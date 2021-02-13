#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/stream.h"
#include "thorin/be/c.h"

#include <cmath>
#include <sstream>
#include <type_traits>
#include <cctype>
#include <variant>

namespace thorin {

class CCodeGen;

struct CEmit : public Streamable<CEmit> {
    CEmit(CCodeGen* cg, const Def* def)
        : cg_(cg)
        , def_(def)
    {}
    CEmit(CCodeGen* cg, const Type* type)
        : cg_(cg)
        , def_(type)
    {}

    Stream& stream(Stream&) const;

private:
    CCodeGen* cg_;
    std::variant<const Def*, const Type*> def_; // we can remove this variant once we nuked Type in favor of just using Def everywhere
};

class CCodeGen {
public:
    CCodeGen(World& world, const Cont2Config& kernel_config, Stream& stream, Lang lang, bool debug)
        : world_(world)
        , kernel_config_(kernel_config)
        , lang_(lang)
        , fn_mem_(world.fn_type({world.mem_type()}))
        , debug_(debug)
        , stream_(stream)
    {}

    void emit();
    void emit_c_int();
    World& world() const { return world_; }

private:
    CEmit wrap(const Def* def) { return CEmit(this, def); }
    CEmit wrap(const Type* type) { return CEmit(this, type); }
    template<class T>
    std::enable_if_t<!std::is_pointer_v<T>, Array<CEmit>> wrap(const T& t) {
        return {std::distance(t.begin(), t.end(), [&](const auto& x) { return CEmit(this, x); })};
    }
    Stream& emit_aggop_defs(const Def*);
    Stream& emit_aggop_decl(const Type*);
    Stream& emit_debug_info(const Def*);
    Stream& emit_addr_space(Stream&, const Type*);
    Stream& emit_string(const Global*);
    Stream& emit_temporaries(const Def*);
    Stream& emit_type(Stream&, const Type*);
    Stream& emit(const Def*);

    template <typename T, typename IsInfFn, typename IsNanFn>
    Stream& emit_float(T, IsInfFn, IsNanFn);

    // TODO use Symbol instead of std::string
    bool lookup(const Type*);
    bool lookup(const Def*);
    void insert(const Type*, std::string);
    void insert(const Def*, std::string);
    std::string& get_name(const Type*);
    std::string& get_name(const Def*);
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
    TypeMap<std::string> type2str_;
    DefMap<std::string> def2str_;
    DefMap<std::string> global2str_;
    DefMap<std::string> primop2str_;
    bool use_64_ = false;
    bool use_16_ = false;
    bool use_channels_ = false;
    bool debug_;
    int primop_counter = 0;
    Stream& stream_;
    Stream func_impl_;
    Stream func_decls_;
    Stream type_decls_;

    friend class CEmit;
};

Stream& CEmit::stream(Stream& s) const {
    if (std::holds_alternative<const Type*>(def_)) return cg_->emit_type(s, std::get<const Type*>(def_));
    return cg_->emit(std::get<const Def*>(def_));
}

// TODO
Stream& CCodeGen::emit_debug_info(const Def* /*def*/) {
    //if (debug_ && !def->loc().file.empty())
        //return streamf(func_impl_, "#line {} \"{}\"", def->loc().begin.row, def->loc().file) << endl;
    return func_impl_;
}

Stream& CCodeGen::emit_addr_space(Stream& s, const Type* type) {
    if (auto ptr = type->isa<PtrType>()) {
        if (lang_==Lang::OPENCL) {
            switch (ptr->addr_space()) {
                default:
                case AddrSpace::Generic:                   break;
                case AddrSpace::Global: s << "__global "; break;
                case AddrSpace::Shared: s << "__local ";  break;
            }
        }
    }

    return s;
}

inline bool is_string_type(const Type* type) {
    if (auto array = type->isa<DefiniteArrayType>())
        if (auto primtype = array->elem_type()->isa<PrimType>())
            if (primtype->primtype_tag() == PrimType_pu8)
                return true;
    return false;
}

std::string handle_string_character(char c) {
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

Stream& CCodeGen::emit_string(const Global* global) {
    if (auto str_array = global->init()->isa<DefiniteArray>()) {
        if (str_array->ops().back()->as<PrimLit>()->pu8_value() == pu8(0)) {
            if (auto primtype = str_array->elem_type()->isa<PrimType>()) {
                if (primtype->primtype_tag() == PrimType_pu8) {
                    std::string str = "\"";
                    for (auto op : str_array->ops().skip_back())
                        str += handle_string_character(op->as<PrimLit>()->pu8_value());
                    str += '"';
                    insert(global, str);
                }
            }
        }
    }

    return type_decls_;
}

Stream& CCodeGen::emit_type(Stream& s, const Type* type) {
    if (lookup(type))
        return s << get_name(type);

    if (type == nullptr) {
        return s << "NULL";
    } else if (type->isa<FrameType>()) {
        return s;
    } else if (type->isa<MemType>() || type == world().unit()) {
        return s << "void";
    } else if (type->isa<FnType>()) {
        THORIN_UNREACHABLE;
    } else if (auto tuple = type->isa<TupleType>()) {
        s.fmt("typedef struct {{\t\n");
        s.rangei(tuple->ops(), ";\n", [&](size_t i) { s.fmt("{} e{}", wrap(tuple->op(i)), i); });
        return s.fmt("\b\n}} {};", tuple_name(tuple));
    } else if (auto variant = type->isa<VariantType>()) {
        auto tag_type =
            variant->num_ops() < (UINT64_C(1) <<  8u) ? world_.type_qu8()  :
            variant->num_ops() < (UINT64_C(1) << 16u) ? world_.type_qu16() :
            variant->num_ops() < (UINT64_C(1) << 32u) ? world_.type_qu32() :
                                                        world_.type_qu64();
        s.fmt("typedef struct {{\t\n");
        // This is required because we have zero-sized types but C/C++ do not
        if (!std::all_of(variant->ops().begin(), variant->ops().end(), is_type_unit)) {
            s.fmt("union {{\t\n");
            s.rangei(variant->ops(), "\n;", [&](size_t i) {
                if (!is_type_unit(variant->op(i)))
                    s.fmt("{} {}", wrap(variant->op(i)), variant->op_name(i));
            });
            s.fmt("\b\n}} data;");
        }
        s.fmt("{} tag;\n", wrap(tag_type));
        return s.fmt("\b\n}} {};", variant->name());
    } else if (auto struct_type = type->isa<StructType>()) {
        s.fmt("typedef struct {{\t\n");
        size_t i = 0;
        s.range(struct_type->ops(), ";\n", [&](const Type* t) { s.fmt("{} e{}", wrap(t), struct_type->op_name(i)); });
        s.fmt("\b\n}} {};", tuple_name(tuple));

        if (struct_type->name().str().find("channel_") != std::string::npos)
            use_channels_ = true;
        return s;
    } else if (auto array = type->isa<IndefiniteArrayType>()) {
        // TODO I'm confused by the orignal code; it just emitted the elem_type Oo
        return s.fmt("{}[]", array->elem_type());
    } else if (auto array = type->isa<DefiniteArrayType>()) {
        return s.fmt("typedef struct {{\t\n{} e[{}];\b\n}} {};", wrap(array->elem_type()), array->dim(), array_name(array));
    } else if (auto ptr = type->isa<PtrType>()) {
        // TODO vector_length - I'm pretty sure the old code was incorrect anyway
        return s.fmt("{}*", wrap(ptr->pointee()));
    } else if (auto primtype = type->isa<PrimType>()) {
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
            case PrimType_pf16: case PrimType_qf16: s << "half";   use_16_ = true; break;
            case PrimType_pf32: case PrimType_qf32: s << "float";                  break;
            case PrimType_pf64: case PrimType_qf64: s << "double"; use_64_ = true; break;
        }
        // TODO vector_length - I'm pretty sure the old code was incorrect anyway
        return s;
    }
    THORIN_UNREACHABLE;
}

Stream& CCodeGen::emit_aggop_defs(const Def* /*def*/) {
#if 0
    if (lookup(def) || is_unit(def))
        return func_impl_;

    // look for nested array
    if (auto array = def->isa<DefiniteArray>()) {
        for (auto op : array->ops())
            emit_aggop_defs(op);
        if (lookup(def))
            return func_impl_;
        emit(array) << endl;
    }

    // look for nested struct
    if (auto agg = def->isa<Aggregate>()) {
        for (auto op : agg->ops())
            emit_aggop_defs(op);
        if (lookup(def))
            return func_impl_;
        emit(agg) << endl;
    }

    // look for nested variants
    if (auto variant = def->isa<Variant>()) {
        for (auto op : variant->ops())
            emit_aggop_defs(op);
        if (lookup(def))
            return func_impl_;
        emit(variant) << endl;
    }

    // emit declarations for bottom - required for nested data structures
    if (def->isa<Bottom>())
        emit(def) << endl;

#endif
    return func_impl_;
}

Stream& CCodeGen::emit_aggop_decl(const Type* /*type*/) {
#if 0
    if (lookup(type) || type == world().unit())
        return type_decls_;

    // set indent to zero
    auto indent = detail::get_indent();
    while (detail::get_indent() != 0)
        type_decls_ << down;

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
        emit_type(type_decls_, array) << endl;
        insert(type, array_name(array));
    }

    // look for nested tuple
    if (auto tuple = type->isa<TupleType>()) {
        for (auto op : tuple->ops())
            emit_aggop_decl(op);
        emit_type(type_decls_, tuple) << endl;
        insert(type, tuple_name(tuple));
    }

    // look for nested struct
    if (auto struct_type = type->isa<StructType>()) {
        for (auto op : struct_type->ops())
            emit_aggop_decl(op);
        emit_type(type_decls_, struct_type) << endl;
        insert(type, struct_type->name().str());
    }

    // look for nested variants
    if (auto variant = type->isa<VariantType>()) {
        for (auto op : variant->ops())
            emit_aggop_decl(op);
        emit_type(type_decls_, variant) << endl;
        insert(type, variant->name().str());
    }

    // restore indent
    while (detail::get_indent() != indent)
        type_decls_ << up;

#endif
    return type_decls_;
}

Stream& CCodeGen::emit_temporaries(const Def* /*def*/) {
#if 0
    // emit definitions of inlined elements, skip match
    if (!def->isa<PrimOp>() || !is_from_match(def->as<PrimOp>()))
        emit_aggop_defs(def);

    // emit temporaries for arguments
    if (def->order() >= 1 || is_mem(def) || is_unit(def) || lookup(def) || def->isa<PrimLit>())
        return func_impl_;

    return emit(def) << endl;
#endif
    return stream_;
}

void CCodeGen::emit() {
    if (lang_==Lang::CUDA) {
        for (auto x : std::array<char, 3>{'x', 'y', 'z'}) {
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
                emit(global) << endl;
            }
        }
    }

    Scope::for_each(world(), [&] (const Scope& scope) {
        if (scope.entry() == world().branch())
            return;

        // continuation declarations
        auto continuation = scope.entry();
        if (continuation->is_intrinsic())
            return;

        assert(continuation->is_returning());

        // retrieve return param
        const Param* ret_param = nullptr;
        for (auto param : continuation->params()) {
            if (param->order() != 0) {
                assert(!ret_param);
                ret_param = param;
            }
        }
        assert(ret_param);

        // emit function & its declaration
        auto ret_param_fn_type = ret_param->type()->as<FnType>();
        auto ret_type = ret_param_fn_type->num_ops() > 2 ? world_.tuple_type(ret_param_fn_type->ops().skip_front()) : ret_param_fn_type->ops().back();
        auto name = (continuation->is_exported() || continuation->empty()) ? continuation->name() : continuation->unique_name();
        if (continuation->is_exported()) {
            auto config = kernel_config_.find(continuation);
            switch (lang_) {
                default: break;
                case Lang::CUDA:
                   func_decls_ << "__global__ ";
                   func_impl_  << "__global__ ";
                   if (config != kernel_config_.end()) {
                       auto block = config->second->as<GPUKernelConfig>()->block_size();
                       if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0)
                           func_impl_ << "__launch_bounds__ (" << std::get<0>(block) << " * " << std::get<1>(block) << " * " << std::get<2>(block) << ") ";
                   }
                   break;
                case Lang::OPENCL:
                   func_decls_ << "__kernel ";
                   func_impl_  << "__kernel ";
                   if (config != kernel_config_.end()) {
                       auto block = config->second->as<GPUKernelConfig>()->block_size();
                       if (std::get<0>(block) > 0 && std::get<1>(block) > 0 && std::get<2>(block) > 0)
                           func_impl_ << "__attribute__((reqd_work_group_size(" << std::get<0>(block) << ", " << std::get<1>(block) << ", " << std::get<2>(block) << "))) ";
                   }
                   break;
            }
        } else {
            if (lang_==Lang::CUDA) {
                func_decls_ << "__device__ ";
                func_impl_  << "__device__ ";
            }
        }
        emit_aggop_decl(ret_type);
        emit_addr_space(func_decls_, ret_type);
        emit_addr_space(func_impl_,  ret_type);
        emit_type(func_decls_, ret_type) << " " << name << "(";
        emit_type(func_impl_,  ret_type) << " " << name << "(";
        size_t i = 0;
        std::string hls_pragmas;
        // emit and store all first-order params
        for (auto param : continuation->params()) {
            if (is_mem(param) || is_unit(param))
                continue;
            if (param->order() == 0) {
                emit_aggop_decl(param->type());
                if (is_texture_type(param->type())) {
                    // emit texture declaration for CUDA
                    type_decls_ << "texture<";
                    emit_type(type_decls_, param->type()->as<PtrType>()->pointee());
                    type_decls_ << ", cudaTextureType1D, cudaReadModeElementType> ";
                    type_decls_ << param->name() << ";" << endl;
                    insert(param, param->name().str());
                    // skip arrays bound to texture memory
                    continue;
                }
                if (i++ > 0) {
                    func_decls_ << ", ";
                    func_impl_  << ", ";
                }

                // get the kernel launch config
                KernelConfig* config = nullptr;
                if (continuation->is_exported()) {
                    auto config_it = kernel_config_.find(continuation);
                    assert(config_it != kernel_config_.end());
                    config = config_it->second.get();
                }

                if (lang_ == Lang::OPENCL && continuation->is_exported() &&
                    (param->type()->isa<DefiniteArrayType>() ||
                     param->type()->isa<StructType>() ||
                     param->type()->isa<TupleType>())) {
                    // structs are passed via buffer; the parameter is a pointer to this buffer
                    func_decls_ << "__global ";
                    func_impl_  << "__global ";
                    emit_type(func_decls_, param->type()) << " *";
                    emit_type(func_impl_,  param->type()) << " *" << param->unique_name() << "_";
                } else if (lang_ == Lang::HLS && continuation->is_exported() && param->type()->isa<PtrType>()) {
                    auto array_size = config->as<HLSKernelConfig>()->param_size(param);
                    assert(array_size > 0);
                    auto ptr_type = param->type()->as<PtrType>();
                    auto elem_type = ptr_type->pointee();
                    if (auto array_type = elem_type->isa<ArrayType>())
                        elem_type = array_type->elem_type();
                    emit_type(func_decls_, elem_type) << "[" << array_size << "]";
                    emit_type(func_impl_,  elem_type) << " " << param->unique_name() << "[" << array_size << "]";
                    if (elem_type->isa<StructType>() || elem_type->isa<DefiniteArrayType>())
                        hls_pragmas += "#pragma HLS data_pack variable=" + param->unique_name() + " struct_level\n";
                } else {
                    std::string qualifier;
                    // add restrict qualifier when possible
                    if ((lang_ == Lang::OPENCL || lang_ == Lang::CUDA) &&
                        config && config->as<GPUKernelConfig>()->has_restrict() &&
                        param->type()->isa<PtrType>()) {
                        qualifier = lang_ == Lang::CUDA ? " __restrict" : " restrict";
                    }
                    emit_addr_space(func_decls_, param->type());
                    emit_addr_space(func_impl_,  param->type());
                    emit_type(func_decls_, param->type()) << qualifier;
                    emit_type(func_impl_,  param->type()) << qualifier << " " << param->unique_name();
                }
                insert(param, param->unique_name());
            }
        }
        func_decls_ << ");" << endl;
        func_impl_  << ") {" << up;
        if (!hls_pragmas.empty())
            func_impl_ << down << endl << hls_pragmas << up;

        // OpenCL: load struct from buffer
        for (auto param : continuation->params()) {
            if (is_mem(param) || is_unit(param))
                continue;
            if (param->order() == 0) {
                if (lang_==Lang::OPENCL && continuation->is_exported() &&
                    (param->type()->isa<DefiniteArrayType>() ||
                     param->type()->isa<StructType>() ||
                     param->type()->isa<TupleType>())) {
                    func_impl_ << endl;
                    emit_type(func_impl_, param->type()) << " " << param->unique_name() << " = *" << param->unique_name() << "_;";
                }
            }
        }

        auto conts = schedule(scope);
        Scheduler scheduler(scope);

        // emit function arguments and phi nodes
        for (auto&& continuation : conts) {
            for (auto param : continuation->params()) {
                if (is_mem(param) || is_unit(param))
                    continue;
                emit_aggop_decl(param->type());
                insert(param, param->unique_name());
            }

            if (scope.entry() != continuation) {
                for (auto param : continuation->params()) {
                    if (!is_mem(param) && !is_unit(param)) {
                        func_impl_ << endl;
                        emit_addr_space(func_impl_, param->type());
                        emit_type(func_impl_, param->type()) << "  " << param->unique_name() << ";" << endl;
                        emit_addr_space(func_impl_, param->type());
                        emit_type(func_impl_, param->type()) << " p" << param->unique_name() << ";";
                    }
                }
            }
            // emit counter for pipeline intrinsic
            if (continuation->callee()->isa_continuation() &&
                continuation->callee()->as_continuation()->intrinsic() == Intrinsic::Pipeline) {
                func_impl_ << endl << "int i" << continuation->callee()->gid() << ";";
            }
        }

        for (auto cont : conts) {
            if (cont->empty()) continue;

            assert(cont == scope.entry() || cont->is_basicblock());
            func_impl_ << endl;

            // print label for the current basic block
            if (cont != scope.entry()) {
                func_impl_.fmt("l{};\t\n",  cont->gid());
                // load params from phi node
                for (auto param : cont->params())
                    if (!is_mem(param) && !is_unit(param))
                        func_impl_.fmt("{} = p{};\n", param->unique_name());
            }

            // TODO rewrite and merge with LLVM backend
#if 0
            for (auto primop : block) {
                if (primop->type()->order() >= 1) {
                    // ignore higher-order primops which come from a match intrinsic
                    if (is_from_match(primop))
                        continue;
                    THORIN_UNREACHABLE;
                }

                // struct/tuple/array declarations
                if (!primop->isa<MemOp>()) {
                    emit_aggop_decl(primop->type());
                    // search for inlined tuples/arrays
                    if (auto aggop = primop->isa<AggOp>()) {
                        if (!aggop->agg()->isa<MemOp>())
                            emit_aggop_decl(aggop->agg()->type());
                    }
                }

                // skip higher-order primops, stuff dealing with frames and all memory related stuff except stores
                if (primop->type()->isa<FnType>() || primop->type()->isa<FrameType>() || ((is_mem(primop) || is_unit(primop)) && !primop->isa<Store>()))
                    continue;

                emit_debug_info(primop);
                emit(primop) << endl;
            }
#endif

            // emit definitions for temporaries
            for (auto arg : cont->args())
                emit_temporaries(arg);

            // terminate bb
            if (cont->callee() == ret_param) { // return
                size_t num_args = cont->num_args();
                if (num_args == 0) func_impl_ << "return ;";
                else {
                    Array<const Def*> values(num_args);
                    Array<const Type*> types(num_args);

                    size_t n = 0;
                    for (auto arg : cont->args()) {
                        if (!is_mem(arg) && !is_unit(arg)) {
                            values[n] = arg;
                            types[n] = arg->type();
                            n++;
                        }
                    }

                    if (n == 0) func_impl_ << "return ;";
                    else if (n == 1) {
                        func_impl_ << "return ";
                        emit(values[0]) << ";";
                    } else {
                        types.shrink(n);
                        auto ret_type = world_.tuple_type(types);
                        auto ret_tuple_name = "ret_tuple" + std::to_string(cont->gid());
                        emit_aggop_decl(ret_type);
                        emit_type(func_impl_, ret_type) << " " << ret_tuple_name << ";";

                        for (size_t i = 0; i != n; ++i) {
                            func_impl_ << endl << ret_tuple_name << ".e" << i << " = ";
                            emit(values[i]) << ";";
                        }

                        func_impl_ << endl << "return " << ret_tuple_name << ";";
                    }
                }
            } else if (cont->callee() == world().branch()) {
                emit_debug_info(cont->arg(0)); // TODO correct?
                func_impl_.fmt("if ({}) {} else {}", wrap(cont->arg(0)), wrap(cont->arg(1)), wrap(cont->arg(2)));
            } else if (cont->callee()->isa<cont>() &&
                       cont->callee()->as<concont>()->intrinsic() == Intrinsic::Match) {
                func_impl_ << "switch (";
                emit(cont->arg(0)) << ") {" << up << endl;
                for (size_t i = 2; i < cont->num_args(); i++) {
                    auto arg = cont->arg(i)->as<Tuple>();
                    func_impl_ << "case ";
                    emit(arg->op(0)) << ": ";
                    emit(arg->op(1)) << endl;
                }
                func_impl_ << "default: ";
                emit(cont->arg(1));
                func_impl_ << down << endl << "}";
            } else if (cont->callee()->isa<Bottom>()) {
                func_impl_ << "return ; // bottom: unreachable";
            } else {
                auto store_phi = [&] (const Def* param, const Def* arg) {
                    func_impl_ << "p" << param->unique_name() << " = ";
                    emit(arg) << ";";
                };

                auto callee = cont->callee()->as_cont();
                emit_debug_info(callee);

                if (callee->is_basicblock()) {   // ordinary jump
                    assert(callee->num_params()==cont->num_args());
                    for (size_t i = 0, size = callee->num_params(); i != size; ++i)
                        if (!is_mem(callee->param(i)) && !is_unit(callee->param(i))) {
                            store_phi(callee->param(i), cont->arg(i));
                            func_impl_ << endl;
                        }
                    emit(callee);
                } else {
                    if (callee->is_intrinsic()) {
                        if (callee->intrinsic() == Intrinsic::Reserve) {
                            if (!cont->arg(1)->isa<PrimLit>())
                                EDEF(cont->arg(1), "reserve_shared: couldn't extract memory size");

                            switch (lang_) {
                                default:                                        break;
                                case Lang::CUDA:   func_impl_ << "__shared__ "; break;
                                case Lang::OPENCL: func_impl_ << "__local ";    break;
                            }

                            auto cont = cont->arg(2)->as_cont();
                            auto elem_type = cont->param(1)->type()->as<PtrType>()->pointee()->as<ArrayType>()->elem_type();
                            auto name = "reserver_" + cont->param(1)->unique_name();
                            emit_type(func_impl_, elem_type) << " " << name << "[";
                            emit(cont->arg(1)) << "];" << endl;
                            // store_phi:
                            func_impl_ << "p" << cont->param(1)->unique_name() << " = " << name << ";";
                            if (lang_ == Lang::HLS)
                                func_impl_ << endl
                                           << "#pragma HLS dependence variable=" << name << " inter false" << endl
                                           << "#pragma HLS data_pack  variable=" << name;
                        } else if (callee->intrinsic() == Intrinsic::Pipeline) {
                            assert((lang_ == Lang::OPENCL || lang_ == Lang::HLS) && "pipelining not supported on this backend");
                            // cast to cont to get unique name of "for index"
                            auto body = cont->arg(4)->as_cont();
                            if (lang_ == Lang::OPENCL) {
                                if (cont->arg(1)->as<PrimLit>()->value().get_s32() !=0) {
                                    func_impl_ << "#pragma ii ";
                                    emit(cont->arg(1)) << endl;
                                } else {
                                    func_impl_ << "#pragma ii 1"<< endl;
                                }
                            }
                            func_impl_ << "for (i" << callee->gid() << " = ";
                            emit(cont->arg(2));
                            func_impl_ << "; i" << callee->gid() << " < ";
                            emit(cont->arg(3)) <<"; i" << callee->gid() << "++) {"<< up << endl;
                            if (lang_ == Lang::HLS) {
                                if (cont->arg(1)->as<PrimLit>()->value().get_s32() != 0) {
                                    func_impl_ << "#pragma HLS PIPELINE II=";
                                    emit(cont->arg(1)) << endl;
                                } else {
                                    func_impl_ << "#pragma HLS PIPELINE"<< endl;
                                }
                            }
                            // emit body and "for index" as the "body parameter"
                            func_impl_ << "p" << body->param(1)->unique_name() << " = i"<< callee->gid()<< ";" << endl;
                            emit(body);
                            // emit "continue" with according label used for goto
                            func_impl_ << down << endl << "l" << cont->arg(6)->gid() << ": continue;" << endl << "}" << endl;
                            if (cont->arg(5) == ret_param)
                                func_impl_ << "return;" << endl;
                            else
                                emit(cont->arg(5));
                        } else if (callee->intrinsic() == Intrinsic::PipelineContinue) {
                            func_impl_ << "goto l" << callee->gid() << ";" << endl;
                        } else {
                            THORIN_UNREACHABLE;
                        }
                    } else {
                        auto emit_call = [&] (const Param* param = nullptr) {
                            auto name = (callee->is_exported() || callee->empty()) ? callee->name() : callee->unique_name();
                            if (param)
                                emit(param) << " = ";
                            func_impl_ << name << "(";
                            // emit all first-order args
                            size_t i = 0;
                            for (auto arg : cont->args()) {
                                if (arg->order() == 0 && !(is_mem(arg) || is_unit(arg))) {
                                    if (i++ > 0)
                                        func_impl_ << ", ";
                                    emit(arg);
                                }
                            }
                            func_impl_ << ");";
                            if (param) {
                                func_impl_ << endl;
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
                            emit_type(func_impl_, ret_type) << " " << ret_tuple_name << ";" << endl << ret_tuple_name << " = ";
                            emit_call();

                            // store arguments to phi node
                            for (size_t i = 0; i != n; ++i)
                                func_impl_ << endl << "p" << values[i]->unique_name() << " = " << ret_tuple_name << ".e" << i << ";";
                        }
                    }
                }
            }
            if (cont != scope.entry())
                func_impl_ << down;
            primop2str_.clear();
        }
        func_impl_ << down << endl << "}" << endl << endl;
        def2str_.clear();
    });

    type2str_.clear();
    global2str_.clear();

    if (lang_==Lang::OPENCL) {
        if (use_channels_)
            stream_ << "#pragma OPENCL EXTENSION cl_intel_channels : enable" << endl;
        if (use_16_)
            stream_ << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" << endl;
        if (use_64_)
            stream_ << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << endl;
        if (use_channels_ || use_16_ || use_64_)
            stream_ << endl;
    }

    if (lang_==Lang::CUDA && use_16_) {
        stream_ << "#include <cuda_fp16.h>" << endl << endl;
        stream_ << "#if __CUDACC_VER_MAJOR__ > 8" << endl
            << "#define half __half_raw" << endl
            << "#endif" << endl << endl;
    }

    if (lang_==Lang::CUDA || lang_==Lang::HLS)
        stream_ << "extern \"C\" {" << endl;

    if (!type_decls_.str().empty())
        stream_ << type_decls_.str() << endl;
    if (!func_decls_.str().empty())
        stream_ << func_decls_.str() << endl;
    stream_ << func_impl_.str();

    if (lang_==Lang::CUDA || lang_==Lang::HLS)
        stream_ << "}"; // extern "C"
#endif
}

void CCodeGen::emit_c_int() {
#if 0
    // Do not emit C interfaces for definitions that are not used
    world().cleanup();

    for (auto cont : world().cont()) {
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
        emit_type(func_decls_, ret_type) << " " << cont->name() << "(";
        size_t i = 0;

        // emit and store all first-order params
        for (auto param : cont->params()) {
            if (is_mem(param) || is_unit(param))
                continue;
            if (param->order() == 0) {
                emit_aggop_decl(param->type());
                if (i++ > 0)
                    func_decls_ << ", ";

                emit_type(func_decls_, param->type());
                insert(param, param->unique_name());
            }
        }
        func_decls_ << ");" << endl;
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

    stream_ << "/* " << name << ": Artic interface file generated by thorin */" << endl;
    stream_ << "#ifndef " << guard << endl;
    stream_ << "#define " << guard << endl << endl;
    stream_ << "#ifdef __cplusplus" << endl;
    stream_ << "extern \"C\" {" << endl;
    stream_ << "#endif" << endl << endl;

    if (!type_decls_.str().empty())
        stream_ << type_decls_.str() << endl;
    if (!func_decls_.str().empty())
        stream_ << func_decls_.str() << endl;

    stream_ << "#ifdef __cplusplus" << endl;
    stream_ << "}" << endl;
    stream_ << "#endif" << endl << endl;
    stream_ << "#endif /* " << guard << " */";
#endif
}

template <typename T, typename IsInfFn, typename IsNanFn>
Stream& CCodeGen::emit_float(T t, IsInfFn is_inf, IsNanFn is_nan) {
#if 0
    auto float_mode = lang_ == Lang::CUDA ? std::scientific : std::hexfloat;
    const char* suf = "", * pref = "";

    if (std::is_same<T, half>::value) {
        if (lang_ == Lang::CUDA) {
            pref = "__float2half(";
            suf  = ")";
        } else {
            suf = "h";
        }
    }
    if (std::is_same<T, float>::value) {
        suf  = "f";
    }

    auto emit_nn = [&] (std::string def, std::string cuda, std::string opencl) {
        switch (lang_) {
            default:           func_impl_ << def;    break;
            case Lang::CUDA:   func_impl_ << cuda;   break;
            case Lang::OPENCL: func_impl_ << opencl; break;
        }
    };

    if (is_inf(t)) {
        if (std::is_same<T, half>::value) {
            emit_nn("std::numeric_limits<half>::infinity()", "__short_as_half(0x7c00)", "as_half(0x7c00)");
        } else if (std::is_same<T, float>::value) {
            emit_nn("std::numeric_limits<float>::infinity()", "__int_as_float(0x7f800000)", "as_float(0x7f800000)");
        } else {
            emit_nn("std::numeric_limits<double>::infinity()", "__longlong_as_double(0x7ff0000000000000LL)", "as_double(0x7ff0000000000000LL)");
        }
    } else if (is_nan(t)) {
        if (std::is_same<T, half>::value) {
            emit_nn("nan(\"\")", "__short_as_half(0x7fff)", "as_half(0x7fff)");
        } else if (std::is_same<T, float>::value) {
            emit_nn("nan(\"\")", "__int_as_float(0x7fffffff)", "as_float(0x7fffffff)");
        } else {
            emit_nn("nan(\"\")", "__longlong_as_double(0x7fffffffffffffffLL)", "as_double(0x7fffffffffffffffLL)");
        }
    } else {
        func_impl_ << float_mode << pref << t << suf;
    }
#endif
    return func_impl_;
}

Stream& CCodeGen::emit(const Def* def) {
    Stream s(std::cout); // TODO

    if (auto continuation = def->isa<Continuation>())
        return func_impl_.fmt("goto l{};", continuation->gid());

    if (lookup(def))
        return func_impl_ << get_name(def);

    auto def_name = var_name(def);

    if (auto bin = def->isa<BinOp>()) {
        if (is_type_unit(bin->lhs()->type()))
            return func_impl_;

        // emit definitions of inlined elements
        emit_aggop_defs(bin->lhs());
        emit_aggop_defs(bin->rhs());
        func_impl_.fmt("{} {};", wrap(bin->type()), def_name);

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
        }

        if (auto arithop = bin->isa<ArithOp>()) {
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

        insert(def, def_name);
        return func_impl_.fmt("{} = {} {} {};", wrap(bin->lhs()), op, wrap(bin->rhs()));
    }

#if 0
    if (auto conv = def->isa<ConvOp>()) {
        emit_aggop_defs(conv->from());
        auto src_type = conv->from()->type();
        auto dst_type = conv->type();
        auto src_ptr = src_type->isa<PtrType>();
        auto dst_ptr = dst_type->isa<PtrType>();

        // string handling: bitcast [n*pu8]* -> [pu8]*
        if (conv->from()->isa<Global>() && is_string_type(conv->from()->as<Global>()->init()->type())) {
            if (dst_ptr && dst_ptr->pointee()->isa<IndefiniteArrayType>()) {
                func_impl_ << "// skipped string bitcast: ";
                emit(conv->from());
                insert(def, get_name(conv->from()));
                return func_impl_;
            }
        }

        emit_addr_space(func_impl_, dst_type);
        emit_type(func_impl_, dst_type) << " " << def_name << ";" << endl;

        if (src_ptr && dst_ptr && src_ptr->addr_space() == dst_ptr->addr_space()) {
            func_impl_ << def_name << " = (";
            emit_addr_space(func_impl_, dst_type);
            emit_type(func_impl_, dst_type) << ")";
            emit(conv->from()) << ";";
            insert(def, def_name);
            return func_impl_;
        }

        if (conv->isa<Cast>()) {
            func_impl_ << def_name << " = ";

            auto from = src_type->as<PrimType>();
            auto to   = dst_type->as<PrimType>();

            if (lang_==Lang::CUDA && from && (from->primtype_tag() == PrimType_pf16 || from->primtype_tag() == PrimType_qf16)) {
                func_impl_ << "(";
                emit_type(func_impl_, dst_type) << ") __half2float(";
                emit(conv->from()) << ");";
            } else if (lang_==Lang::CUDA && to && (to->primtype_tag() == PrimType_pf16 || to->primtype_tag() == PrimType_qf16)) {
                func_impl_ << "__float2half((float)";
                emit(conv->from()) << ");";
            } else {
                func_impl_ << "(";
                emit_addr_space(func_impl_, dst_type);
                emit_type(func_impl_, dst_type) << ")";
                emit(conv->from()) << ";";
            }
        }

        if (conv->isa<Bitcast>()) {
            func_impl_ << "union { ";
            emit_addr_space(func_impl_, dst_type);
            emit_type(func_impl_, dst_type) << " dst; ";
            emit_addr_space(func_impl_, src_type);
            emit_type(func_impl_, src_type) << " src; ";
            func_impl_ << "} u" << def_name << ";" << endl;
            func_impl_ << "u" << def_name << ".src = ";
            emit(conv->from()) << ";" << endl;
            func_impl_ << def_name << " = u" << def_name << ".dst;";
        }

        insert(def, def_name);
        return func_impl_;
    }

    if (auto align_of = def->isa<AlignOf>()) {
        func_impl_ << "alignof(";
        emit_type(func_impl_, align_of->of()) << ")";
        return func_impl_;
    }

    if (auto size_of = def->isa<SizeOf>()) {
        func_impl_ << "sizeof(";
        emit_type(func_impl_, size_of->of()) << ")";
        return func_impl_;
    }

    if (auto array = def->isa<DefiniteArray>()) { // DefArray is mapped to a struct
        emit_aggop_decl(def->type());
        // emit definitions of inlined elements
        for (auto op : array->ops())
            emit_aggop_defs(op);

        emit_type(func_impl_, array->type()) << " " << def_name << ";" << endl << "{" << up << endl;
        emit_type(func_impl_, array->type()) << " " << def_name << "_tmp = { { ";
        for (size_t i = 0, e = array->num_ops(); i != e; ++i)
            emit(array->op(i)) << ", ";
        func_impl_ << "} };" << endl;
        func_impl_ << def_name << " = " << def_name << "_tmp;" << down << endl << "}";
        insert(def, def_name);
        return func_impl_;
    }

    if (auto agg = def->isa<Aggregate>()) {
        emit_aggop_decl(def->type());
        assert(def->isa<Tuple>() || def->isa<StructAgg>());
        // emit definitions of inlined elements
        for (auto op : agg->ops())
            emit_aggop_defs(op);

        emit_type(func_impl_, agg->type()) << " " << def_name << ";" << endl << "{" << up<< endl;
        emit_type(func_impl_, agg->type()) << " " << def_name << "_tmp = { " << up;
        for (size_t i = 0, e = agg->ops().size(); i != e; ++i) {
            func_impl_ << endl;
            emit(agg->op(i)) << ",";
        }
        func_impl_ << down << endl << "};" << endl;
        func_impl_ << def_name << " = " << def_name << "_tmp;" << down << endl << "}";
        insert(def, def_name);
        return func_impl_;
    }

    if (auto aggop = def->isa<AggOp>()) {
        emit_aggop_defs(aggop->agg());

        auto emit_access = [&] (const Def* def, const Def* index) -> Stream& {
            if (def->type()->isa<ArrayType>()) {
                func_impl_ << ".e[";
                emit(index) << "]";
            } else if (def->type()->isa<TupleType>()) {
                func_impl_ << ".e";
                emit(index);
            } else if (def->type()->isa<StructType>()) {
                func_impl_ << "." << def->type()->as<StructType>()->op_name(primlit_value<size_t>(index));
            } else if (def->type()->isa<VectorType>()) {
                if (is_primlit(index, 0))
                    func_impl_ << ".x";
                else if (is_primlit(index, 1))
                    func_impl_ << ".y";
                else if (is_primlit(index, 2))
                    func_impl_ << ".z";
                else if (is_primlit(index, 3))
                    func_impl_ << ".w";
                else {
                    func_impl_ << ".s";
                    emit(index);
                }
            } else {
                THORIN_UNREACHABLE;
            }
            return func_impl_;
        };

        if (auto extract = aggop->isa<Extract>()) {
            if (is_mem(extract) || extract->type()->isa<FrameType>())
                return func_impl_;
            if (!extract->agg()->isa<Assembly>()) { // extract is a nop for inline assembly
                emit_type(func_impl_, aggop->type()) << " " << def_name << ";" << endl;
                func_impl_ << def_name << " = ";
                if (auto memop = extract->agg()->isa<MemOp>())
                    emit(memop) << ";";
                else {
                    emit(aggop->agg());
                    emit_access(aggop->agg(), aggop->index()) << ";";
                }
            }
            insert(def, def_name);
            return func_impl_;
        }

        auto ins = def->as<Insert>();
        emit_type(func_impl_, aggop->type()) << " " << def_name << ";" << endl;
        func_impl_ << def_name << " = ";
        emit(ins->agg()) << ";" << endl;
        func_impl_ << def_name;
        emit_access(def, ins->index()) << " = ";
        emit(ins->value()) << ";";
        insert(def, def_name);
        return func_impl_;
    }

    if (auto primlit = def->isa<PrimLit>()) {
        switch (primlit->primtype_tag()) {
            case PrimType_bool:                     func_impl_ << (primlit->bool_value() ? "true" : "false");      break;
            case PrimType_ps8:  case PrimType_qs8:  func_impl_ << (int) primlit->ps8_value();                      break;
            case PrimType_pu8:  case PrimType_qu8:  func_impl_ << (unsigned) primlit->pu8_value();                 break;
            case PrimType_ps16: case PrimType_qs16: func_impl_ << primlit->ps16_value();                           break;
            case PrimType_pu16: case PrimType_qu16: func_impl_ << primlit->pu16_value();                           break;
            case PrimType_ps32: case PrimType_qs32: func_impl_ << primlit->ps32_value();                           break;
            case PrimType_pu32: case PrimType_qu32: func_impl_ << primlit->pu32_value();                           break;
            case PrimType_ps64: case PrimType_qs64: func_impl_ << primlit->ps64_value();                           break;
            case PrimType_pu64: case PrimType_qu64: func_impl_ << primlit->pu64_value();                           break;
            case PrimType_pf16: case PrimType_qf16: emit_float<half>(primlit->pf16_value(),
                                                                     [](half v) { return half_float::isinf(v); },
                                                                     [](half v) { return half_float::isnan(v); }); break;
            case PrimType_pf32: case PrimType_qf32: emit_float<float>(primlit->pf32_value(),
                                                                      [](float v) { return std::isinf(v); },
                                                                      [](float v) { return std::isnan(v); });      break;
            case PrimType_pf64: case PrimType_qf64: emit_float<double>(primlit->pf64_value(),
                                                                       [](double v) { return std::isinf(v); },
                                                                       [](double v) { return std::isnan(v); });    break;
        }
        return func_impl_;
    }

    if (auto variant = def->isa<Variant>()) {
        emit_type(func_impl_, variant->type()) << " " << def_name << ";" << endl;
        func_impl_ << "{" << up << endl;
        emit_type(func_impl_, variant->type()) << " " << def_name << "_tmp;" << endl;
        if (!is_type_unit(variant->op(0)->type())) {
            auto variant_type = variant->type()->as<VariantType>();
            func_impl_ << def_name << "_tmp.data." << variant_type->op_name(variant->index()) << " = ";
            emit(variant->op(0)) << ";" << endl;
        }
        func_impl_
            << def_name << "_tmp.tag = " << variant->index() << ";" << endl
            << def_name << " = " << def_name << "_tmp;" << down << endl
            << "}";
        insert(def, def_name);
        return func_impl_;
    }

    if (auto variant_index = def->isa<VariantIndex>()) {
        emit_type(func_impl_, variant_index->type()) << " " << def_name << ";" << endl;
        func_impl_ << def_name << " = ";
        emit(variant_index->op(0)) << ".tag" << ";";
        insert(def, def_name);
        return func_impl_;
    }

    if (auto variant_extract = def->isa<VariantExtract>()) {
        emit_type(func_impl_, variant_extract->type()) << " " << def_name << ";" << endl;
        func_impl_ << def_name << " = ";
        auto variant_type = variant_extract->value()->type()->as<VariantType>();
        emit(variant_extract->op(0)) << ".data." << variant_type->op_name(variant_extract->index()) << ";";
        insert(def, def_name);
        return func_impl_;
    }

    if (auto bottom = def->isa<Bottom>()) {
        emit_addr_space(func_impl_, bottom->type());
        emit_type(func_impl_, bottom->type()) << " " << def_name << "; // bottom";
        insert(def, def_name);
        return func_impl_;
    }

    if (auto load = def->isa<Load>()) {
        emit_type(func_impl_, load->out_val()->type()) << " " << def_name << ";" << endl;
        func_impl_ << def_name << " = ";
        // handle texture fetches
        if (!is_texture_type(load->ptr()->type()))
            func_impl_ << "*";
        emit(load->ptr()) << ";";

        insert(def, def_name);
        return func_impl_;
    }

    if (auto store = def->isa<Store>()) {
        emit_aggop_defs(store->val()) << "*";
        emit(store->ptr()) << " = ";
        emit(store->val()) << ";";

        insert(def, def_name);
        return func_impl_;
    }

    if (auto slot = def->isa<Slot>()) {
        emit_type(func_impl_, slot->alloced_type()) << " " << def_name << "_slot;" << endl;
        emit_type(func_impl_, slot->alloced_type()) << "* " << def_name << ";" << endl;
        func_impl_ << def_name << " = &" << def_name << "_slot;";
        insert(def, def_name);
        return func_impl_;
    }

    if (def->isa<Enter>())
        return func_impl_;

    if (def->isa<Vector>()) {
        THORIN_UNREACHABLE;
    }

    if (auto lea = def->isa<LEA>()) {
        emit_aggop_defs(lea->ptr());
        emit_aggop_defs(lea->index());
        if (is_texture_type(lea->type())) { // handle texture fetches
            emit_type(func_impl_, lea->ptr_pointee()) << " " << def_name << ";" << endl;
            func_impl_ << def_name << " = tex1Dfetch(";
            emit(lea->ptr()) << ", ";
            emit(lea->index()) << ");";
        } else if (lea->ptr_pointee()->isa<TupleType>()) {
            emit_type(func_impl_, lea->type()) << " " << def_name << ";" << endl;
            func_impl_ << def_name << " = &";
            emit(lea->ptr()) << "->e";
            emit(lea->index()) << ";";
        } else if (lea->ptr_pointee()->isa<StructType>()) {
            emit_type(func_impl_, lea->type()) << " " << def_name << ";" << endl;
            func_impl_ << def_name << " = &";
            emit(lea->ptr()) << "->";
            func_impl_ << lea->ptr_pointee()->isa<StructType>()->op_name(primlit_value<size_t>(lea->index())) << ";";
        } else if (lea->ptr_pointee()->isa<DefiniteArrayType>()) {
            emit_type(func_impl_, lea->type()) << " " << def_name << ";" << endl;
            func_impl_ << def_name << " = &";
            emit(lea->ptr()) << "->e[";
            emit(lea->index()) << "];";
        } else {
            emit_addr_space(func_impl_, lea->ptr()->type());
            emit_type(func_impl_, lea->type()) << " " << def_name << ";" << endl;
            func_impl_ << def_name << " = ";
            emit(lea->ptr()) << " + ";
            emit(lea->index()) << ";";
        }

        insert(def, def_name);
        return func_impl_;
    }

    if (auto assembly = def->isa<Assembly>()) {
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
            emit_type(func_impl_, assembly->type()->op(index)) << " " << name << ";" << endl;
        }
        // some outputs that were originally there might have been pruned because
        // they are not used but we still need them as operands for the asm
        // statement so we need to generate them here
        for (size_t i = 0; i < out_size; ++i) {
            if (outputs[i] == "") {
                auto name = var_name(assembly) + "_" + std::to_string(i + 1);
                emit_type(func_impl_, assembly->type()->op(i + 1)) << " " << name << ";" << endl;
                outputs[i] = name;
            }
        }

        // emit temporaries
        for (auto op : assembly->ops())
            emit_temporaries(op);

        func_impl_ << "asm ";
        if (assembly->has_sideeffects())
            func_impl_ << "volatile ";
        if (assembly->is_alignstack() || assembly->is_inteldialect())
            WDEF(assembly, "stack alignment and inteldialect flags unsupported for C output");
        func_impl_ << "(\"";
        for (auto c : assembly->asm_template())
            func_impl_ << handle_string_character(c);
        func_impl_ << "\"";

        // emit the outputs
        const char* separator = " : ";
        const auto& output_constraints = assembly->output_constraints();
        for (size_t i = 0; i < output_constraints.size(); ++i) {
            func_impl_ << separator << "\"" << output_constraints[i] << "\"("
                << outputs[i] << ")";
            separator = ", ";
        }

        // emit the inputs
        separator = output_constraints.empty() ? " :: " : " : ";
        auto input_constraints = assembly->input_constraints();
        for (size_t i = 0; i < input_constraints.size(); ++i) {
            func_impl_ << separator << "\"" << input_constraints[i] << "\"(";
            emit(assembly->op(i + 1)) << ")";
            separator = ", ";
        }

        // emit the clobbers
        separator = input_constraints.empty() ? output_constraints.empty() ? " ::: " : " :: " : " : ";
        for (auto clob : assembly->clobbers()) {
            func_impl_ << separator << "\"" << clob << "\"";
            separator = ", ";
        }
        return func_impl_ << ");";
    }

    if (auto global = def->isa<Global>()) {
        assert(!global->init()->isa_continuation() && "no global init continuation supported");

        if (global->is_mutable())
            WDEF(global, "{}: Global variable '{}' will not be synced with host", get_lang(), global);

        switch (lang_) {
            default:                                        break;
            case Lang::CUDA:   func_impl_ << "__device__ "; break;
            case Lang::OPENCL: func_impl_ << "__constant "; break;
        }
        bool bottom = global->init()->isa<Bottom>();
        if (!bottom)
            emit(global->init()) << endl;
        emit_type(func_impl_, global->alloced_type()) << " " << def_name << "_slot";
        if (bottom) {
            func_impl_ << "; // bottom";
        } else {
            func_impl_ << " = ";
            emit(global->init()) << ";";
        }
        func_impl_ << endl;

        switch (lang_) {
            default:                                        break;
            case Lang::CUDA:   func_impl_ << "__device__ "; break;
            case Lang::OPENCL: func_impl_ << "__constant "; break;
        }
        emit_type(func_impl_, global->alloced_type()) << " *" << def_name << " = &" << def_name << "_slot;";

        insert(def, def_name);
        return func_impl_;
    }

    if (auto select = def->isa<Select>()) {
        emit_aggop_defs(select->cond());
        emit_aggop_defs(select->tval());
        emit_aggop_defs(select->fval());
        emit_type(func_impl_, select->type()) << " " << def_name << ";" << endl;
        func_impl_ << def_name << " = ";
        emit(select->cond()) << " ? ";
        emit(select->tval()) << " : ";
        emit(select->fval()) << ";";
        insert(def, def_name);
        return func_impl_;
    }
#endif

    THORIN_UNREACHABLE;
}

bool CCodeGen::lookup(const Type* type) {
    return type2str_.contains(type);
}

bool CCodeGen::lookup(const Def* def) {
    if (def->isa<Global>())
        return global2str_.contains(def);
    else if (def->isa<PrimOp>() && is_const(def))
        return primop2str_.contains(def);
    else
        return def2str_.contains(def);
}

std::string& CCodeGen::get_name(const Type* type) {
    return *type2str_[type];
}

std::string& CCodeGen::get_name(const Def* def) {
    if (def->isa<Global>())
        return *global2str_[def];
    else if (def->isa<PrimOp>() && is_const(def))
        return *primop2str_[def];
    else
        return *def2str_[def];
}

const std::string CCodeGen::var_name(const Def* def) {
    if (def->isa<PrimOp>() && is_const(def))
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
        case Lang::OPENCL: return "OpenCL";
    }
}

void CCodeGen::insert(const Type* type, std::string str) {
    type2str_[type] = str;
}

void CCodeGen::insert(const Def* def, std::string str) {
    if (def->isa<Global>())
        global2str_[def] = str;
    else if (def->isa<PrimOp>() && is_const(def))
        primop2str_[def] = str;
    else
        def2str_[def] = str;
}

bool CCodeGen::is_texture_type(const Type* type) {
    if (auto ptr = type->isa<PtrType>()) {
        if (ptr->addr_space()==AddrSpace::Texture) {
            assert(lang_==Lang::CUDA && "Textures currently only supported in CUDA");
            return true;
        }
    }
    return false;
}

inline std::string make_identifier(const std::string& str) {
    auto copy = str;
    for (auto& c : copy) {
        if (c == ' ') c = '_';
    }
    return copy;
}

std::string CCodeGen::type_name(const Type* type) {
    std::stringstream os;
    emit_type(stream_, type);
    return make_identifier(std::string(os.str()));
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

void emit_c(World& world, const Cont2Config& kernel_config, Stream& stream, Lang lang, bool debug) {
    CCodeGen(world, kernel_config, stream, lang, debug).emit();
}

void emit_c_int(World& world, Stream& stream) {
    CCodeGen(world, {}, stream, Lang::C99, false).emit_c_int();
}

//------------------------------------------------------------------------------

}
