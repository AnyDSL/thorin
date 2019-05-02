#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/log.h"
#include "thorin/util/stream.h"
#include "thorin/be/c.h"

#include <cmath>
#include <sstream>
#include <type_traits>

namespace thorin {

class CCodeGen {
public:
    CCodeGen(World& world, const Cont2Config& kernel_config, std::ostream& stream, Lang lang, bool debug)
        : world_(world)
        , kernel_config_(kernel_config)
        , lang_(lang)
        , fn_mem_(world.fn_type({world.mem_type()}))
        , debug_(debug)
        , os_(stream)
    {}

    void emit();
    World& world() const { return world_; }

private:
    std::ostream& emit_aggop_defs(const Def*);
    std::ostream& emit_aggop_decl(const Type*);
    std::ostream& emit_debug_info(const Def*);
    std::ostream& emit_addr_space(std::ostream&, const Type*);
    std::ostream& emit_type(std::ostream&, const Type*);
    std::ostream& emit(const Def*);

    template <typename T, typename IsInfFn, typename IsNanFn>
    std::ostream& emit_float(T, IsInfFn, IsNanFn);

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
    std::ostream& os_;
    std::ostringstream func_impl_;
    std::ostringstream func_decls_;
    std::ostringstream type_decls_;
    std::ostringstream hls_top_;
    std::string hls_pragmas;

};


std::ostream& CCodeGen::emit_debug_info(const Def* def) {
    if (debug_ && def->location().filename())
        return streamf(func_impl_, "#line {} \"{}\"", def->location().front_line(), def->location().filename()) << endl;
    return func_impl_;
}

std::ostream& CCodeGen::emit_addr_space(std::ostream& os, const Type* type) {
    if (auto ptr = type->isa<PtrType>()) {
        if (lang_==Lang::OPENCL) {
            switch (ptr->addr_space()) {
                default:
                case AddrSpace::Generic:                   break;
                case AddrSpace::Global: os << "__global "; break;
                case AddrSpace::Shared: os << "__local ";  break;
            }
        }
    }

    return os;
}

inline bool is_string_type(const Type* type) {
    if (auto array = type->isa<DefiniteArrayType>())
        if (auto primtype = array->elem_type()->isa<PrimType>())
            if (primtype->primtype_tag() == PrimType_pu8)
                return true;
    return false;
}

inline bool is_channel_type(const StructType* struct_type) {
    return struct_type->name().str().find("channel_") != std::string::npos;
}

std::ostream& CCodeGen::emit_type(std::ostream& os, const Type* type) {
    if (lookup(type))
        return os << get_name(type);

    if (type == nullptr) {
        return os << "NULL";
    } else if (type->isa<FrameType>()) {
        return os;
    } else if (type->isa<MemType>() || type == world().unit()) {
        return os << "void";
    } else if (type->isa<FnType>()) {
        THORIN_UNREACHABLE;
    } else if (auto tuple = type->isa<TupleType>()) {
        os << "typedef struct {" << up;
        for (size_t i = 0, e = tuple->ops().size(); i != e; ++i) {
            os << endl;
            emit_type(os, tuple->op(i)) << " e" << i << ";";
        }
        os << down << endl << "} tuple_" << tuple->gid() << ";";
        return os;
    } else if (auto variant = type->isa<VariantType>()) {
        os << "union variant_" << variant->gid() << " {" << up;
        for (size_t i = 0, e = variant->ops().size(); i != e; ++i) {
            os << endl;
            emit_type(os, variant->op(i)) << " " << variant->op(i) << ";";
        }
        os << down << endl << "};";
        return os;
    } else if (auto struct_type = type->isa<StructType>()) {
        os << "typedef struct {" << up;
        for (size_t i = 0, e = struct_type->num_ops(); i != e; ++i) {
            os << endl;
            emit_type(os, struct_type->op(i)) << " e" << i << ";";
        }
        os << down << endl << "} struct_" << struct_type->name() << "_" << struct_type->gid() << ";";
        if (is_channel_type(struct_type))
            use_channels_ = true;
        return os;
    } else if (type->isa<Var>()) {
        THORIN_UNREACHABLE;
    } else if (auto array = type->isa<IndefiniteArrayType>()) {
        emit_type(os, array->elem_type());
        return os;
    } else if (auto array = type->isa<DefiniteArrayType>()) { // DefArray is mapped to a struct
        os << "typedef struct {" << up << endl;
        emit_type(os, array->elem_type()) << " e[" << array->dim() << "];";
        os << down << endl << "} array_" << array->gid() << ";";
        return os;
    } else if (auto ptr = type->isa<PtrType>()) {
        emit_type(os, ptr->pointee());
        os << '*';
        if (ptr->is_vector())
            os << vector_length(ptr->pointee());
        return os;
    } else if (auto primtype = type->isa<PrimType>()) {
        switch (primtype->primtype_tag()) {
            case PrimType_bool:                     os << "bool";                   break;
            case PrimType_ps8:  case PrimType_qs8:  os << "char";                   break;
            case PrimType_pu8:  case PrimType_qu8:  os << "unsigned char";          break;
            case PrimType_ps16: case PrimType_qs16: os << "short";                  break;
            case PrimType_pu16: case PrimType_qu16: os << "unsigned short";         break;
            case PrimType_ps32: case PrimType_qs32: os << "int";                    break;
            case PrimType_pu32: case PrimType_qu32: os << "unsigned int";           break;
            case PrimType_ps64: case PrimType_qs64: os << "long";                   break;
            case PrimType_pu64: case PrimType_qu64: os << "unsigned long";          break;
            case PrimType_pf16: case PrimType_qf16: os << "half";   use_16_ = true; break;
            case PrimType_pf32: case PrimType_qf32: os << "float";                  break;
            case PrimType_pf64: case PrimType_qf64: os << "double"; use_64_ = true; break;
        }
        if (primtype->is_vector())
            os << primtype->length();
        return os;
    }
    THORIN_UNREACHABLE;
}

std::ostream& CCodeGen::emit_aggop_defs(const Def* def) {
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

    return func_impl_;
}

std::ostream& CCodeGen::emit_aggop_decl(const Type* type) {
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
        insert(type, "array_" + std::to_string(type->gid()));
    }

    // look for nested tuple
    if (auto tuple = type->isa<TupleType>()) {
        for (auto op : tuple->ops())
            emit_aggop_decl(op);
        emit_type(type_decls_, tuple) << endl;
        insert(type, "tuple_" + std::to_string(type->gid()));
    }

    // look for nested struct
    if (auto struct_type = type->isa<StructType>()) {
        for (auto op : struct_type->ops())
            emit_aggop_decl(op);
        emit_type(type_decls_, struct_type) << endl;
        if (lang_ != Lang::HLS)
            insert(type, "struct_" + struct_type->name().str() + "_" + std::to_string(type->gid()));
        else if (is_channel_type(struct_type))
            insert(type,"    hls::stream<struct_" + struct_type->name().str() + "_" + std::to_string(type->gid()) + ">");

    }

    // look for nested variants
    if (auto variant = type->isa<VariantType>()) {
        for (auto op : variant->ops())
            emit_aggop_decl(op);
        emit_type(type_decls_, variant) << endl;
        insert(type, "union variant_" + std::to_string(type->gid()));
    }

    // restore indent
    while (detail::get_indent() != indent)
        type_decls_ << up;

    return type_decls_;
}

void CCodeGen::emit() {
    if (lang_==Lang::CUDA) {
        func_decls_ << "__device__ inline int threadIdx_x() { return threadIdx.x; }" << endl;
        func_decls_ << "__device__ inline int threadIdx_y() { return threadIdx.y; }" << endl;
        func_decls_ << "__device__ inline int threadIdx_z() { return threadIdx.z; }" << endl;
        func_decls_ << "__device__ inline int blockIdx_x() { return blockIdx.x; }" << endl;
        func_decls_ << "__device__ inline int blockIdx_y() { return blockIdx.y; }" << endl;
        func_decls_ << "__device__ inline int blockIdx_z() { return blockIdx.z; }" << endl;
        func_decls_ << "__device__ inline int blockDim_x() { return blockDim.x; }" << endl;
        func_decls_ << "__device__ inline int blockDim_y() { return blockDim.y; }" << endl;
        func_decls_ << "__device__ inline int blockDim_z() { return blockDim.z; }" << endl;
        func_decls_ << "__device__ inline int gridDim_x() { return gridDim.x; }" << endl;
        func_decls_ << "__device__ inline int gridDim_y() { return gridDim.y; }" << endl;
        func_decls_ << "__device__ inline int gridDim_z() { return gridDim.z; }" << endl;
    }

    // emit all globals
        for (auto primop : world().primops()) {
            if (auto global = primop->isa<Global>()) {
                // skip strings as they are emitted inline
                if (is_string_type(global->init()->type()))
                    continue;
                emit_aggop_decl(global->type());
                if(lang_ != Lang::HLS) {
                    emit(global) << endl;
                }
            }
        }

    // HLS top function
    if (lang_ == Lang::HLS) {
        enum io_type: bool {input, output};
        enum stream_lvl: char {source, mid, sink};
        io_type io = io_type::input;
        stream_lvl lvl = stream_lvl::source;
        std::string io_params[sizeof(io_type)+1] = "";
        hls_pragmas += "#pragma HLS DATAFLOW";
        size_t kernel_cnt = 0;
        hls_top_ << "void hls_top(";

        Scope::for_each(world(), [&] (const Scope& scope) {
            if (scope.entry() == world().branch())
                return;

            auto continuation = scope.entry();
            if (continuation->is_intrinsic())
                return;

            kernel_cnt++;
            for (auto param : continuation->params()) {
                KernelConfig* config = nullptr;
                if (continuation->is_external()) {
                    auto config_it = kernel_config_.find(continuation);
                    assert(config_it != kernel_config_.end());
                    config = config_it->second.get();
                    }
                if (param->type()->isa<PtrType>()) {
                    auto array_size = config->as<HLSKernelConfig>()->param_size(param);
                    assert(array_size > 0);
                    auto ptr_type = param->type()->as<PtrType>();
                    auto elem_type = ptr_type->pointee();
                    if (auto array_type = elem_type->isa<ArrayType>())
                        elem_type = array_type->elem_type();
                    // Top I/O ports(input,output)
                    emit_type(hls_top_,  elem_type) << " " << param->unique_name() << "[" << array_size << "]";
                    if (io_params[io].empty())
                        io_params[io] = param->unique_name();
                    if (io == input) {
                        hls_top_ << ", ";
                        io = io_type::output;
                    }
                }
            }
        });

        hls_top_ <<") {" << endl << up;
        if (!hls_pragmas.empty() && (kernel_cnt > 1)) {
            hls_top_ << down << hls_pragmas << endl;
        }
        hls_pragmas.clear();
        hls_pragmas += "#pragma HLS top name=AnyHLS\n";
        hls_pragmas += "#pragma HLS INTERFACE ap_ctrl_none port=return\n";
        for (auto param : io_params) {
            hls_pragmas += "#pragma HLS INTERFACE axis port=";
            hls_pragmas.append(param);
            hls_pragmas += " bundle=";
            if (io == output){
                hls_pragmas.append("input_s\n");
                io =io_type::input;
            }
            else
                hls_pragmas.append("output_s\n");

        }
        if (!hls_pragmas.empty())
            if (kernel_cnt == 1 )
                hls_top_ << down ;
        hls_top_ << hls_pragmas << up;

        for (auto primop : world().primops()) {
            if (auto global = primop->isa<Global>()) {
                // skip strings as they are emitted inline
                if (is_string_type(global->init()->type()))
                    continue;
                emit(global);
            }
        }
        hls_top_ << endl;

        Scope::for_each(world(), [&] (const Scope& scope) {
            auto continuation = scope.entry();
            if (continuation->is_intrinsic())
                return;

            auto kernel_name = (continuation->is_external() || continuation->empty()) ? continuation->name() : continuation->unique_name();
            // Functions calls
            if (lvl == source) {
            hls_top_ <<  kernel_name << "();" << endl;
            } else if (lvl == mid) {
            hls_top_ <<  kernel_name << "();" << endl;
            }  else {
            hls_top_ <<  kernel_name << "();" << endl;
            }
        });
        hls_top_ << down << endl << "}";
        hls_pragmas.clear();
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
        auto name = (continuation->is_external() || continuation->empty()) ? continuation->name() : continuation->unique_name();
        if (continuation->is_external()) {
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
                if (continuation->is_external()) {
                    auto config_it = kernel_config_.find(continuation);
                    assert(config_it != kernel_config_.end());
                    config = config_it->second.get();
                }

                if (lang_ == Lang::OPENCL && continuation->is_external() &&
                    (param->type()->isa<DefiniteArrayType>() ||
                     param->type()->isa<StructType>() ||
                     param->type()->isa<TupleType>())) {
                    // structs are passed via buffer; the parameter is a pointer to this buffer
                    func_decls_ << "__global ";
                    func_impl_  << "__global ";
                    emit_type(func_decls_, param->type()) << " *";
                    emit_type(func_impl_,  param->type()) << " *" << param->unique_name() << "_";
                } else if (lang_ == Lang::HLS && continuation->is_external() && param->type()->isa<PtrType>()) {
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
                if (lang_==Lang::OPENCL && continuation->is_external() &&
                    (param->type()->isa<DefiniteArrayType>() ||
                     param->type()->isa<StructType>() ||
                     param->type()->isa<TupleType>())) {
                    func_impl_ << endl;
                    emit_type(func_impl_, param->type()) << " " << param->unique_name() << " = *" << param->unique_name() << "_;";
                }
            }
        }

        Schedule schedule(scope);

        // emit function arguments and phi nodes
        for (const auto& block : schedule) {
            for (auto param : block.continuation()->params()) {
                if (is_mem(param) || is_unit(param))
                    continue;
                emit_aggop_decl(param->type());
                insert(param, param->unique_name());
            }

            auto continuation = block.continuation();
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
            // Emit counter for pipeline intrinsic
            if (continuation->callee()->isa_continuation() &&
                continuation->callee()->as_continuation()->intrinsic() == Intrinsic::Pipeline) {
                func_impl_ << endl << "int i" << continuation->callee()->gid() << ";";
            }
        }

        for (const auto& block : schedule) {
            auto continuation = block.continuation();
            if (continuation->empty())
                continue;
            assert(continuation == scope.entry() || continuation->is_basicblock());
            func_impl_ << endl;

            // print label for the current basic block
            if (continuation != scope.entry()) {
                func_impl_ << "l" << continuation->gid() << ": ;" << up << endl;
                // load params from phi node
                for (auto param : continuation->params())
                    if (!is_mem(param) && !is_unit(param))
                        func_impl_ << param->unique_name() << " = p" << param->unique_name() << ";" << endl;
            }

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

            for (auto arg : continuation->args()) {
                // emit definitions of inlined elements, skip match
                if (!arg->isa<PrimOp>() || !is_from_match(arg->as<PrimOp>()))
                    emit_aggop_defs(arg);

                // emit temporaries for arguments
                if (arg->order() >= 1 || is_mem(arg) || is_unit(arg) || lookup(arg) || arg->isa<PrimLit>())
                    continue;

                emit(arg) << endl;
            }

            // terminate bb
            if (continuation->callee() == ret_param) { // return
                size_t num_args = continuation->num_args();
                if (num_args == 0) func_impl_ << "return ;";
                else {
                    Array<const Def*> values(num_args);
                    Array<const Type*> types(num_args);

                    size_t n = 0;
                    for (auto arg : continuation->args()) {
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
                        auto ret_tuple_name = "ret_tuple" + std::to_string(continuation->gid());
                        emit_aggop_decl(ret_type);
                        emit_type(func_impl_, ret_type) << " " << ret_tuple_name << ";";

                        for (size_t i = 0; i != n; ++i) {
                            func_impl_ << endl << ret_tuple_name << ".e" << i << " = ";
                            emit(values[i]) << ";";
                        }

                        func_impl_ << endl << "return " << ret_tuple_name << ";";
                    }
                }
            } else if (continuation->callee() == world().branch()) {
                emit_debug_info(continuation->arg(0)); // TODO correct?
                func_impl_ << "if (";
                emit(continuation->arg(0));
                func_impl_ << ") ";
                emit(continuation->arg(1));
                func_impl_ << " else ";
                emit(continuation->arg(2));
            } else if (continuation->callee()->isa<Continuation>() &&
                       continuation->callee()->as<Continuation>()->intrinsic() == Intrinsic::Match) {
                func_impl_ << "switch (";
                emit(continuation->arg(0)) << ") {" << up << endl;
                for (size_t i = 2; i < continuation->num_args(); i++) {
                    auto arg = continuation->arg(i)->as<Tuple>();
                    func_impl_ << "case ";
                    emit(arg->op(0)) << ": ";
                    emit(arg->op(1)) << endl;
                }
                func_impl_ << "default: ";
                emit(continuation->arg(1));
                func_impl_ << down << endl << "}";
            } else if (continuation->callee()->isa<Bottom>()) {
                func_impl_ << "return ; // bottom: unreachable";
            } else {
                auto store_phi = [&] (const Def* param, const Def* arg) {
                    func_impl_ << "p" << param->unique_name() << " = ";
                    emit(arg) << ";";
                };

                auto callee = continuation->callee()->as_continuation();
                emit_debug_info(callee);

                if (callee->is_basicblock()) {   // ordinary jump
                    assert(callee->num_params()==continuation->num_args());
                    for (size_t i = 0, size = callee->num_params(); i != size; ++i)
                        if (!is_mem(callee->param(i)) && !is_unit(callee->param(i))) {
                            store_phi(callee->param(i), continuation->arg(i));
                            func_impl_ << endl;
                        }
                    emit(callee);
                } else {
                    if (callee->is_intrinsic()) {
                        if (callee->intrinsic() == Intrinsic::Reserve) {
                            if (!continuation->arg(1)->isa<PrimLit>())
                                EDEF(continuation->arg(1), "reserve_shared: couldn't extract memory size");

                            switch (lang_) {
                                default:                                        break;
                                case Lang::CUDA:   func_impl_ << "__shared__ "; break;
                                case Lang::OPENCL: func_impl_ << "__local ";    break;
                            }

                            auto cont = continuation->arg(2)->as_continuation();
                            auto elem_type = cont->param(1)->type()->as<PtrType>()->pointee()->as<ArrayType>()->elem_type();
                            auto name = "reserver_" + cont->param(1)->unique_name();
                            emit_type(func_impl_, elem_type) << " " << name << "[";
                            emit(continuation->arg(1)) << "];" << endl;
                            // store_phi:
                            func_impl_ << "p" << cont->param(1)->unique_name() << " = " << name << ";";
                            if (lang_ == Lang::HLS)
                                func_impl_ << endl
                                           << "#pragma HLS dependence variable=" << name << " inter false" << endl
                                           << "#pragma HLS data_pack  variable=" << name;
                        } else if (callee->intrinsic() == Intrinsic::Pipeline) {
                            assert((lang_ == Lang::OPENCL || lang_ == Lang::HLS) && "pipelining not supported on this backend");
                            // casting to contunation to get unique name of "for index"
                            auto body = continuation->arg(4)->as_continuation();
                            if (lang_ == Lang::OPENCL) {
                                if (continuation->arg(1)->as<PrimLit>()->value().get_s32() !=0) {
                                    func_impl_ << "#pragma ii ";
                                    emit(continuation->arg(1)) << endl;
                                } else {
                                    func_impl_ << "#pragma ii 1"<< endl;
                                }
                            }
                            func_impl_ << "for (i" << callee->gid() << " = ";
                            emit(continuation->arg(2));
                            func_impl_ << "; i" << callee->gid() << " < ";
                            emit(continuation->arg(3)) <<"; i" << callee->gid() << "++) {"<< up << endl;
                            if (lang_ == Lang::HLS) {
                                if (continuation->arg(1)->as<PrimLit>()->value().get_s32() != 0) {
                                    func_impl_ << "#pragma HLS PIPELINE II=";
                                    emit(continuation->arg(1)) << endl;
                                } else {
                                    func_impl_ << "#pragma HLS PIPELINE"<< endl;
                                }
                            }
                            // Emiting body and "for index" as the "body parameter"
                            func_impl_ << "p" << body->param(1)->unique_name() << " = i"<< callee->gid()<< ";" << endl;
                            emit(body);
                            // Emitting "continue" with accroding label used for goto
                            func_impl_ << down << endl << "l" << continuation->arg(6)->gid() << ": continue;" << endl << "}" << endl;
                            if (continuation->arg(5) == ret_param)
                                func_impl_ << "return;" << endl;
                            else
                                emit(continuation->arg(5));
                        } else if (callee->intrinsic() == Intrinsic::PipelineContinue) {
                            func_impl_ << "goto l" << callee->gid() << ";" << endl;
                        } else {
                            THORIN_UNREACHABLE;
                        }
                    } else {
                        auto emit_call = [&] (const Param* param = nullptr) {
                            auto name = (callee->is_external() || callee->empty()) ? callee->name() : callee->unique_name();
                            if (param)
                                emit(param) << " = ";
                            func_impl_ << name << "(";
                            // emit all first-order args
                            size_t i = 0;
                            for (auto arg : continuation->args()) {
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
                        for (auto arg : continuation->args()) {
                            if (arg->order() != 0) {
                                assert(!ret_arg);
                                ret_arg = arg;
                            }
                        }

                        // must be call + continuation --- call + return has been removed by codegen_prepare
                        auto succ = ret_arg->as_continuation();
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
                            auto ret_tuple_name = "ret_tuple" + std::to_string(continuation->gid());
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
            if (continuation != scope.entry())
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
            os_ << "#pragma OPENCL EXTENSION cl_intel_channels : enable" << endl;
        if (use_16_)
            os_ << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" << endl;
        if (use_64_)
            os_ << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << endl;
        if (use_channels_ || use_16_ || use_64_)
            os_ << endl;
    }

    if (lang_==Lang::CUDA && use_16_)
        os_ << "#include <cuda_fp16.h>" << endl << endl;

    if (lang_==Lang::CUDA || lang_==Lang::HLS) {
        if (lang_==Lang::HLS)
            os_ << "#include \"hls_stream.h\""<< endl << "#include \"hls_math.h\""<< endl;
        os_ << "extern \"C\" {" << endl;
    }
    if (!type_decls_.str().empty())
        os_ << type_decls_.str() << endl;
    if (!func_decls_.str().empty())
        os_ << func_decls_.str() << endl;
    os_ << func_impl_.str();
    if (!hls_top_.str().empty() && lang_==Lang::HLS)
        os_ << hls_top_.str() << endl;
    if (lang_==Lang::CUDA || lang_==Lang::HLS)
        os_ << "}"; // extern "C"
}

template <typename T, typename IsInfFn, typename IsNanFn>
std::ostream& CCodeGen::emit_float(T t, IsInfFn is_inf, IsNanFn is_nan) {
    auto float_mode = lang_ == Lang::CUDA ? std::scientific : std::hexfloat;
    const char* suf = "", * pref = "";

    if (lang_ == Lang::CUDA) {
        if (std::is_same<T, half>::value) {
            pref = "__float2half(";
            suf  = ")";
        } else if (std::is_same<T, float>::value) {
            suf  = "f";
        }
    } else if (std::is_same<T, half>::value) {
        suf = "h";
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
    return func_impl_;
}

std::ostream& CCodeGen::emit(const Def* def) {
    if (auto continuation = def->isa<Continuation>())
        return func_impl_ << "goto l" << continuation->gid() << ";";

    if (lookup(def))
        return func_impl_ << get_name(def);

    auto def_name = var_name(def);

    if (auto bin = def->isa<BinOp>()) {
        // emit definitions of inlined elements
        emit_aggop_defs(bin->lhs());
        emit_aggop_defs(bin->rhs());
        emit_type(func_impl_, bin->type()) << " " << def_name << ";" << endl;
        func_impl_ << def_name << " = ";
        emit(bin->lhs());
        if (auto cmp = bin->isa<Cmp>()) {
            switch (cmp->cmp_tag()) {
                case Cmp_eq: func_impl_ << " == "; break;
                case Cmp_ne: func_impl_ << " != "; break;
                case Cmp_gt: func_impl_ << " > ";  break;
                case Cmp_ge: func_impl_ << " >= "; break;
                case Cmp_lt: func_impl_ << " < ";  break;
                case Cmp_le: func_impl_ << " <= "; break;
            }
        }

        if (auto arithop = bin->isa<ArithOp>()) {
            switch (arithop->arithop_tag()) {
                case ArithOp_add: func_impl_ << " + ";  break;
                case ArithOp_sub: func_impl_ << " - ";  break;
                case ArithOp_mul: func_impl_ << " * ";  break;
                case ArithOp_div: func_impl_ << " / ";  break;
                case ArithOp_rem: func_impl_ << " % ";  break;
                case ArithOp_and: func_impl_ << " & ";  break;
                case ArithOp_or:  func_impl_ << " | ";  break;
                case ArithOp_xor: func_impl_ << " ^ ";  break;
                case ArithOp_shl: func_impl_ << " << "; break;
                case ArithOp_shr: func_impl_ << " >> "; break;
            }
        }
        emit(bin->rhs()) << ";";
        insert(def, def_name);
        return func_impl_;
    }

    if (auto conv = def->isa<ConvOp>()) {
        emit_aggop_defs(conv->from());
        auto src_type = conv->from()->type();
        auto dst_type = conv->type();

        // string handling: bitcast [n*pu8]* -> [pu8]*
        if (conv->isa<Bitcast>() && conv->from()->isa<Global>() && is_string_type(conv->from()->as<Global>()->init()->type())) {
            auto dst_ptr = dst_type->isa<PtrType>();
            if (dst_ptr && dst_ptr->pointee()->isa<IndefiniteArrayType>()) {
                func_impl_ << "// skipped string bitcast: ";
                emit(conv->from());
                insert(def, get_name(conv->from()));
                return func_impl_;
            }
        }

        emit_addr_space(func_impl_, dst_type);
        emit_type(func_impl_, dst_type) << " " << def_name << ";" << endl;

        if (conv->isa<Cast>()) {
            func_impl_ << def_name << " = ";

            if (src_type->isa<VariantType>()) {
                emit(conv->from()) << "." << dst_type << ";";
            } else {
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
        }

        if (conv->isa<Bitcast>()) {
            auto src_ptr = src_type->isa<PtrType>();
            auto dst_ptr = dst_type->isa<PtrType>();
            if (src_ptr && dst_ptr && src_ptr->addr_space() == dst_ptr->addr_space()) {
                func_impl_ << def_name << " = (";
                emit_addr_space(func_impl_, dst_type);
                emit_type(func_impl_, dst_type) << ")";
                emit(conv->from()) << ";";
            } else {
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
        }

        insert(def, def_name);
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

        emit_type(func_impl_, array->type()) << " " << def_name << ";" << endl << "{" << endl;
        emit_type(func_impl_, array->type()) << " " << def_name << "_tmp = { { ";
        for (size_t i = 0, e = array->num_ops(); i != e; ++i)
            emit(array->op(i)) << ", ";
        func_impl_ << "} };" << endl;
        func_impl_ << " " << def_name << " = " << def_name << "_tmp;" << endl << "}" << endl;
        insert(def, def_name);
        return func_impl_;
    }

    // aggregate operations
    {
        auto emit_access = [&] (const Def* def, const Def* index) -> std::ostream& {
            if (def->type()->isa<ArrayType>()) {
                func_impl_ << ".e[";
                emit(index) << "]";
            } else if (def->type()->isa<TupleType>() || def->type()->isa<StructType>()) {
                func_impl_ << ".e";
                emit(index);
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

        if (auto agg = def->isa<Aggregate>()) {
            emit_aggop_decl(def->type());
            assert(def->isa<Tuple>() || def->isa<StructAgg>());
            // emit definitions of inlined elements
            for (auto op : agg->ops())
                emit_aggop_defs(op);

            emit_type(func_impl_, agg->type()) << " " << def_name << ";" << endl << "{" << endl;
            emit_type(func_impl_, agg->type()) << " " << def_name << "_tmp = { " << up;
            for (size_t i = 0, e = agg->ops().size(); i != e; ++i) {
                func_impl_ << endl;
                emit(agg->op(i)) << ",";
            }
            func_impl_ << down << endl << "};" << endl;
            func_impl_ << " " << def_name << " = " << def_name << "_tmp;" << endl << "}" << endl;
            insert(def, def_name);
            return func_impl_;
        }

        if (auto aggop = def->isa<AggOp>()) {
            emit_aggop_defs(aggop->agg());

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
        func_impl_ << def_name << "." << variant->op(0)->type() << " = ";
        emit(variant->op(0)) << ";";
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
        } else {
            if (lea->ptr_pointee()->isa<TupleType>() || lea->ptr_pointee()->isa<StructType>()) {
                emit_type(func_impl_, lea->type()) << " " << def_name << ";" << endl;
                func_impl_ << def_name << " = &";
                emit(lea->ptr()) << "->e";
                emit(lea->index()) << ";";
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

        func_impl_ << "asm ";
        if (assembly->has_sideeffects())
            func_impl_ << "volatile ";
        if (assembly->is_alignstack() || assembly->is_inteldialect())
            WDEF(assembly, "stack alignment and inteldialect flags unsupported for C output");
        func_impl_ << "(\"" << assembly->asm_template() << "\"";

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

        // string handling
        if (auto str_array = global->init()->isa<DefiniteArray>()) {
            if (str_array->ops().back()->as<PrimLit>()->pu8_value() == pu8(0)) {
                if (auto primtype = str_array->elem_type()->isa<PrimType>()) {
                    if (primtype->primtype_tag() == PrimType_pu8) {
                        std::string str = "\"";
                        for (auto op : str_array->ops().skip_back())
                            str += op->as<PrimLit>()->pu8_value();
                        str += '"';
                        insert(def, str);
                        return func_impl_;
                    }
                }
            }
        }

        WDEF(global, "{}: Global variable '{}' will not be synced with host", get_lang(), global);
        switch (lang_) {
            default:                                        break;
            case Lang::CUDA:   func_impl_ << "__device__ "; break;
            case Lang::OPENCL: func_impl_ << "__constant "; break;
        }
        bool bottom = global->init()->isa<Bottom>();
        if (lang_ != Lang::HLS) {
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
        } else {
            if (!bottom)
                emit(global->init()) << endl;
            emit_type(hls_top_, global->alloced_type()) << " " << def_name <<";\n";
        }
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
        return type2str_[type];
}

std::string& CCodeGen::get_name(const Def* def) {
    if (def->isa<Global>())
        return global2str_[def];
    else if (def->isa<PrimOp>() && is_const(def))
        return primop2str_[def];
    else
        return def2str_[def];
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

//------------------------------------------------------------------------------

void emit_c(World& world, const Cont2Config& kernel_config, std::ostream& stream, Lang lang, bool debug) { CCodeGen(world, kernel_config, stream, lang, debug).emit(); }

//------------------------------------------------------------------------------

}
