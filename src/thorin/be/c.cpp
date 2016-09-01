#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/log.h"
#include "thorin/util/stream.h"
#include "thorin/be/c.h"

#include <sstream>

namespace thorin {

class CCodeGen {
public:
    CCodeGen(World& world, std::ostream& stream, Lang lang, bool debug)
        : world_(world)
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
    std::ostream& emit_bitcast(const Def*, const Def*);
    std::ostream& emit_addr_space(std::ostream&, const Type*);
    std::ostream& emit_type(std::ostream&, const Type*);
    std::ostream& emit(const Def*);
    bool lookup(const Type*);
    bool lookup(const Def*);
    void insert(const Type*, std::string);
    void insert(const Def*, std::string);
    std::string& get_name(const Type*);
    std::string& get_name(const Def*);
    bool is_texture_type(const Type*);

    World& world_;
    Lang lang_;
    const FnType* fn_mem_;
    TypeMap<std::string> type2str_;
    DefMap<std::string> def2str_;
    bool use_64_ = false;
    bool use_16_ = false;
    bool debug_;
    std::ostream& os_;
    std::ostringstream func_impl_;
    std::ostringstream func_decls_;
    std::ostringstream type_decls_;
};


std::ostream& CCodeGen::emit_debug_info(const Def* def) {
    if (debug_)
        return streamf(func_impl_, "#line % \"%\"", def->loc().begin().line(), def->loc().begin().filename()) << endl;
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

std::ostream& CCodeGen::emit_type(std::ostream& os, const Type* type) {
    if (type == nullptr) {
        return os << "NULL";
    } else if (type->isa<FrameType>()) {
        return os;
    } else if (type->isa<MemType>()) {
        return os << "void";
    } else if (type->isa<FnType>()) {
        THORIN_UNREACHABLE;
    } else if (auto tuple = type->isa<TupleType>()) {
        if (lookup(tuple))
            return os << get_name(tuple);
        os << "typedef struct tuple_" << tuple->gid() << " {" << up;
        for (size_t i = 0, e = tuple->args().size(); i != e; ++i) {
            os << endl;
            emit_type(os, tuple->arg(i)) << " e" << i << ";";
        }
        os << down << endl << "} tuple_" << tuple->gid() << ";";
        return os;
    } else if (auto struct_abs = type->isa<StructAbsType>()) {
        return os << struct_abs->name();
    } else if (auto struct_app = type->isa<StructAppType>()) {
        if (lookup(struct_app))
            return os << get_name(struct_app);
        os << "typedef struct struct_" << struct_app->gid() << " {" << up;
        for (size_t i = 0, e = struct_app->elems().size(); i != e; ++i) {
            os << endl;
            emit_type(os, struct_app->elem(i)) << " e" << i << ";";
        }
        os << down << endl << "} struct_" << struct_app->gid() << ";";
        return os;
    } else if (type->isa<TypeParam>()) {
        THORIN_UNREACHABLE;
    } else if (auto array = type->isa<IndefiniteArrayType>()) {
        emit_type(os, array->elem_type());
        return os;
    } else if (auto array = type->isa<DefiniteArrayType>()) { // DefArray is mapped to a struct
        if (lookup(array))
            return os << get_name(array);
        os << "typedef struct array_" << array->gid() << " {" << up << endl;
        emit_type(os, array->elem_type()) << " e[" << array->dim() << "];";
        os << down << endl << "} array_" << array->gid() << ";";
        return os;
    } else if (auto ptr = type->isa<PtrType>()) {
        emit_type(os, ptr->referenced_type());
        os << '*';
        if (ptr->is_vector())
            os << vector_length(ptr->referenced_type());
        return os;
    } else if (auto primtype = type->isa<PrimType>()) {
        switch (primtype->primtype_kind()) {
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
    if (lookup(def))
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

    // argument is a cast or bitcast
    if (auto conv = def->isa<ConvOp>())
        emit(conv) << endl;

    return func_impl_;
}


std::ostream& CCodeGen::emit_aggop_decl(const Type* type) {
    if (lookup(type))
        return type_decls_;

    // set indent to zero
    auto indent = detail::get_indent();
    while (detail::get_indent() != 0)
        type_decls_ << down;

    if (auto ptr = type->isa<PtrType>())
        emit_aggop_decl(ptr->referenced_type());

    if (auto array = type->isa<IndefiniteArrayType>())
        emit_aggop_decl(array->elem_type());

    if (auto fn = type->isa<FnType>())
        for (auto type : fn->args())
            emit_aggop_decl(type);

    // look for nested array
    if (auto array = type->isa<DefiniteArrayType>()) {
        emit_aggop_decl(array->elem_type());
        emit_type(type_decls_, array) << endl;
        insert(type, "array_" + std::to_string(type->gid()));
    }

    // look for nested tuple
    if (auto tuple = type->isa<TupleType>()) {
        for (auto arg : tuple->args())
            emit_aggop_decl(arg);
        emit_type(type_decls_, tuple) << endl;
        insert(type, "tuple_" + std::to_string(type->gid()));
    }

    // look for nested struct
    if (auto struct_app = type->isa<StructAppType>()) {
        for (auto elem : struct_app->elems())
            emit_aggop_decl(elem);
        emit_type(type_decls_, struct_app) << endl;
        insert(type, "struct_" + std::to_string(type->gid()));
    }

    // restore indent
    while (detail::get_indent() != indent)
        type_decls_ << up;

    return type_decls_;
}

std::ostream& CCodeGen::emit_bitcast(const Def* val, const Def* dst) {
    auto dst_type = dst->type();
    func_impl_ << "union { ";
    emit_addr_space(func_impl_, dst_type);
    emit_type(func_impl_, dst_type) << " dst; ";
    emit_addr_space(func_impl_, val->type());
    emit_type(func_impl_, val->type()) << " src; ";
    func_impl_ << "} u" << dst->unique_name() << ";" << endl;
    func_impl_ << "u" << dst->unique_name() << ".src = ";
    emit(val) << ";" << endl;
    func_impl_ << dst->unique_name() << " = u" << dst->unique_name() << ".dst;";

    return func_impl_;
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

    // emit all globals: do we have globals for CUDA/OpenCL ?
    for (auto primop : world().primops()) {
        if (auto global = primop->isa<Global>()) {
            emit_aggop_decl(global->type());
            emit(global) << endl;
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
        auto ret_type = ret_param_fn_type->num_args() > 2 ? world_.tuple_type(ret_param_fn_type->args().skip_front()) : ret_param_fn_type->args().back();
        auto name = (continuation->is_external() || continuation->empty()) ? continuation->name : continuation->unique_name();
        if (continuation->is_external()) {
            switch (lang_) {
                case Lang::C99:                                  break;
                case Lang::CUDA:   func_decls_ << "__global__ ";
                                   func_impl_  << "__global__ "; break;
                case Lang::OPENCL: func_decls_ << "__kernel ";
                                   func_impl_  << "__kernel ";   break;
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
            if (param->order() == 0 && !param->is_mem()) {
                emit_aggop_decl(param->type());
                if (is_texture_type(param->type())) {
                    // emit texture declaration for CUDA
                    type_decls_ << "texture<";
                    emit_type(type_decls_, param->type()->as<PtrType>()->referenced_type());
                    type_decls_ << ", cudaTextureType1D, cudaReadModeElementType> ";
                    type_decls_ << param->name << ";" << endl;
                    insert(param, param->name);
                    // skip arrays bound to texture memory
                    continue;
                }
                if (i++ > 0) {
                    func_decls_ << ", ";
                    func_impl_  << ", ";
                }

                if (lang_==Lang::OPENCL && continuation->is_external() &&
                    (param->type()->isa<DefiniteArrayType>() ||
                     param->type()->isa<StructAppType>() ||
                     param->type()->isa<TupleType>())) {
                    // structs are passed via buffer; the parameter is a pointer to this buffer
                    func_decls_ << "__global ";
                    func_impl_  << "__global ";
                    emit_type(func_decls_, param->type()) << " *";
                    emit_type(func_impl_,  param->type()) << " *" << param->unique_name() << "_";
                } else {
                    emit_addr_space(func_decls_, param->type());
                    emit_addr_space(func_impl_,  param->type());
                    emit_type(func_decls_, param->type());
                    emit_type(func_impl_,  param->type()) << " " << param->unique_name();
                }
                insert(param, param->unique_name());
            }
        }
        func_decls_ << ");" << endl;
        func_impl_  << ") {" << up;

        // OpenCL: load struct from buffer
        for (auto param : continuation->params()) {
            if (param->order() == 0 && !param->is_mem()) {
                if (lang_==Lang::OPENCL && continuation->is_external() &&
                    (param->type()->isa<DefiniteArrayType>() ||
                     param->type()->isa<StructAppType>() ||
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
                emit_aggop_decl(param->type());
                insert(param, param->unique_name());
            }

            auto continuation = block.continuation();
            if (scope.entry() != continuation) {
                for (auto param : continuation->params()) {
                    if (!param->is_mem()) {
                        func_impl_ << endl;
                        emit_addr_space(func_impl_, param->type());
                        emit_type(func_impl_, param->type()) << "  " << param->unique_name() << ";" << endl;
                        emit_addr_space(func_impl_, param->type());
                        emit_type(func_impl_, param->type()) << " p" << param->unique_name() << ";";
                    }
                }
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
                    if (!param->is_mem())
                        func_impl_ << param->unique_name() << " = p" << param->unique_name() << ";" << endl;
            }

            for (auto primop : block) {
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
                if (!primop->type()->isa<FnType>() && !primop->type()->isa<FrameType>()
                        && (!primop->is_mem() || primop->isa<Store>())) {
                    emit_debug_info(primop);
                    emit(primop) << endl;
                }
            }

            for (auto arg : continuation->args()) {
                // emit definitions of inlined elements
                emit_aggop_defs(arg);
                // emit temporaries for arguments
                if (arg->order() == 0 && !arg->is_mem() && !lookup(arg) && !arg->isa<PrimLit>())
                    emit(arg) << endl;
            }

            // terminate bb
            if (continuation->callee() == ret_param) { // return
                size_t num_args = continuation->num_args();
                switch (num_args) {
                    case 0: break;
                    case 1:
                        if (continuation->arg(0)->is_mem()) {
                            func_impl_ << "return ;";
                            break;
                        } else {
                            func_impl_ << "return ";
                            emit(continuation->arg(0)) << ";";
                        }
                        break;
                    case 2:
                        if (continuation->arg(0)->is_mem()) {
                            func_impl_ << "return ";
                            emit(continuation->arg(1)) << ";";
                            break;
                        } else if (continuation->arg(1)->is_mem()) {
                            func_impl_ << "return ";
                            emit(continuation->arg(0)) << ";";
                            break;
                        }
                        // FALLTHROUGH
                    default: {
                        auto ret_param_fn_type = continuation->arg_fn_type();
                        auto ret_type = world_.tuple_type(ret_param_fn_type->args().skip_front());
                        auto ret_tuple_name = "ret_tuple" + std::to_string(continuation->gid());
                        emit_aggop_decl(ret_type);
                        emit_type(func_impl_, ret_type) << " " << ret_tuple_name << ";";

                        auto tuple = continuation->args().skip_front();
                        for (size_t i = 0, e = tuple.size(); i != e; ++i) {
                            func_impl_ << endl << ret_tuple_name << ".e" << i << " = ";
                            emit(tuple[i]) << ";";
                        }

                        func_impl_ << endl << "return " << ret_tuple_name << ";";
                        break;
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
            } else if (continuation->callee()->isa<Bottom>()) {
                func_impl_ << "return ; // bottom: unreachable";
            } else {
                auto store_phi = [&] (const Def* param, const Def* arg) {
                    if (arg->isa<Bottom>())
                        func_impl_ << "// bottom: ";
                    func_impl_ << "p" << param->unique_name() << " = ";
                    emit(arg) << ";";
                };

                auto callee = continuation->callee()->as_continuation();
                emit_debug_info(callee);

                if (callee->is_basicblock()) {   // ordinary jump
                    assert(callee->num_params()==continuation->num_args());
                    for (size_t i = 0, size = callee->num_params(); i != size; ++i)
                        if (!callee->param(i)->is_mem()) {
                            store_phi(callee->param(i), continuation->arg(i));
                            func_impl_ << endl;
                        }
                    emit(callee);
                } else {
                    if (callee->is_intrinsic()) {
                        if (callee->intrinsic() == Intrinsic::Bitcast) {
                            auto cont = continuation->arg(2)->as_continuation();
                            emit_bitcast(continuation->arg(1), cont->param(1)) << endl;
                            store_phi(cont->param(1), continuation->arg(1));
                        } else if (callee->intrinsic() == Intrinsic::Reserve) {
                            if (!continuation->arg(1)->isa<PrimLit>())
                                ELOG("reserve_shared: couldn't extract memory size at %", continuation->arg(1)->loc());

                            switch (lang_) {
                                case Lang::C99:                                 break;
                                case Lang::CUDA:   func_impl_ << "__shared__ "; break;
                                case Lang::OPENCL: func_impl_ << "__local ";    break;
                            }

                            auto cont = continuation->arg(2)->as_continuation();
                            auto elem_type = cont->param(1)->type()->as<PtrType>()->referenced_type()->as<ArrayType>()->elem_type();
                            emit_type(func_impl_, elem_type) << " " << continuation->arg(1)->unique_name() << "[";
                            emit(continuation->arg(1)) << "];" << endl;
                            insert(continuation->arg(1), continuation->arg(1)->unique_name());
                            store_phi(cont->param(1), continuation->arg(1));
                        } else {
                            THORIN_UNREACHABLE;
                        }
                    } else {
                        auto emit_call = [&] (const Param* param = nullptr) {
                            auto name = (callee->is_external() || callee->empty()) ? callee->name : callee->unique_name();
                            if (param)
                                emit(param) << " = ";
                            func_impl_ << name << "(";
                            // emit all first-order args
                            size_t i = 0;
                            for (auto arg : continuation->args()) {
                                if (arg->order() == 0 && !arg->is_mem()) {
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

                        if (ret_arg == ret_param) {     // call + return
                            if (ret_arg->type() == fn_mem_) {
                                emit_call();
                                func_impl_ << endl << "return ;";
                            } else {
                                func_impl_ << "return ";
                                emit_call();
                            }
                        } else {                        // call + continuation
                            auto succ = ret_arg->as_continuation();
                            const Param* param = nullptr;
                            switch (succ->num_params()) {
                                case 0:
                                    emit_call();
                                    break;
                                case 1:
                                    param = succ->param(0)->is_mem();
                                    if (param->is_mem())
                                        param = nullptr;
                                    emit_call(param);
                                    break;
                                case 2:
                                    assert(succ->mem_param() && "no mem_param found for succ");
                                    param = succ->param(0);
                                    param = param->is_mem() ? succ->param(1) : param;
                                    emit_call(param);
                                    break;
                                default: {
                                    assert(succ->param(0)->is_mem());
                                    auto ret_param_fn_type = ret_arg->type()->as<FnType>();
                                    auto ret_type = world_.tuple_type(ret_param_fn_type->args().skip_front());

                                    auto ret_tuple_name = "ret_tuple" + std::to_string(continuation->gid());
                                    emit_aggop_decl(ret_type);
                                    emit_type(func_impl_, ret_type) << " " << ret_tuple_name << ";" << endl << ret_tuple_name << " = ";
                                    emit_call();

                                    // store arguments to phi node
                                    auto tuple = succ->params().skip_front();
                                    for (size_t i = 0, e = tuple.size(); i != e; ++i)
                                        func_impl_ << endl << "p" << tuple[i]->unique_name() << " = " << ret_tuple_name << ".e" << i << ";";
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if (continuation != scope.entry())
                func_impl_ << down;
        }
        func_impl_ << down << endl << "}" << endl << endl;
        def2str_.clear();
    });

    type2str_.clear();
    def2str_.clear();

    if (lang_==Lang::OPENCL) {
        if (use_16_)
            os_ << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" << endl;
        if (use_64_)
            os_ << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << endl;
        if (use_16_ || use_64_)
            os_ << endl;
    }

    if (lang_==Lang::CUDA) {
        if (use_16_)
            os_ << "#include <cuda_fp16.h>" << endl << endl;
        os_ << "extern \"C\" {" << endl;
    }

    if (!type_decls_.str().empty())
        os_ << type_decls_.str() << endl;
    if (!func_decls_.str().empty())
        os_ << func_decls_.str() << endl;
    os_ << func_impl_.str();

    if (lang_==Lang::CUDA)
        os_ << "}"; // extern "C"
}


std::ostream& CCodeGen::emit(const Def* def) {
    if (auto continuation = def->isa<Continuation>())
        return func_impl_ << "goto l" << continuation->gid() << ";";

    if (lookup(def))
        return func_impl_ << get_name(def);

    if (auto bin = def->isa<BinOp>()) {
        // emit definitions of inlined elements
        emit_aggop_defs(bin->lhs());
        emit_aggop_defs(bin->rhs());
        emit_type(func_impl_, bin->type()) << " " << bin->unique_name() << ";" << endl;
        func_impl_ << bin->unique_name() << " = ";
        emit(bin->lhs());
        if (auto cmp = bin->isa<Cmp>()) {
            switch (cmp->cmp_kind()) {
                case Cmp_eq: func_impl_ << " == "; break;
                case Cmp_ne: func_impl_ << " != "; break;
                case Cmp_gt: func_impl_ << " > ";  break;
                case Cmp_ge: func_impl_ << " >= "; break;
                case Cmp_lt: func_impl_ << " < ";  break;
                case Cmp_le: func_impl_ << " <= "; break;
            }
        }

        if (auto arithop = bin->isa<ArithOp>()) {
            switch (arithop->arithop_kind()) {
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
        insert(def, def->unique_name());
        return func_impl_;
    }

    if (auto conv = def->isa<ConvOp>()) {
        if (conv->from()->type() == conv->type()) {
            insert(def, conv->from()->unique_name());
            return func_impl_;
        }

        emit_addr_space(func_impl_, conv->type());
        emit_type(func_impl_, conv->type()) << " " << conv->unique_name() << ";" << endl;

        if (conv->isa<Cast>()) {
            auto from = conv->from()->type()->as<PrimType>();
            auto to   = conv->type()->as<PrimType>();

            func_impl_ << conv->unique_name() << " = ";

            if (lang_==Lang::CUDA && from && (from->primtype_kind() == PrimType_pf16 || from->primtype_kind() == PrimType_qf16)) {
                func_impl_ << "(";
                emit_type(func_impl_, conv->type()) << ") __half2float(";
                emit(conv->from()) << ");";
            } else if (lang_==Lang::CUDA && to && (to->primtype_kind() == PrimType_pf16 || to->primtype_kind() == PrimType_qf16)) {
                func_impl_ << "__float2half((float)";
                emit(conv->from()) << ");";
            } else {
                func_impl_ << "(";
                emit_addr_space(func_impl_, conv->type());
                emit_type(func_impl_, conv->type()) << ")";
                emit(conv->from()) << ";";
            }
        }

        if (conv->isa<Bitcast>())
            emit_bitcast(conv->from(), conv);

        insert(def, def->unique_name());
        return func_impl_;
    }

    if (auto array = def->isa<DefiniteArray>()) { // DefArray is mapped to a struct
        // emit definitions of inlined elements
        for (auto op : array->ops())
            emit_aggop_defs(op);

        emit_type(func_impl_, array->type()) << " " << array->unique_name() << ";";
        for (size_t i = 0, e = array->size(); i != e; ++i) {
            func_impl_ << endl;
            if (array->op(i)->isa<Bottom>())
                func_impl_ << "// bottom: ";
            func_impl_ << array->unique_name() << ".e[" << i << "] = ";
            emit(array->op(i)) << ";";
        }
        insert(def, def->unique_name());
        return func_impl_;
    }

    // aggregate operations
    {
        auto emit_access = [&] (const Def* def, const Def* index) -> std::ostream& {
            if (def->type()->isa<ArrayType>()) {
                func_impl_ << ".e[";
                emit(index) << "]";
            } else if (def->type()->isa<TupleType>() || def->type()->isa<StructAppType>()) {
                func_impl_ << ".e";
                emit(index);
            } else if (def->type()->isa<VectorType>()) {
                if (index->is_primlit(0))
                    func_impl_ << ".x";
                else if (index->is_primlit(1))
                    func_impl_ << ".y";
                else if (index->is_primlit(2))
                    func_impl_ << ".z";
                else if (index->is_primlit(3))
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
            assert(def->isa<Tuple>() || def->isa<StructAgg>() || def->isa<Vector>());
            // emit definitions of inlined elements
            for (auto op : agg->ops())
                emit_aggop_defs(op);

            emit_type(func_impl_, agg->type()) << " " << agg->unique_name() << ";";
            for (size_t i = 0, e = agg->ops().size(); i != e; ++i) {
                func_impl_ << endl;
                if (agg->op(i)->isa<Bottom>())
                    func_impl_ << "// bottom: ";
                func_impl_ << agg->unique_name();
                emit_access(def, world_.literal_qs32(i, def->loc())) << " = ";
                emit(agg->op(i)) << ";";
            }
            insert(def, def->unique_name());
            return func_impl_;
        }

        if (auto aggop = def->isa<AggOp>()) {
            emit_aggop_defs(aggop->agg());

            if (auto extract = aggop->isa<Extract>()) {
                if (extract->is_mem() || extract->type()->isa<FrameType>())
                    return func_impl_;
                emit_type(func_impl_, aggop->type()) << " " << aggop->unique_name() << ";" << endl;
                func_impl_ << aggop->unique_name() << " = ";
                if (auto memop = extract->agg()->isa<MemOp>())
                    emit(memop) << ";";
                else {
                    emit(aggop->agg());
                    emit_access(aggop->agg(), aggop->index()) << ";";
                }
                insert(def, def->unique_name());
                return func_impl_;
            }

            auto ins = def->as<Insert>();
            emit_type(func_impl_, aggop->type()) << " " << aggop->unique_name() << ";" << endl;
            func_impl_ << aggop->unique_name() << " = ";
            emit(ins->agg()) << ";" << endl;
            func_impl_ << aggop->unique_name();
            emit_access(def, ins->index()) << " = ";
            emit(ins->value()) << ";";
            insert(def, aggop->unique_name());
            return func_impl_;
        }
    }

    if (auto primlit = def->isa<PrimLit>()) {
#if __GNUC__ == 4 || (__GNUC__ == 5 && __GNUC_MINOR__ < 1)
        auto float_mode = std::scientific;
        auto fs = "f";
#else
        auto float_mode = lang_ == Lang::CUDA ? std::scientific : std::hexfloat;
        auto fs = lang_ == Lang::CUDA ? "f" : "";
#endif
        auto hp = lang_ == Lang::CUDA ? "__float2half(" : "";
        auto hs = lang_ == Lang::CUDA ? ")" : "h";

        switch (primlit->primtype_kind()) {
            case PrimType_bool: func_impl_ << (primlit->bool_value() ? "true" : "false");                          break;
            case PrimType_ps8:  case PrimType_qs8:  func_impl_ << (int) primlit->ps8_value();                      break;
            case PrimType_pu8:  case PrimType_qu8:  func_impl_ << (unsigned) primlit->pu8_value();                 break;
            case PrimType_ps16: case PrimType_qs16: func_impl_ << primlit->ps16_value();                           break;
            case PrimType_pu16: case PrimType_qu16: func_impl_ << primlit->pu16_value();                           break;
            case PrimType_ps32: case PrimType_qs32: func_impl_ << primlit->ps32_value();                           break;
            case PrimType_pu32: case PrimType_qu32: func_impl_ << primlit->pu32_value();                           break;
            case PrimType_ps64: case PrimType_qs64: func_impl_ << primlit->ps64_value();                           break;
            case PrimType_pu64: case PrimType_qu64: func_impl_ << primlit->pu64_value();                           break;
            case PrimType_pf16: case PrimType_qf16: func_impl_ << float_mode << hp << primlit->pf16_value() << hs; break;
            case PrimType_pf32: case PrimType_qf32: func_impl_ << float_mode << primlit->pf32_value() << fs;       break;
            case PrimType_pf64: case PrimType_qf64: func_impl_ << float_mode << primlit->pf64_value();             break;
        }
        return func_impl_;
    }

    if (auto bottom = def->isa<Bottom>()) {
        emit_type(func_impl_, bottom->type()) << " " << bottom->unique_name() << "; // bottom";
        insert(def, def->unique_name());
        return func_impl_;
    }

    if (auto load = def->isa<Load>()) {
        emit_type(func_impl_, load->out_val()->type()) << " " << load->unique_name() << ";" << endl;
        func_impl_ << load->unique_name() << " = ";
        // handle texture fetches
        if (!is_texture_type(load->ptr()->type()))
            func_impl_ << "*";
        emit(load->ptr()) << ";";

        insert(def, def->unique_name());
        return func_impl_;
    }

    if (auto store = def->isa<Store>()) {
        emit_aggop_defs(store->val()) << "*";
        emit(store->ptr()) << " = ";
        emit(store->val()) << ";";

        insert(def, def->unique_name());
        return func_impl_;
    }

    if (auto slot = def->isa<Slot>()) {
        emit_type(func_impl_, slot->alloced_type()) << " " << slot->unique_name() << "_slot;" << endl;
        emit_type(func_impl_, slot->alloced_type()) << "* " << slot->unique_name() << ";" << endl;
        func_impl_ << slot->unique_name() << " = &" << slot->unique_name() << "_slot;";
        insert(def, def->unique_name());
        return func_impl_;
    }

    if (def->isa<Enter>())
        return func_impl_;

    if (def->isa<Vector>()) {
        THORIN_UNREACHABLE;
    }

    if (auto lea = def->isa<LEA>()) {
        if (is_texture_type(lea->type())) { // handle texture fetches
            emit_type(func_impl_, lea->ptr_referenced_type()) << " " << lea->unique_name() << ";" << endl;
            func_impl_ << lea->unique_name() << " = tex1Dfetch(";
            emit(lea->ptr()) << ", ";
            emit(lea->index()) << ");";
        } else {
            if (lea->ptr_referenced_type()->isa<TupleType>() || lea->ptr_referenced_type()->isa<StructAppType>()) {
                emit_type(func_impl_, lea->type()) << " " << lea->unique_name() << ";" << endl;
                func_impl_ << lea->unique_name() << " = &";
                emit(lea->ptr()) << "->e";
                emit(lea->index()) << ";";
            } else if (lea->ptr_referenced_type()->isa<DefiniteArrayType>()) {
                emit_type(func_impl_, lea->type()) << " " << lea->unique_name() << ";" << endl;
                func_impl_ << lea->unique_name() << " = &";
                emit(lea->ptr()) << "->e[";
                emit(lea->index()) << "];";
            } else {
                emit_addr_space(func_impl_, lea->ptr()->type());
                emit_type(func_impl_, lea->type()) << " " << lea->unique_name() << ";" << endl;
                func_impl_ << lea->unique_name() << " = ";
                emit(lea->ptr()) << " + ";
                emit(lea->index()) << ";";
            }
        }

        insert(def, def->unique_name());
        return func_impl_;
    }

    if (auto global = def->isa<Global>()) {
        assert(!global->init()->isa_continuation() && "no global init continuation supported");
        switch (lang_) {
            case Lang::C99:                                 break;
            case Lang::CUDA:   func_impl_ << "__device__ "; break;
            case Lang::OPENCL: func_impl_ << "__constant "; break;
        }
        emit_type(func_impl_, global->alloced_type()) << " " << global->unique_name() << "_slot";
        if (global->init()->isa<Bottom>()) {
            func_impl_ << "; // bottom";
        } else {
            func_impl_ << " = ";
            emit(global->init()) << ";";
        }
        func_impl_ << endl;

        switch (lang_) {
            case Lang::C99:                                 break;
            case Lang::CUDA:   func_impl_ << "__device__ "; break;
            case Lang::OPENCL: func_impl_ << "__constant "; break;
        }
        emit_type(func_impl_, global->alloced_type()) << " *" << global->unique_name() << " = &" << global->unique_name() << "_slot;";

        insert(def, def->unique_name());
        return func_impl_;
    }

    THORIN_UNREACHABLE;
}

bool CCodeGen::lookup(const Type* type) {
    return type2str_.contains(type);
}
bool CCodeGen::lookup(const Def* def) {
    return def2str_.contains(def);
}

std::string& CCodeGen::get_name(const Type* type) {
    return type2str_[type];
}

std::string& CCodeGen::get_name(const Def* def) {
    return def2str_[def];
}

void CCodeGen::insert(const Type* type, std::string str) {
    type2str_[type] = str;
}

void CCodeGen::insert(const Def* def, std::string str) {
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

void emit_c(World& world, std::ostream& stream, Lang lang, bool debug) { CCodeGen(world, stream, lang, debug).emit(); }

//------------------------------------------------------------------------------

}
