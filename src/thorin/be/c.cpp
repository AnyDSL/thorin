#include "thorin/lambda.h"
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

namespace thorin {

class CCodeGen {
public:
    CCodeGen(World& world, std::ostream& stream, Lang lang, bool debug)
        : world_(world)
        , lang_(lang)
        , debug_(debug)
        , os(stream)
    {}

    void emit();
    World& world() const { return world_; }

private:
    std::ostream& emit_aggop_defs(Def def);
    std::ostream& emit_aggop_decl(Type);
    std::ostream& emit_debug_info(Def def);
    std::ostream& emit_type(Type);
    std::ostream& emit(Def def);
    bool lookup(size_t gid);
    std::ostream& insert(size_t gid, std::string str);
    std::string &get_name(size_t gid);
    bool is_texture_type(Type type);

    World& world_;
    Lang lang_;
    HashMap<size_t, std::string> globals_;
    HashMap<size_t, std::string> primops_;
    bool process_kernel_ = false;
    bool debug_;
    std::ostream& os;
};


std::ostream& CCodeGen::emit_debug_info(Def def) {
    if (debug_)
        return streamf(os, "#line % \"%\"", def->loc().pos1().line(), def->loc().pos1().filename()) << endl;
    return os;
}


std::ostream& CCodeGen::emit_type(Type type) {
    if (type.empty()) {
        return os << "NULL";
    } else if (type.isa<FrameType>()) {
        return os;
    } else if (type.isa<MemType>()) {
        return os << "void";
    } else if (type.isa<FnType>()) {
        THORIN_UNREACHABLE;
    } else if (auto tuple = type.isa<TupleType>()) {
        if (lookup(tuple->gid()))
            return os << get_name(tuple->gid());
        os << "typedef struct tuple_" << tuple->gid() << " {" << up;
        for (size_t i = 0, e = tuple->args().size(); i != e; ++i) {
            os << endl;
            emit_type(tuple->arg(i)) << " e" << i << ";";
        }
        os << down << endl << "} tuple_" << tuple->gid() << ";";
        return os;
    } else if (auto struct_abs = type.isa<StructAbsType>()) {
        return os << struct_abs->name();
    } else if (auto struct_app = type.isa<StructAppType>()) {
        if (lookup(struct_app->gid()))
            return os << get_name(struct_app->gid());
        os << "typedef struct struct_" << struct_app->gid() << " {" << up;
        for (size_t i = 0, e = struct_app->elems().size(); i != e; ++i) {
            os << endl;
            emit_type(struct_app->elem(i)) << " e" << i << ";";
        }
        os << down << endl << "} struct_" << struct_app->gid() << ";";
        return os;
    } else if (type.isa<TypeVar>()) {
        THORIN_UNREACHABLE;
    } else if (auto array = type.isa<IndefiniteArrayType>()) {
        emit_type(array->elem_type());
        return os;
    } else if (auto array = type.isa<DefiniteArrayType>()) { // DefArray is mapped to a struct
        if (lookup(array->gid()))
            return os << get_name(array->gid());
        os << "typedef struct array_" << array->gid() << " {" << up << endl;
        emit_type(array->elem_type()) << " e[" << array->dim() << "];";
        os << down << endl << "} array_" << array->gid() << ";";
        return os;
    } else if (auto ptr = type.isa<PtrType>()) {
        if (lang_==Lang::CUDA) {
            switch (ptr->addr_space()) {
                default: break;
                // only declaration need __shared__
                case AddressSpace::Shared: os << "";  break;
            }
        }
        if (lang_==Lang::OPENCL) {
            switch (ptr->addr_space()) {
                default: break;
                case AddressSpace::Generic: // once address spaces are correct, ::Global should be sufficient
                case AddressSpace::Global: os << "__global "; break;
                case AddressSpace::Shared: os << "__local ";  break;
            }
        }
        emit_type(ptr->referenced_type());
        os << '*';
        if (ptr->is_vector())
            os << ptr->referenced_type()->length();
        return os;
    } else if (auto primtype = type.isa<PrimType>()) {
        switch (primtype->primtype_kind()) {
            case PrimType_bool:                     os << "bool";             break;
            case PrimType_ps8:  case PrimType_qs8:  os << "char";             break;
            case PrimType_pu8:  case PrimType_qu8:  os << "unsigned char";    break;
            case PrimType_ps16: case PrimType_qs16: os << "short";            break;
            case PrimType_pu16: case PrimType_qu16: os << "unsigned short";   break;
            case PrimType_ps32: case PrimType_qs32: os << "int";              break;
            case PrimType_pu32: case PrimType_qu32: os << "unsigned int";     break;
            case PrimType_ps64: case PrimType_qs64: os << "long";             break;
            case PrimType_pu64: case PrimType_qu64: os << "unsigned long";    break;
            case PrimType_pf32: case PrimType_qf32: os << "float";            break;
            case PrimType_pf64: case PrimType_qf64: os << "double";           break;
        }
        if (primtype->is_vector())
            os << primtype->length();
        return os;
    }
    THORIN_UNREACHABLE;
}


std::ostream& CCodeGen::emit_aggop_defs(Def def) {
    if (lookup(def->gid()))
        return os;

    // recurse into (multi-dimensional) array
    if (auto array = def->isa<DefiniteArray>()) {
        for (auto op : array->ops())
            emit_aggop_defs(op);
        if (lookup(def->gid()))
            return os;
        emit(array) << endl;
    }

    // recurse into (multi-dimensional) tuple or struct
    if (auto agg = def->isa<Aggregate>()) {
        for (auto op : agg->ops())
            emit_aggop_defs(op);
        if (lookup(def->gid()))
            return os;
        emit(agg) << endl;
    }

    // argument is a cast
    if (auto conv = def->isa<ConvOp>()) {
        emit(conv) << endl;
    }

    return os;
}


std::ostream& CCodeGen::emit_aggop_decl(Type type) {
    type.unify(); // make sure that we get the same id if types are equal

    if (lookup(type->gid()))
        return os;

    if (auto ptr = type.isa<PtrType>()) {
        emit_aggop_decl(ptr->referenced_type());
        return os;
    }
    if (auto array = type.isa<IndefiniteArrayType>()) {
        emit_aggop_decl(array->elem_type());
        return os;
    }
    if (auto fn = type.isa<FnType>()) {
        for (auto type : fn->args())
            emit_aggop_decl(type);
        return os;
    }

    // recurse into (multi-dimensional) array
    if (auto array = type.isa<DefiniteArrayType>()) {
        emit_aggop_decl(array->elem_type());
        emit_type(array) << endl;
        insert(type->gid(), "array_" + std::to_string(type->gid()));
        return os;
    }

    // recurse into (multi-dimensional) tuple
    if (auto tuple = type.isa<TupleType>()) {
        for (auto arg : tuple->args())
            emit_aggop_decl(arg);
        emit_type(tuple) << endl;
        insert(type->gid(), "tuple_" + std::to_string(type->gid()));
        return os;
    }

    // recurse into (multi-dimensional) struct
    if (auto struct_app = type.isa<StructAppType>()) {
        for (auto elem : struct_app->elems())
            emit_aggop_decl(elem);
        emit_type(struct_app) << endl;
        insert(type->gid(), "struct_" + std::to_string(type->gid()));
        return os;
    }

    return os;
}


void CCodeGen::emit() {
    if (lang_==Lang::CUDA) {
        os << "extern \"C\" {\n";
        os << "__device__ inline int threadIdx_x() { return threadIdx.x; }" << endl;
        os << "__device__ inline int threadIdx_y() { return threadIdx.y; }" << endl;
        os << "__device__ inline int threadIdx_z() { return threadIdx.z; }" << endl;
        os << "__device__ inline int blockIdx_x() { return blockIdx.x; }" << endl;
        os << "__device__ inline int blockIdx_y() { return blockIdx.y; }" << endl;
        os << "__device__ inline int blockIdx_z() { return blockIdx.z; }" << endl;
        os << "__device__ inline int blockDim_x() { return blockDim.x; }" << endl;
        os << "__device__ inline int blockDim_y() { return blockDim.y; }" << endl;
        os << "__device__ inline int blockDim_z() { return blockDim.z; }" << endl;
        os << "__device__ inline int gridDim_x() { return gridDim.x; }" << endl;
        os << "__device__ inline int gridDim_y() { return gridDim.y; }" << endl;
        os << "__device__ inline int gridDim_z() { return gridDim.z; }" << endl;
        os << "__device__ inline int as_int(float val) { return __float_as_int(val); }" << endl;
        os << "__device__ inline int as_float(int val) { return __int_as_float(val); }" << endl;
    }
    if (lang_==Lang::OPENCL) {
        os << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << endl;
    }

    // emit declarations
    Scope::for_each<false>(world(), [&] (const Scope& scope) {
        if (scope.entry() == world().branch()) return;
        auto schedule = schedule_smart(scope);

        // tuple declarations
        for (auto& block : schedule) {
            for (auto param : block.lambda()->params()) {
                emit_aggop_decl(param->type());
                insert(param->gid(), param->unique_name());
            }

            for (auto primop : block) {
                if (!primop->isa<MemOp>()) {
                    emit_aggop_decl(primop->type());
                    // search for inlined tuples/arrays
                    if (auto aggop = primop->isa<AggOp>()) {
                        if (!aggop->agg()->isa<MemOp>())
                            emit_aggop_decl(aggop->agg()->type());
                    }
                }
            }
        }

        // lambda declarations
        auto lambda = scope.entry();
        if (lambda->is_intrinsic())
            return;

        // retrieve return param
        const Param *ret_param = nullptr;
        for (auto param : lambda->params()) {
            emit_aggop_decl(param->type());
            if (param->order() != 0) {
                assert(!ret_param);
                ret_param = param;
            }
        }
        assert(ret_param);

        // emit texture declaration for CUDA
        for (auto param : lambda->params()) {
            if (param->order() == 0 && !param->is_mem()) {
                if (is_texture_type(param->type())) {
                    os << "texture<";
                    emit_type(param->type().as<PtrType>()->referenced_type());
                    os << ", cudaTextureType1D, cudaReadModeElementType> ";
                    os << param->name << ";" << endl;
                    insert(param->gid(), param->name);
                }
            }
        }

        // skip device functions and kernel entries (the kernel signature below is different)
        if (lambda->cc() == CC::Device || lambda->is_external())
            return;

        // emit function declaration
        auto ret_fn_type = ret_param->type().as<FnType>();
        auto name = (lambda->is_external() || lambda->empty()) ? lambda->name : lambda->unique_name();
        if (lang_==Lang::CUDA)
            os << "__device__ ";
        emit_type(ret_fn_type->args().back()) << " " << name << "(";
        size_t i = 0;
        for (auto param : lambda->params()) {
            if (param->order() == 0 && !param->is_mem()) {
                // skip arrays bound to texture memory
                if (is_texture_type(param->type())) continue;
                if (i++ > 0) os << ", ";
                emit_type(param->type());
                insert(param->gid(), param->unique_name());
            }
        }
        os << ");" << endl;
    });
    os << endl;

    // emit all globals
    for (auto primop : world().primops()) {
        if (auto global = primop->isa<Global>()) {
            emit_aggop_decl(global->type());
            emit(global) << endl;
        }
    }

    // emit connected functions first
    process_kernel_ = true;

    Scope::for_each(world(), [&] (const Scope& scope) {
        auto lambda = scope.entry();
        if (lambda->is_intrinsic())
            return;

        assert(lambda->is_returning());

        // retrieve return param
        const Param* ret_param = nullptr;
        for (auto param : lambda->params()) {
            if (param->order() != 0) {
                assert(!ret_param);
                ret_param = param;
            }
        }
        assert(ret_param);

        auto ret_fn_type = ret_param->type().as<FnType>();
        auto name = (lambda->is_external() || lambda->empty()) ? lambda->name : lambda->unique_name();
        if (lambda->is_external()) {
            switch (lang_) {
                case Lang::C99:                         break;
                case Lang::CUDA:   os << "__global__ "; break;
                case Lang::OPENCL: os << "__kernel ";   break;
            }
        } else {
            if (lang_==Lang::CUDA) os << "__device__ ";
        }
        emit_type(ret_fn_type->args().back()) << " " << name << "(";
        size_t i = 0;
        // emit and store all first-order params
        for (auto param : lambda->params()) {
            if (param->order() == 0 && !param->is_mem()) {
                // skip arrays bound to texture memory
                if (is_texture_type(param->type())) continue;
                if (i++ > 0) os << ", ";
                if (lang_==Lang::OPENCL && lambda->is_external() &&
                    (param->type().isa<DefiniteArrayType>() ||
                     param->type().isa<StructAppType>() ||
                     param->type().isa<TupleType>())) {
                    // structs are passed via buffer; the parameter is a pointer to this buffer
                    os << "__global ";
                    emit_type(param->type()) << " *" << param->unique_name() << "_";
                } else {
                    emit_type(param->type()) << " " << param->unique_name();
                }
            }
        }
        os << ") {" << up;

        // emit and store all first-order params
        for (auto param : lambda->params()) {
            if (param->order() == 0 && !param->is_mem()) {
                if (lang_==Lang::OPENCL && lambda->is_external() &&
                    (param->type().isa<DefiniteArrayType>() ||
                     param->type().isa<StructAppType>() ||
                     param->type().isa<TupleType>())) {
                    // load struct from buffer
                    os << endl;
                    emit_type(param->type()) << " " << param->unique_name() << " = *" << param->unique_name() << "_;";
                }
            }
        }

        auto schedule = schedule_smart(scope);

        // emit function arguments and phi nodes
        for (const auto& block : schedule) {
            auto lambda = block.lambda();
            if (scope.entry() != lambda) {
                for (auto param : lambda->params()) {
                    if (!param->is_mem()) {
                        os << endl;
                        emit_type(param->type()) << "  " << param->unique_name() << ";" << endl;
                        emit_type(param->type()) << " p" << param->unique_name() << ";";
                    }
                }
            }
        }

        for (const auto& block : schedule) {
            auto lambda = block.lambda();
            if (lambda->empty())
                continue;
            assert(lambda == scope.entry() || lambda->is_basicblock());
            os << endl;

            // print label for the current basic block
            if (lambda != scope.entry()) {
                os << "l" << lambda->gid() << ": ;" << up << endl;
                // load params from phi node
                if (lambda->to() != ret_param) // skip for return
                    for (auto param : lambda->params())
                        if (!param->is_mem()) {
                            os << param->unique_name() << " = p" << param->unique_name() << ";" << endl;
                        }
            }

            for (auto primop : block) {
                // skip higher-order primops, stuff dealing with frames and all memory related stuff except stores
                if (!primop->type().isa<FnType>() && !primop->type().isa<FrameType>()
                        && (!primop->is_mem() || primop->isa<Store>())) {
                    emit_debug_info(primop);
                    emit(primop) << endl;
                }
            }

            // terminate bb
            if (lambda->to() == ret_param) { // return
                size_t num_args = lambda->num_args();
                os << "return ";
                switch (num_args) {
                    case 0: break;
                    case 1:
                        if (lambda->arg(0)->is_mem())
                            break;
                        else
                            emit(lambda->arg(0));
                        break;
                    case 2:
                        if (lambda->arg(0)->is_mem()) {
                            emit(lambda->arg(1));
                            break;
                        } else if (lambda->arg(1)->is_mem()) {
                            emit(lambda->arg(0));
                            break;
                        }
                        // FALLTHROUGH
                    default:
                        THORIN_UNREACHABLE;
                }
                os << ";";
            } else if (lambda->to() == world().branch()) {
                emit_debug_info(lambda->arg(0)); // TODO correct?
                os << "if (";
                emit(lambda->arg(0));
                os << ") ";
                emit(lambda->arg(1));
                os << " else ";
                emit(lambda->arg(2));
            } else if (lambda->to()->isa<Bottom>()) {
                os << "return ; // bottom: unreachable";
            } else {
                Lambda* to_lambda = lambda->to()->as_lambda();
                emit_debug_info(to_lambda);

                // emit inlined arrays/tuples/structs before the call operation
                for (auto arg : lambda->args()) emit_aggop_defs(arg);

                if (to_lambda->is_basicblock()) {   // ordinary jump
                    assert(to_lambda->num_params()==lambda->num_args());
                    // store argument to phi nodes
                    for (size_t i = 0, size = to_lambda->num_params(); i != size; ++i)
                        if (!to_lambda->param(i)->is_mem()) {
                            os << "p" << to_lambda->param(i)->unique_name() << " = ";
                            emit(lambda->arg(i)) << ";" << endl;
                        }
                    emit(to_lambda);
                } else {
                    if (to_lambda->is_intrinsic()) {
                        if (to_lambda->intrinsic() == Intrinsic::Reinterpret) {
                            auto cont = lambda->arg(2)->as_lambda();
                            emit(cont->param(1)) << " = as_";
                            emit_type(cont->param(1)->type()) << "(";
                            emit(lambda->arg(1)) << ");" << endl;
                        } else if (to_lambda->intrinsic() == Intrinsic::Reserve) {
                            if (!lambda->arg(1)->isa<PrimLit>())
                                WLOG("error: reserve_shared: couldn't extract memory size at %", lambda->arg(1)->loc());

                            switch (lang_) {
                                case Lang::C99:                         break;
                                case Lang::CUDA:   os << "__shared__ "; break;
                                case Lang::OPENCL: os << "__local ";    break;
                            }

                            auto cont = lambda->arg(2)->as_lambda();
                            auto fn_type = to_lambda->type().as<FnType>();
                            emit_type(fn_type->args().back().as<FnType>()->arg(1).as<PtrType>()->referenced_type()) << " " << to_lambda->name << lambda->gid() << "[";
                            emit(lambda->arg(1)) << "];" << endl;
                            // store argument to phi nodes
                            os << "p" << cont->param(1)->unique_name() << " = " << to_lambda->name << lambda->gid() << ";";
                        } else {
                            THORIN_UNREACHABLE;
                        }
                    } else {
                        auto emit_call = [&] () {
                            auto name = (to_lambda->is_external() || to_lambda->empty()) ? to_lambda->name : to_lambda->unique_name();
                            os << name << "(";
                            // emit all first-order args
                            size_t i = 0;
                            for (auto arg : lambda->args()) {
                                if (arg->order() == 0 && !arg->is_mem()) {
                                    if (i++ > 0) os << ", ";
                                    emit(arg);
                                }
                            }
                            os << ");";
                        };

                        Def ret_arg = 0;
                        for (auto arg : lambda->args()) {
                            // retrieve return argument
                            if (arg->order() != 0) {
                                assert(!ret_arg);
                                ret_arg = arg;
                            }
                            // emit temporaries for args
                            if (arg->order() == 0 && !arg->is_mem() &&
                                !lookup(arg->gid()) && !arg->isa<PrimLit>()) {
                                emit(arg) << endl;
                            }
                        }

                        if (ret_arg == ret_param) {     // call + return
                            os << "return ";
                            emit_call();
                        } else {                        // call + continuation
                            auto succ = ret_arg->as_lambda();
                            auto param = succ->param(0)->is_mem() ? nullptr : succ->param(0);
                            if (param == nullptr && succ->num_params() == 2)
                                param = succ->param(1);

                            if (param)
                                emit(param) << " = ";

                            emit_call();

                            if (param) {
                                // store argument to phi nodes
                                os << endl << "p" << succ->param(1)->unique_name() << " = ";
                                emit(param) << ";";
                            }
                        }
                    }
                }
            }
            if (lambda != scope.entry())
                os << down;
        }
        os << down << endl << "}" << endl << endl;

        primops_.clear();
    });

    globals_.clear();
    primops_.clear();
    if (lang_==Lang::CUDA)
        os << "}" << endl; // extern "C"
}


std::ostream& CCodeGen::emit(Def def) {
    if (auto lambda = def->isa<Lambda>())
        return os << "goto l" << lambda->gid() << ";";

    if (lookup(def->gid()))
        return os << get_name(def->gid());

    if (auto bin = def->isa<BinOp>()) {
        // emit definitions of inlined elements
        emit_aggop_defs(bin->lhs());
        emit_aggop_defs(bin->rhs());
        emit_type(bin->type()) << " " << bin->unique_name() << ";" << endl;
        os << bin->unique_name() << " = ";
        emit(bin->lhs());
        if (auto cmp = bin->isa<Cmp>()) {
            switch (cmp->cmp_kind()) {
                case Cmp_eq: os << " == "; break;
                case Cmp_ne: os << " != "; break;
                case Cmp_gt: os << " > ";  break;
                case Cmp_ge: os << " >= "; break;
                case Cmp_lt: os << " < ";  break;
                case Cmp_le: os << " <= "; break;
            }
        }

        if (auto arithop = bin->isa<ArithOp>()) {
            switch (arithop->arithop_kind()) {
                case ArithOp_add: os << " + ";  break;
                case ArithOp_sub: os << " - ";  break;
                case ArithOp_mul: os << " * ";  break;
                case ArithOp_div: os << " / ";  break;
                case ArithOp_rem: os << " % ";  break;
                case ArithOp_and: os << " & ";  break;
                case ArithOp_or:  os << " | ";  break;
                case ArithOp_xor: os << " ^ ";  break;
                case ArithOp_shl: os << " << "; break;
                case ArithOp_shr: os << " >> "; break;
            }
        }
        emit(bin->rhs()) << ";";
        return insert(def->gid(), def->unique_name());
    }

    if (auto conv = def->isa<ConvOp>()) {
        emit_type(conv->type()) << " " << conv->unique_name() << ";" << endl;
        os << conv->unique_name() << " = (";
        emit_type(conv->type()) << ")";
        emit(conv->from()) << ";";
        return insert(def->gid(), def->unique_name());
    }

    if (auto array = def->isa<DefiniteArray>()) { // DefArray is mapped to a struct
        // emit definitions of inlined elements
        for (auto op : array->ops())
            emit_aggop_defs(op);

        emit_type(array->type()) << " " << array->unique_name() << ";";
        for (size_t i = 0, e = array->size(); i != e; ++i) {
            os << endl << array->unique_name() << ".e[" << i << "] = ";
            emit(array->op(i)) << ";";
        }
        return insert(def->gid(), def->unique_name());
    }

    if (auto agg = def->isa<Aggregate>()) {
        assert(def->isa<Tuple>() || def->isa<StructAgg>() || def->isa<Vector>());
        // emit definitions of inlined elements
        for (auto op : agg->ops())
            emit_aggop_defs(op);

        emit_type(agg->type()) << " " << agg->unique_name() << ";";
        char elem_prefix = (def->isa<Vector>()) ? 's' : 'e';
        for (size_t i = 0, e = agg->ops().size(); i != e; ++i) {
            os << endl << agg->unique_name() << "." << elem_prefix << i << " = ";
            emit(agg->op(i)) << ";";
        }
        return insert(def->gid(), def->unique_name());
    }

    if (auto aggop = def->isa<AggOp>()) {
        auto emit_access = [&] (Def def, Def index) -> std::ostream& {
            if (def->type().isa<ArrayType>()) {
                emit(def) << ".e[";
                emit(index) << "]";
            } else if (def->type().isa<TupleType>() || def->type().isa<StructAppType>()) {
                emit(def) << ".e";
                emit(index);
            } else if (def->type().isa<VectorType>()) {
                if (index->is_primlit(0))
                    emit(def) << ".x";
                else if (index->is_primlit(1))
                    emit(def) << ".y";
                else if (index->is_primlit(2))
                    emit(def) << ".z";
                else if (index->is_primlit(3))
                    emit(def) << ".w";
                else
                    THORIN_UNREACHABLE;
            } else {
                THORIN_UNREACHABLE;
            }
            return os;
        };

        emit_aggop_defs(aggop->agg());

        if (auto extract = aggop->isa<Extract>()) {
            if (extract->is_mem() || extract->type().isa<FrameType>())
                return os;
            emit_type(aggop->type()) << " " << aggop->unique_name() << ";" << endl;
            os << aggop->unique_name() << " = ";
            if (auto memop = extract->agg()->isa<MemOp>())
                emit(memop) << ";";
            else
                emit_access(aggop->agg(), aggop->index()) << ";";
            return insert(def->gid(), def->unique_name());
        }

        auto ins = def->as<Insert>();
        emit_type(aggop->type()) << " " << aggop->unique_name() << " = ";
        emit(ins->agg()) << ";" << endl;
        insert(def->gid(), aggop->unique_name());
        emit_access(def, ins->index()) << " = ";
        emit(ins->value()) << ";";
        return os;
    }

    if (auto primlit = def->isa<PrimLit>()) {
        switch (primlit->primtype_kind()) {
            case PrimType_bool: os << (primlit->bool_value() ? "true" : "false");                       break;
            case PrimType_ps8:  case PrimType_qs8:  os << (int) primlit->ps8_value();                   break;
            case PrimType_pu8:  case PrimType_qu8:  os << (unsigned) primlit->pu8_value();              break;
            case PrimType_ps16: case PrimType_qs16: os << primlit->ps16_value();                        break;
            case PrimType_pu16: case PrimType_qu16: os << primlit->pu16_value();                        break;
            case PrimType_ps32: case PrimType_qs32: os << primlit->ps32_value();                        break;
            case PrimType_pu32: case PrimType_qu32: os << primlit->pu32_value();                        break;
            case PrimType_ps64: case PrimType_qs64: os << primlit->ps64_value();                        break;
            case PrimType_pu64: case PrimType_qu64: os << primlit->pu64_value();                        break;
            case PrimType_pf32: case PrimType_qf32: os << std::fixed << primlit->pf32_value() << 'f';   break;
            case PrimType_pf64: case PrimType_qf64: os << std::fixed << primlit->pf64_value();          break;
        }
        return os;
    }

    if (def->isa<Bottom>())
        return os << "42 /* bottom */";

    if (auto load = def->isa<Load>()) {
        emit_type(load->out_val()->type()) << " " << load->unique_name() << ";" << endl;
        os << load->unique_name() << " = ";
        // handle texture fetches
        if (!is_texture_type(load->ptr()->type()))
            os << "*";
        emit(load->ptr()) << ";";

        return insert(def->gid(), def->unique_name());
    }

    if (auto store = def->isa<Store>()) {
        emit_aggop_defs(store->val()) << "*";
        emit(store->ptr()) << " = ";
        emit(store->val()) << ";";

        return insert(def->gid(), def->unique_name());
    }

    if (auto slot = def->isa<Slot>()) {
        emit_type(slot->alloced_type()) << " " << slot->unique_name() << "_slot;" << endl;
        emit_type(slot->alloced_type()) << "* " << slot->unique_name() << ";" << endl;
        os << slot->unique_name() << " = &" << slot->unique_name() << "_slot;";
        return insert(def->gid(), def->unique_name());
    }

    if (def->isa<Enter>())
        return os;

    if (def->isa<Vector>()) {
        THORIN_UNREACHABLE;
    }

    if (auto lea = def->isa<LEA>()) {
        if (is_texture_type(lea->type())) { // handle texture fetches
            emit_type(lea->ptr_referenced_type()) << " " << lea->unique_name() << ";" << endl;
            os << lea->unique_name() << " = tex1Dfetch(";
            emit(lea->ptr()) << ", ";
            emit(lea->index()) << ");";
        } else {
            emit_type(lea->type()) << " " << lea->unique_name() << ";" << endl;
            os << lea->unique_name() << " = ";
            if (lea->ptr_referenced_type().isa<TupleType>() || lea->ptr_referenced_type().isa<StructAppType>()) {
                os << "&";
                emit(lea->ptr()) << "->e";
                emit(lea->index()) << ";";
            } else if (lea->ptr_referenced_type().isa<DefiniteArrayType>()) {
                os << "&";
                emit(lea->ptr()) << "->e[";
                emit(lea->index()) << "];";
            } else {
                emit(lea->ptr()) << " + ";
                emit(lea->index()) << ";";
            }
        }

        return insert(def->gid(), def->unique_name());
    }

    if (auto global = def->isa<Global>()) {
        assert(!global->init()->isa_lambda() && "no global init lambda supported");
        switch (lang_) {
            case Lang::C99:                         break;
            case Lang::CUDA:   os << "__device__ "; break;
            case Lang::OPENCL: os << "__constant "; break;
        }
        emit_type(global->alloced_type()) << " " << global->unique_name() << "_slot";
        if (global->init()->isa<Bottom>()) {
            os << ";";
        } else {
            os << " = ";
            emit(global->init()) << ";";
        }
        os << endl;

        switch (lang_) {
            case Lang::C99:                         break;
            case Lang::CUDA:   os << "__device__ "; break;
            case Lang::OPENCL: os << "__constant "; break;
        }
        emit_type(global->alloced_type()) << " *" << global->unique_name() << " = &" << global->unique_name() << "_slot;";

        return insert(def->gid(), def->unique_name());
    }

    THORIN_UNREACHABLE;
}

bool CCodeGen::lookup(size_t gid) {
    if (globals_.count(gid)) return true;
    if (primops_.count(gid)) return true;
    return false;
}

std::string &CCodeGen::get_name(size_t gid) {
    if (globals_.count(gid)) return globals_[gid];
    if (primops_.count(gid)) return primops_[gid];
    THORIN_UNREACHABLE; // couldn't find def
}

std::ostream& CCodeGen::insert(size_t gid, std::string str) {
    if (process_kernel_)
        primops_[gid] = str;
    else
        globals_[gid] = str;
    return os;
}

bool CCodeGen::is_texture_type(Type type) {
    if (auto ptr = type.isa<PtrType>()) {
        if (ptr->addr_space()==AddressSpace::Texture) {
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
