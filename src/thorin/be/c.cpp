#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/cfg.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/printer.h"
#include "thorin/be/c.h"

namespace thorin {

class CCodeGen : public Printer {
public:
    CCodeGen(World& world, std::ostream& stream, Lang lang, bool debug)
        : Printer(stream)
        , world_(world)
        , lang_(lang)
        , debug_(debug)
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
};


std::ostream& CCodeGen::emit_debug_info(Def def) {
    if (debug_) {
        stream() << "#line " << def->loc().pos1().line() << " \"" <<  def->loc().pos1().filename() << "\"";
        newline();
    }
    return stream();
}


std::ostream& CCodeGen::emit_type(Type type) {
    if (type.empty()) {
        return stream() << "NULL";
    } else if (type.isa<FrameType>()) {
        return stream();
    } else if (type.isa<MemType>()) {
        return stream() << "void";
    } else if (type.isa<FnType>()) {
        THORIN_UNREACHABLE;
    } else if (auto tuple = type.isa<TupleType>()) {
        if (lookup(tuple->gid())) return stream() << get_name(tuple->gid());
        stream() << "typedef struct tuple_" << tuple->gid() << " {";
        ++indent;
        for (size_t i = 0, e = tuple->args().size(); i != e; ++i) {
            newline();
            emit_type(tuple->arg(i)) << " e" << i << ";";
        }
        --indent; newline();
        stream() << "} tuple_" << tuple->gid() << ";";
        return stream();
    } else if (auto struct_abs = type.isa<StructAbsType>()) {
        return stream() << struct_abs->name();
    } else if (auto struct_app = type.isa<StructAppType>()) {
        if (lookup(struct_app->gid())) return stream() << get_name(struct_app->gid());
        stream() << "typedef struct struct_" << struct_app->gid() << " {";
        ++indent;
        for (size_t i = 0, e = struct_app->elems().size(); i != e; ++i) {
            newline();
            emit_type(struct_app->elem(i)) << " e" << i << ";";
        }
        --indent; newline();
        stream() << "} struct_" << struct_app->gid() << ";";
        return stream();
    } else if (type.isa<TypeVar>()) {
        THORIN_UNREACHABLE;
    } else if (auto array = type.isa<IndefiniteArrayType>()) {
        emit_type(array->elem_type());
        return stream();
    } else if (auto array = type.isa<DefiniteArrayType>()) { // DefArray is mapped to a struct
        if (lookup(array->gid())) return stream() << get_name(array->gid());
        stream() << "typedef struct array_" << array->gid() << " {";
        ++indent; newline();
        emit_type(array->elem_type()) << " e[" << array->dim() << "];";
        --indent; newline();
        stream() << "} array_" << array->gid() << ";";
        return stream();
    } else if (auto ptr = type.isa<PtrType>()) {
        if (lang_==Lang::CUDA) {
            switch (ptr->addr_space()) {
                default: break;
                // only declaration need __shared__
                case AddressSpace::Shared: stream() << "";  break;
            }
        }
        if (lang_==Lang::OPENCL) {
            switch (ptr->addr_space()) {
                default: break;
                //case AddressSpace::Generic: // once address spaces are correct, ::Global should be sufficient -> use mmap
                case AddressSpace::Global: stream() << "__global "; break;
                case AddressSpace::Shared: stream() << "__local ";  break;
            }
        }
        emit_type(ptr->referenced_type());
        stream() << '*';
        if (ptr->is_vector())
            stream() << ptr->referenced_type()->length();
        return stream();
    } else if (auto primtype = type.isa<PrimType>()) {
        switch (primtype->primtype_kind()) {
            case PrimType_bool:                     stream() << "bool";             break;
            case PrimType_ps8:  case PrimType_qs8:  stream() << "char";             break;
            case PrimType_pu8:  case PrimType_qu8:  stream() << "unsigned char";    break;
            case PrimType_ps16: case PrimType_qs16: stream() << "short";            break;
            case PrimType_pu16: case PrimType_qu16: stream() << "unsigned short";   break;
            case PrimType_ps32: case PrimType_qs32: stream() << "int";              break;
            case PrimType_pu32: case PrimType_qu32: stream() << "unsigned int";     break;
            case PrimType_ps64: case PrimType_qs64: stream() << "long";             break;
            case PrimType_pu64: case PrimType_qu64: stream() << "unsigned long";    break;
            case PrimType_pf32: case PrimType_qf32: stream() << "float";            break;
            case PrimType_pf64: case PrimType_qf64: stream() << "double";           break;
        }
        if (primtype->is_vector())
            stream() << primtype->length();
        return stream();
    }
    THORIN_UNREACHABLE;
}


std::ostream& CCodeGen::emit_aggop_defs(Def def) {
    if (lookup(def->gid())) return stream();

    // recurse into (multi-dimensional) array
    if (auto array = def->isa<DefiniteArray>()) {
        for (auto op : array->ops()) emit_aggop_defs(op);
        if (lookup(def->gid())) return stream();
        emit(array);
        newline();
    }

    // recurse into (multi-dimensional) tuple or struct
    if (auto agg = def->isa<Aggregate>()) {
        for (auto op : agg->ops()) emit_aggop_defs(op);
        if (lookup(def->gid())) return stream();
        emit(agg);
        newline();
    }

    // argument is a cast
    if (auto conv = def->isa<ConvOp>()) {
        emit(conv);
        newline();
    }

    return stream();
}


std::ostream& CCodeGen::emit_aggop_decl(Type type) {
    type.unify(); // make sure that we get the same id if types are equal

    if (lookup(type->gid())) return stream();

    if (auto ptr = type.isa<PtrType>()) {
        emit_aggop_decl(ptr->referenced_type());
        return stream();
    }
    if (auto array = type.isa<IndefiniteArrayType>()) {
        emit_aggop_decl(array->elem_type());
        return stream();
    }
    if (auto fn = type.isa<FnType>()) {
        for (auto type : fn->args()) emit_aggop_decl(type);
        return stream();
    }

    // recurse into (multi-dimensional) array
    if (auto array = type.isa<DefiniteArrayType>()) {
        emit_aggop_decl(array->elem_type());
        emit_type(array);
        insert(type->gid(), "array_" + std::to_string(type->gid()));
        return newline();
    }

    // recurse into (multi-dimensional) tuple
    if (auto tuple = type.isa<TupleType>()) {
        for (auto arg : tuple->args()) emit_aggop_decl(arg);
        emit_type(tuple);
        insert(type->gid(), "tuple_" + std::to_string(type->gid()));
        return newline();
    }

    // recurse into (multi-dimensional) struct
    if (auto struct_app = type.isa<StructAppType>()) {
        for (auto elem : struct_app->elems()) emit_aggop_decl(elem);
        emit_type(struct_app);
        insert(type->gid(), "struct_" + std::to_string(type->gid()));
        return newline();
    }

    return stream();
}


void CCodeGen::emit() {
    if (lang_==Lang::CUDA) {
        stream() << "extern \"C\" {\n";
        stream() << "__device__ inline int threadIdx_x() { return threadIdx.x; }\n";
        stream() << "__device__ inline int threadIdx_y() { return threadIdx.y; }\n";
        stream() << "__device__ inline int threadIdx_z() { return threadIdx.z; }\n";
        stream() << "__device__ inline int blockIdx_x() { return blockIdx.x; }\n";
        stream() << "__device__ inline int blockIdx_y() { return blockIdx.y; }\n";
        stream() << "__device__ inline int blockIdx_z() { return blockIdx.z; }\n";
        stream() << "__device__ inline int blockDim_x() { return blockDim.x; }\n";
        stream() << "__device__ inline int blockDim_y() { return blockDim.y; }\n";
        stream() << "__device__ inline int blockDim_z() { return blockDim.z; }\n";
        stream() << "__device__ inline int gridDim_x() { return gridDim.x; }\n";
        stream() << "__device__ inline int gridDim_y() { return gridDim.y; }\n";
        stream() << "__device__ inline int gridDim_z() { return gridDim.z; }\n";
        stream() << "__device__ inline int as_int(float val) { return __float_as_int(val); }\n";
        stream() << "__device__ inline int as_float(int val) { return __int_as_float(val); }\n";
    }
    if (lang_==Lang::OPENCL) {
        stream() << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
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
                    stream() << "texture<";
                    emit_type(param->type().as<PtrType>()->referenced_type());
                    stream() << ", cudaTextureType1D, cudaReadModeElementType> ";
                    stream() << param->name << ";";
                    newline();
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
        if (lang_==Lang::CUDA) stream() << "__device__ ";
        emit_type(ret_fn_type->args().back()) << " " << name << "(";
        size_t i = 0;
        for (auto param : lambda->params()) {
            if (param->order() == 0 && !param->is_mem()) {
                // skip arrays bound to texture memory
                if (is_texture_type(param->type())) continue;
                if (i++ > 0) stream() << ", ";
                emit_type(param->type());
                insert(param->gid(), param->unique_name());
            }
        }
        stream() << ");";
        newline();
    });
    newline();

    // emit all globals
    for (auto primop : world().primops()) {
        if (auto global = primop->isa<Global>()) {
            emit_aggop_decl(global->type());
            emit(global);
            newline();
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
                case Lang::C99:                               break;
                case Lang::CUDA:   stream() << "__global__ "; break;
                case Lang::OPENCL: stream() << "__kernel ";   break;
            }
        } else {
            if (lang_==Lang::CUDA) stream() << "__device__ ";
        }
        emit_type(ret_fn_type->args().back()) << " " << name << "(";
        size_t i = 0;
        // emit and store all first-order params
        for (auto param : lambda->params()) {
            if (param->order() == 0 && !param->is_mem()) {
                // skip arrays bound to texture memory
                if (is_texture_type(param->type())) continue;
                if (i++ > 0) stream() << ", ";
                if (lang_==Lang::OPENCL && lambda->is_external() &&
                    (param->type().isa<DefiniteArrayType>() ||
                     param->type().isa<StructAppType>() ||
                     param->type().isa<TupleType>())) {
                    // structs are passed via buffer; the parameter is a pointer to this buffer
                    stream() << "__global ";
                    emit_type(param->type()) << " *" << param->unique_name() << "_";
                } else {
                    emit_type(param->type()) << " " << param->unique_name();
                }
            }
        }
        stream() << ") {";
        ++indent;

        // emit and store all first-order params
        for (auto param : lambda->params()) {
            if (param->order() == 0 && !param->is_mem()) {
                if (lang_==Lang::OPENCL && lambda->is_external() &&
                    (param->type().isa<DefiniteArrayType>() ||
                     param->type().isa<StructAppType>() ||
                     param->type().isa<TupleType>())) {
                    // load struct from buffer
                    newline();
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
                        newline();
                        emit_type(param->type()) << "  " << param->unique_name() << ";";
                        newline();
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
            newline();

            // print label for the current basic block
            if (lambda != scope.entry()) {
                stream() << "l" << lambda->gid() << ": ;";
                up();
                // load params from phi node
                if (lambda->to() != ret_param) // skip for return
                    for (auto param : lambda->params())
                        if (!param->is_mem()) {
                            stream() << param->unique_name() << " = p" << param->unique_name() << ";";
                            newline();
                        }
            }

            for (auto primop : block) {
                // skip higher-order primops, stuff dealing with frames and all memory related stuff except stores
                if (!primop->type().isa<FnType>() && !primop->type().isa<FrameType>()
                        && (!primop->is_mem() || primop->isa<Store>())) {
                    emit_debug_info(primop);
                    emit(primop);
                    newline();
                }
            }

            // terminate bb
            if (lambda->to() == ret_param) { // return
                size_t num_args = lambda->num_args();
                stream() << "return ";
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
                stream() << ";";
            } else if (lambda->to() == world().branch()) {
                emit_debug_info(lambda->arg(0)); // TODO correct?
                stream() << "if (";
                emit(lambda->arg(0));
                stream() << ") ";
                emit(lambda->arg(1));
                stream() << " else ";
                emit(lambda->arg(2));
            } else if (lambda->to()->isa<Bottom>()) {
                stream() << "return ; // bottom: unreachable";
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
                            stream() << "p" << to_lambda->param(i)->unique_name() << " = ";
                            emit(lambda->arg(i)) << ";";
                            newline();
                        }
                    emit(to_lambda);
                } else {
                    if (to_lambda->is_intrinsic()) {
                        if (to_lambda->intrinsic() == Intrinsic::Reinterpret) {
                            auto cont = lambda->arg(2)->as_lambda();
                            emit(cont->param(1)) << " = as_";
                            emit_type(cont->param(1)->type()) << "(";
                            emit(lambda->arg(1)) << ");";
                            newline();
                        } else {
                            THORIN_UNREACHABLE;
                        }
                    } else {
                        auto emit_call = [&] () {
                            auto name = (to_lambda->is_external() || to_lambda->empty()) ? to_lambda->name : to_lambda->unique_name();
                            stream() << name << "(";
                            // emit all first-order args
                            size_t i = 0;
                            for (auto arg : lambda->args()) {
                                if (arg->order() == 0 && !arg->is_mem()) {
                                    if (i++ > 0) stream() << ", ";
                                    emit(arg);
                                }
                            }
                            stream() << ");";
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
                                emit(arg);
                                newline();
                            }
                        }

                        if (ret_arg == ret_param) {     // call + return
                            stream() << "return ";
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
                                newline();
                                // store argument to phi nodes
                                stream() << "p" << succ->param(1)->unique_name() << " = ";
                                emit(param) << ";";
                            }
                        }
                    }
                }
            }
            if (lambda != scope.entry()) --indent;
        }
        down();
        stream() << "}";
        newline();
        newline();

        primops_.clear();
    });

    globals_.clear();
    primops_.clear();
    if (lang_==Lang::CUDA) stream() << "}\n"; // extern "C"
}


std::ostream& CCodeGen::emit(Def def) {
    if (auto lambda = def->isa<Lambda>()) {
        return stream() << "goto l" << lambda->gid() << ";";
    }

    if (lookup(def->gid())) return stream() << get_name(def->gid());

    if (auto bin = def->isa<BinOp>()) {
        // emit definitions of inlined elements
        emit_aggop_defs(bin->lhs());
        emit_aggop_defs(bin->rhs());
        emit_type(bin->type()) << " " << bin->unique_name() << ";";
        newline() << bin->unique_name() << " = ";
        emit(bin->lhs());
        if (auto cmp = bin->isa<Cmp>()) {
            switch (cmp->cmp_kind()) {
                case Cmp_eq: stream() << " == "; break;
                case Cmp_ne: stream() << " != "; break;
                case Cmp_gt: stream() << " > ";  break;
                case Cmp_ge: stream() << " >= "; break;
                case Cmp_lt: stream() << " < ";  break;
                case Cmp_le: stream() << " <= "; break;
            }
        }

        if (auto arithop = bin->isa<ArithOp>()) {
            switch (arithop->arithop_kind()) {
                case ArithOp_add: stream() << " + ";  break;
                case ArithOp_sub: stream() << " - ";  break;
                case ArithOp_mul: stream() << " * ";  break;
                case ArithOp_div: stream() << " / ";  break;
                case ArithOp_rem: stream() << " % ";  break;
                case ArithOp_and: stream() << " & ";  break;
                case ArithOp_or:  stream() << " | ";  break;
                case ArithOp_xor: stream() << " ^ ";  break;
                case ArithOp_shl: stream() << " << "; break;
                case ArithOp_shr: stream() << " >> "; break;
            }
        }
        emit(bin->rhs());
        stream() << ";";
        return insert(def->gid(), def->unique_name());
    }

    if (auto conv = def->isa<ConvOp>()) {
        emit_type(conv->type()) << " " << conv->unique_name() << ";";
        newline() << conv->unique_name() << " = (";
        emit_type(conv->type()) << ")";
        emit(conv->from()) << ";";
        return insert(def->gid(), def->unique_name());
    }

    if (auto array = def->isa<DefiniteArray>()) { // DefArray is mapped to a struct
        // emit definitions of inlined elements
        for (auto op : array->ops()) emit_aggop_defs(op);

        emit_type(array->type()) << " " << array->unique_name() << ";";
        for (size_t i = 0, e = array->size(); i != e; ++i) {
            newline() << array->unique_name() << ".e[" << i << "] = ";
            emit(array->op(i)) << ";";
        }
        return insert(def->gid(), def->unique_name());
    }

    if (auto agg = def->isa<Aggregate>()) {
        assert(def->isa<Tuple>() || def->isa<StructAgg>() || def->isa<Vector>());
        // emit definitions of inlined elements
        for (auto op : agg->ops()) emit_aggop_defs(op);

        emit_type(agg->type()) << " " << agg->unique_name() << ";";
        char elem_prefix = (def->isa<Vector>()) ? 's' : 'e';
        for (size_t i = 0, e = agg->ops().size(); i != e; ++i) {
            newline() << agg->unique_name() << "." << elem_prefix << i << " = ";
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
            return stream();
        };

        if (auto extract = aggop->isa<Extract>()) {
            if (extract->is_mem() || extract->type().isa<FrameType>())
                return stream();
            emit_type(aggop->type()) << " " << aggop->unique_name() << ";";
            newline() << aggop->unique_name() << " = ";
            if (auto memop = extract->agg()->isa<MemOp>())
                emit(memop) << ";";
            else
                emit_access(aggop->agg(), aggop->index()) << ";";
            return insert(def->gid(), def->unique_name());
        }

        auto ins = def->as<Insert>();
        emit_type(aggop->type()) << " " << aggop->unique_name() << " = ";
        emit(ins->agg()) << ";";
        insert(def->gid(), aggop->unique_name());
        newline();
        emit_access(def, ins->index()) << " = ";
        emit(ins->value()) << ";";
        return stream();
    }

    if (auto primlit = def->isa<PrimLit>()) {
        auto kind = primlit->primtype_kind();
        if (kind == PrimType_bool) stream() << (primlit->bool_value() ? "true" : "false");
        else if (kind == PrimType_qs8) stream() << (int) primlit->qs8_value();
        else if (kind == PrimType_ps8) stream() << (int) primlit->ps8_value();
        else if (kind == PrimType_qu8) stream() << (unsigned) primlit->qu8_value();
        else if (kind == PrimType_pu8) stream() << (unsigned) primlit->pu8_value();
        else if (kind == PrimType_pf32 || kind==PrimType_qf32) {
            stream() << std::fixed << primlit->pf32_value() << 'f';
        } else if (kind == PrimType_pf64 || kind==PrimType_qf64) {
            stream() << std::fixed << primlit->pf64_value();
        } else {
            switch (kind) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: stream() << primlit->T##_value(); break;
#include "thorin/tables/primtypetable.h"
                default: THORIN_UNREACHABLE;
            }
        }
        return stream();
    }

    if (def->isa<Bottom>())
        return stream() << "42 /* bottom */";

    if (auto load = def->isa<Load>()) {
        emit_type(load->out_val()->type()) << " " << load->unique_name() << ";";
        newline() << load->unique_name() << " = ";
        // handle texture fetches
        if (!is_texture_type(load->ptr()->type())) stream() << "*";
        emit(load->ptr()) << ";";

        return insert(def->gid(), def->unique_name());
    }

    if (auto store = def->isa<Store>()) {
        emit_aggop_defs(store->val());
        stream() << "*";
        emit(store->ptr()) << " = ";
        emit(store->val()) << ";";

        return insert(def->gid(), def->unique_name());
    }

    if (auto slot = def->isa<Slot>()) {
        emit_type(slot->alloced_type()) << " " << slot->unique_name() << "_slot;";
        newline();
        emit_type(slot->alloced_type()) << "* " << slot->unique_name() << ";";
        newline() << slot->unique_name() << " = &" << slot->unique_name() << "_slot;";
        return insert(def->gid(), def->unique_name());
    }

    if (def->isa<Enter>())
        return stream();

    if (def->isa<Vector>()) {
        THORIN_UNREACHABLE;
    }

    if (auto lea = def->isa<LEA>()) {
        if (is_texture_type(lea->type())) { // handle texture fetches
            emit_type(lea->ptr_referenced_type()) << " " << lea->unique_name() << ";";
            newline() << lea->unique_name() << " = ";
            stream() << "tex1Dfetch(";
            emit(lea->ptr()) << ", ";
            emit(lea->index()) << ");";
        } else {
            emit_type(lea->type()) << " " << lea->unique_name() << ";";
            newline() << lea->unique_name() << " = ";
            if (lea->ptr_referenced_type().isa<TupleType>() || lea->ptr_referenced_type().isa<StructAppType>()) {
                stream() << "&";
                emit(lea->ptr()) << "->e";
                emit(lea->index()) << ";";
            } else if (lea->ptr_referenced_type().isa<DefiniteArrayType>()) {
                stream() << "&";
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
        if (lang_==Lang::CUDA) stream() << "__device__ ";
        if (lang_==Lang::OPENCL) stream() << "__constant ";
        emit_type(global->alloced_type()) << " " << global->unique_name() << "_slot";
        if (global->init()->isa<Bottom>()) {
            stream() << ";";
        } else {
            stream() << " = ";
            emit(global->init()) << ";";
        }
        newline();

        if (lang_==Lang::CUDA) stream() << "__device__ ";
        if (lang_==Lang::OPENCL) stream() << "__constant ";
        emit_type(global->alloced_type()) << " *" << global->unique_name() << " = &" << global->unique_name() << "_slot;";

        return insert(def->gid(), def->unique_name());
    }

    if (auto mmap = def->isa<Map>()) {
        assert(mmap->mem_size()->isa<PrimLit>() && "mmap: couldn't extract memory size");

        if (lang_==Lang::CUDA) {
            switch (mmap->addr_space()) {
                default: break;
                case AddressSpace::Shared: stream() << "__shared__ ";  break;
            }
        }
        if (lang_==Lang::OPENCL) {
            switch (mmap->addr_space()) {
                default: break;
                case AddressSpace::Global: stream() << "__global "; break;
                case AddressSpace::Shared: stream() << "__local ";  break;
            }
        }
        emit_type(mmap->out_ptr_type()->referenced_type()) << " " << mmap->unique_name() << "[";
        emit(mmap->mem_size()) << "];";

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
    return stream();
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
