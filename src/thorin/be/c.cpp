#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/memop.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/util/autoptr.h"
#include "thorin/util/printer.h"
#include "thorin/be/c.h"

namespace thorin {

class CCodeGen : public Printer {
public:
    CCodeGen(World& world, std::ostream& stream, LangType lang)
        : Printer(stream)
        , world_(world)
        , lang_(lang)
        , process_kernel(false)
    {}

    void emit();
private:
    World& world_;
    LangType lang_;
    HashMap<size_t, std::string> globals_;
    HashMap<size_t, std::string> primops_;

    std::ostream& emit_aggop_defs(Def def);
    std::ostream& emit_aggop_decl(Type);
    std::ostream& emit_type(Type);
    std::ostream& emit(Def def);
    bool lookup(size_t gid);
    void insert(size_t gid, std::string str);
    std::string &get_name(size_t gid);
    bool is_texture_type(Type type);
    bool process_kernel;
};

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
        if (lang_==CUDA) {
            switch (ptr->addr_space()) {
                default: break;
                // only declaration need __shared__
                case AddressSpace::Shared: stream() << "";  break;
            }
        }
        if (lang_==OPENCL) {
            switch (ptr->addr_space()) {
                default: break;
                case AddressSpace::Generic: // once address spaces are correct, ::Global should be sufficient
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
        newline();
    }

    // recurse into (multi-dimensional) tuple
    if (auto tuple = type.isa<TupleType>()) {
        // check for a memory-mapped extract: map() -> (mem, [type]*[dev][mem])
        if (tuple->num_args() == 2 &&
            tuple->arg(0).isa<MemType>() && tuple->arg(1).isa<PtrType>()) {
            insert(type->gid(), "");
            return stream();
        }
        for (auto arg : tuple->args()) emit_aggop_decl(arg);
        emit_type(tuple);
        insert(type->gid(), "tuple_" + std::to_string(type->gid()));
        newline();
    }

    // recurse into (multi-dimensional) struct
    if (auto struct_app = type.isa<StructAppType>()) {
        for (auto elem : struct_app->elems()) emit_aggop_decl(elem);
        emit_type(struct_app);
        insert(type->gid(), "struct_" + std::to_string(type->gid()));
        newline();
    }

    return stream();
}


void CCodeGen::emit() {
    if (lang_==CUDA) {
        stream() << "extern \"C\" {\n";
        stream() << "__device__ int threadIdx_x() { return threadIdx.x; }\n";
        stream() << "__device__ int threadIdx_y() { return threadIdx.y; }\n";
        stream() << "__device__ int threadIdx_z() { return threadIdx.z; }\n";
        stream() << "__device__ int blockIdx_x() { return blockIdx.x; }\n";
        stream() << "__device__ int blockIdx_y() { return blockIdx.y; }\n";
        stream() << "__device__ int blockIdx_z() { return blockIdx.z; }\n";
        stream() << "__device__ int blockDim_x() { return blockDim.x; }\n";
        stream() << "__device__ int blockDim_y() { return blockDim.y; }\n";
        stream() << "__device__ int blockDim_z() { return blockDim.z; }\n";
        stream() << "__device__ int gridDim_x() { return gridDim.x; }\n";
        stream() << "__device__ int gridDim_y() { return gridDim.y; }\n";
        stream() << "__device__ int gridDim_z() { return gridDim.z; }\n";
    }

    auto scopes = top_level_scopes(world_);

    // emit declarations
    for (auto scope : scopes) {
        Schedule schedule = schedule_smart(*scope);

        // tuple declarations
        for (auto lambda : scope->rpo()) {
            for (auto param : lambda->params()) {
                emit_aggop_decl(param->type());
                insert(param->gid(), param->unique_name());
            }

            for (auto primop : schedule[lambda]) {
                emit_aggop_decl(primop->type());
                // search for inlined tuples/arrays
                if (auto aggop = primop->isa<AggOp>()) {
                    emit_aggop_decl(aggop->agg()->type());
                }
            }
        }

        // lambda declarations
        auto lambda = scope->entry();
        if (lambda->is_builtin() || lambda->attribute().is(Lambda::Device))
            continue;

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
            if (param->order() == 0 && !param->type().isa<MemType>()) {
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
        auto ret_fn_type = ret_param->type().as<FnType>();
        if (lambda->attribute().is(Lambda::KernelEntry)) {
            if (lang_==CUDA) stream() << "__global__ ";
            if (lang_==OPENCL) stream() << "__kernel ";
        } else {
            if (lang_==CUDA) stream() << "__device__ ";
        }
        if (lambda->attribute().is(Lambda::Extern)) {
            emit_type(ret_fn_type->args().back()) << " " << lambda->name << "(";
        } else {
            emit_type(ret_fn_type->args().back()) << " " << lambda->unique_name() << "(";
        }
        size_t i = 0;
        // emit all first-order params
        for (auto param : lambda->params()) {
            if (param->order() == 0 && !param->type().isa<MemType>()) {
                // skip arrays bound to texture memory
                if (is_texture_type(param->type())) continue;
                if (i++ > 0) stream() << ", ";
                emit_type(param->type());
                insert(param->gid(), param->unique_name());
            }
        }
        stream() << ");";
        newline();
    }
    newline();

    // emit all globals
    for (auto primop : world_.primops()) {
        if (auto global = primop->isa<Global>()) {
            emit_aggop_decl(global->type());
            emit(global);
            newline();
        }
    }

    // emit connected functions first
    std::stable_sort(scopes.begin(), scopes.end(), [] (Scope* s1, Scope* s2) { return s1->entry()->is_connected_to_builtin(); });
    process_kernel = true;

    for (auto ptr_scope : scopes) {
        auto& scope = *ptr_scope;
        auto lambda = scope.entry();
        if (lambda->is_builtin() || lambda->empty())
            continue;

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
        if (lambda->attribute().is(Lambda::KernelEntry)) {
            if (lang_==CUDA) stream() << "__global__ ";
            if (lang_==OPENCL) stream() << "__kernel ";
            emit_type(ret_fn_type->args().back()) << " " << lambda->name << "(";
        } else {
            if (lang_==CUDA) stream() << "__device__ ";
            emit_type(ret_fn_type->args().back()) << " " << lambda->unique_name() << "(";
        }
        size_t i = 0;
        // emit and store all first-order params
        for (auto param : lambda->params()) {
            if (param->order() == 0 && !param->type().isa<MemType>()) {
                // skip arrays bound to texture memory
                if (is_texture_type(param->type())) continue;
                if (i++ > 0) stream() << ", ";
                emit_type(param->type()) << " " << param->unique_name();
            }
        }
        stream() << ") {";
        ++indent;

        for (auto lambda : scope.rpo()) {
            // dump declarations for variables set in gotos
            if (!lambda->is_cascading() && scope.entry() != lambda)
                for (auto param : lambda->params())
                    if (!param->type().isa<MemType>()) {
                        newline();
                        emit_type(param->type()) << " " << param->unique_name() << ";";
                    }
        }

        // never use early schedule here - this may break memory operations
        Schedule schedule = schedule_smart(scope);

        // emit body for each bb
        for (auto lambda : scope.rpo()) {
            if (lambda->empty())
                continue;
            assert(lambda == scope.entry() || lambda->is_basicblock());
            newline();

            // print label for the current basic block
            if (lambda != scope.entry()) {
                stream() << "l" << lambda->gid() << ": ;";
                up();
            }

            for (auto primop : schedule[lambda]) {
                // skip higher-order primops, stuff dealing with frames and all memory related stuff except stores
                if (!primop->type().isa<FnType>() && !primop->type().isa<FrameType>()
                        && (!primop->type().isa<MemType>() || primop->isa<Store>())) {
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
                        if (lambda->arg(0)->type().isa<MemType>())
                            break;
                        else
                            emit(lambda->arg(0));
                        break;
                    case 2:
                        if (lambda->arg(0)->type().isa<MemType>()) {
                            emit(lambda->arg(1));
                            break;
                        } else if (lambda->arg(1)->type().isa<MemType>()) {
                            emit(lambda->arg(0));
                            break;
                        }
                        // FALLTHROUGH
                    default:
                        THORIN_UNREACHABLE;
                }
                stream() << ";";
            } else if (auto select = lambda->to()->isa<Select>()) { // conditional branch
                stream() << "if (";
                emit(select->cond());
                stream() << ") {";
                up(); emit(select->tval()); down();
                stream() << "} else {";
                up(); emit(select->fval()); down();
                stream() << "}";
            } else if (lambda->to()->isa<Bottom>()) {
                stream() << "return ; // bottom: unreachable";
            } else {
                Lambda* to_lambda = lambda->to()->as_lambda();

                // emit inlined arrays/tuples/structs before the call operation
                for (auto arg : lambda->args()) emit_aggop_defs(arg);

                if (to_lambda->is_basicblock()) {    // ordinary jump
                    assert(to_lambda->num_params()==lambda->num_args());
                    size_t size = to_lambda->num_params();
                    for (size_t i = 0; i != size; ++i) {
                        if (!to_lambda->param(i)->type().isa<MemType>()) {
                            if (to_lambda->param(i)->gid() == lambda->arg(i)->gid())
                                stream() << "// ";  // self-assignment
                            stream() << to_lambda->param(i)->unique_name() << " = ";
                            emit(lambda->arg(i)) << ";";
                            newline();
                        }
                    }
                    emit(to_lambda);
                } else {
                    if (to_lambda->is_builtin()) {
                        THORIN_UNREACHABLE;
                    } else {
                        // retrieve return argument
                        Def ret_arg = 0;
                        for (auto arg : lambda->args()) {
                            if (arg->order() != 0) {
                                assert(!ret_arg);
                                ret_arg = arg;
                            }
                        }

                        if (ret_arg == ret_param) {     // call + return
                            stream() << "return ";
                            if (to_lambda->attribute().is(Lambda::Extern | Lambda::Device))
                                stream() << to_lambda->name << "(";
                            else
                                stream() << to_lambda->unique_name() << "(";
                            // emit all first-order args
                            size_t i = 0;
                            for (auto arg : lambda->args()) {
                                if (arg->order() == 0 && !arg->type().isa<MemType>()) {
                                    if (i++ > 0) stream() << ", ";
                                    emit(arg);
                                }
                            }
                            stream() << ");";
                        } else {                        // call + continuation
                            Lambda* succ = ret_arg->as_lambda();
                            const Param* param = succ->param(0)->type().isa<MemType>() ? nullptr : succ->param(0);
                            if (param == nullptr && succ->num_params() == 2)
                                param = succ->param(1);

                            if (param) {
                                emit_type(param->type()) << " ";
                                emit(param) << ";";
                                newline();
                                emit(param) << " = ";
                            }
                            if (to_lambda->attribute().is(Lambda::Extern | Lambda::Device))
                                stream() << to_lambda->name << "(";
                            else
                                stream() << to_lambda->unique_name() << "(";
                            // emit all first-order args
                            size_t i = 0;
                            for (auto arg : lambda->args()) {
                                if (arg->order() == 0 && !arg->type().isa<MemType>()) {
                                    if (i++ > 0) stream() << ", ";
                                    emit(arg);
                                }
                            }
                            stream() << ");";
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
    }

    globals_.clear();
    primops_.clear();
    if (lang_==CUDA) stream() << "}\n"; // extern "C"
}


std::ostream& CCodeGen::emit(Def def) {
    if (auto lambda = def->isa<Lambda>()) {
        return stream() << "goto l" << lambda->gid() << ";";
    }

    if (lookup(def->gid())) return stream() << get_name(def->gid());

    if (auto bin = def->isa<BinOp>()) {
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
        insert(def->gid(), def->unique_name());
        return stream();
    }

    if (auto conv = def->isa<ConvOp>()) {
        emit_type(conv->type()) << " " << conv->unique_name() << ";";
        newline() << conv->unique_name() << " = (";
        emit_type(conv->type()) << ")";
        emit(conv->from()) << ";";
        insert(def->gid(), def->unique_name());
        return stream();
    }

    if (auto array = def->isa<DefiniteArray>()) {
        if (array->is_const()) { // DefArray is mapped to a struct
            // emit definitions of inlined elements
            for (auto op : array->ops()) emit_aggop_defs(op);

            emit_type(array->type()) << " " << array->unique_name() << ";";
            for (size_t i = 0, e = array->size(); i != e; ++i) {
                newline() << array->unique_name() << ".e[" << i << "] = ";
                emit(array->op(i)) << ";";
            }
            insert(def->gid(), def->unique_name());
            return stream();
        }
        THORIN_UNREACHABLE;
    }

    if (auto agg = def->isa<Aggregate>()) {
        assert(def->isa<Tuple>() || def->isa<StructAgg>());
        // emit definitions of inlined elements
        for (auto op : agg->ops()) emit_aggop_defs(op);

        emit_type(agg->type()) << " " << agg->unique_name() << ";";
        for (size_t i = 0, e = agg->ops().size(); i != e; ++i) {
            newline() << agg->unique_name() << ".e" << i << " = ";
            emit(agg->op(i)) << ";";
        }
        insert(def->gid(), def->unique_name());
        return stream();
    }

    if (auto aggop = def->isa<AggOp>()) {
        // emit definitions of inlined elements
        emit_aggop_defs(aggop->agg());
        if (aggop->agg()->type().isa<TupleType>() ||
            aggop->agg()->type().isa<StructAppType>()) {
            if (auto extract = aggop->isa<Extract>()) {
                emit_type(aggop->type()) << " " << aggop->unique_name() << ";";
                newline() << aggop->unique_name() << " = ";
                auto agg_type = extract->agg()->type();
                auto agg_tuple = agg_type.isa<TupleType>();
                // check for a memory-mapped extract: map() -> (mem, [type]*[dev][mem])
                if (agg_tuple && agg_tuple->num_args() == 2 &&
                    agg_tuple->arg(0).isa<MemType>() &&
                    agg_tuple->arg(1).isa<PtrType>()) {
                    emit(aggop->agg()) << ";";
                } else {
                    emit(aggop->agg()) << ".e";
                    emit(aggop->index()) << ";";
                }
                insert(def->gid(), def->unique_name());
            } else {
                emit(aggop->agg()) << ".e";
                emit(aggop->index()) << " = ";
                emit(aggop->as<Insert>()->value()) << ";";
                insert(def->gid(), aggop->agg()->unique_name());
            }
        } else if (aggop->agg()->type().isa<ArrayType>()) {
            if (aggop->isa<Extract>()) {
                emit_type(aggop->type()) << " " << aggop->unique_name() << ";";
                newline() << aggop->unique_name() << " = ";
                emit(aggop->agg()) << ".e[";
                emit(aggop->index()) << "];";
                insert(def->gid(), def->unique_name());
            } else {
                emit(aggop->agg()) << ".e[";
                emit(aggop->index()) << "] = ";
                emit(aggop->as<Insert>()->value()) << ";";
                insert(def->gid(), aggop->agg()->unique_name());
            }
        } else {
            THORIN_UNREACHABLE;
        }

        return stream();
    }

    if (auto primlit = def->isa<PrimLit>()) {
        auto kind = primlit->primtype_kind();
        if (kind == PrimType_bool) {
            stream() << (primlit->bool_value() ? "true" : "false");
        } else if (kind==PrimType_pf32 || kind==PrimType_qf32) {
            stream() << std::fixed << primlit->pf32_value();
            if ((lang_==C99 || lang_==CUDA)) stream() << 'f';
        } else if (kind==PrimType_pf64 || kind==PrimType_qf64) {
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

    if (def->isa<Undef>()) {   // bottom and any
        return stream() << "42 /* undef: bottom and any */";
    }

    if (auto load = def->isa<Load>()) {
        emit_type(load->type()) << " " << load->unique_name() << ";";
        newline() << load->unique_name() << " = ";
        // handle texture fetches
        if (!is_texture_type(load->ptr()->type())) stream() << "*";
        emit(load->ptr()) << ";";

        insert(def->gid(), def->unique_name());
        return stream();
    }

    if (auto store = def->isa<Store>()) {
        emit_aggop_defs(store->val());
        stream() << "*";
        emit(store->ptr()) << " = ";
        emit(store->val()) << ";";

        insert(def->gid(), def->unique_name());
        return stream();
    }

    if (auto slot = def->isa<Slot>()) {
        emit_type(slot->ptr_type()->referenced_type()) << " " << slot->unique_name() << "_slot;";
        newline();
        emit_type(slot->ptr_type()->referenced_type()) << "* " << slot->unique_name() << ";";
        newline() << slot->unique_name() << " = &" << slot->unique_name() << "_slot;";
        insert(def->gid(), def->unique_name());
        return stream();
    }

    if (def->isa<Enter>() || def->isa<Leave>())
        return stream();

    if (def->isa<Vector>()) {
        THORIN_UNREACHABLE;
    }

    if (auto lea = def->isa<LEA>()) {
        if (is_texture_type(lea->type())) { // handle texture fetches
            emit_type(lea->referenced_type()) << " " << lea->unique_name() << ";";
            newline() << lea->unique_name() << " = ";
            stream() << "tex1Dfetch(";
            emit(lea->ptr()) << ", ";
            emit(lea->index()) << ");";
        } else {
            emit_type(lea->type()) << " " << lea->unique_name() << ";";
            newline() << lea->unique_name() << " = ";
            if (lea->referenced_type().isa<TupleType>() ||
                lea->referenced_type().isa<StructAppType>()) {
                stream() << "&";
                emit(lea->ptr()) << "->e";
                emit(lea->index()) << ";";
            } else if (lea->referenced_type().isa<DefiniteArrayType>()) {
                stream() << "&";
                emit(lea->ptr()) << "->e[";
                emit(lea->index()) << "];";
            } else {
                emit(lea->ptr()) << " + ";
                emit(lea->index()) << ";";
            }
        }

        insert(def->gid(), def->unique_name());
        return stream();
    }

    if (auto global = def->isa<Global>()) {
        assert(!global->init()->isa_lambda() && "no global init lambda supported");
        if (lang_==CUDA) stream() << "__device__ ";
        if (lang_==OPENCL) stream() << "__constant ";
        emit_type(global->referenced_type()) << " " << global->unique_name();
        if (global->init()->isa<Bottom>()) {
            stream() << ";";
        } else {
            stream() << " = ";
            emit(global->init()) << ";";
        }

        insert(def->gid(), def->unique_name());
        return stream();
    }

    if (auto map = def->isa<Map>()) {
        assert(map->mem_size()->isa<PrimLit>() && "couldn't extract memory size");

        if (lang_==CUDA) {
            switch (map->addr_space()) {
                default: break;
                case AddressSpace::Shared: stream() << "__shared__ ";  break;
            }
        }
        if (lang_==OPENCL) {
            switch (map->addr_space()) {
                default: break;
                case AddressSpace::Global: stream() << "__global "; break;
                case AddressSpace::Shared: stream() << "__local ";  break;
            }
        }
        emit_type(map->ptr_type()->referenced_type()) << " " << map->unique_name() << "[";
        emit(map->mem_size()) << "];";

        insert(def->gid(), def->unique_name());
        return stream();
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
    assert(false && "couldn't find def");
}
void CCodeGen::insert(size_t gid, std::string str) {
    if (process_kernel) primops_[gid] = str;
    else globals_[gid] = str;
}
bool CCodeGen::is_texture_type(Type type) {
    if (auto ptr = type.isa<PtrType>()) {
        if (ptr->addr_space()==AddressSpace::Texture) {
            assert(lang_==CUDA && "Textures currently only supported in CUDA");
            return true;
        }
    }
    return false;
}


//------------------------------------------------------------------------------

void emit_c(World& world, std::ostream& stream, LangType lang) { CCodeGen(world, stream, lang).emit(); }

//------------------------------------------------------------------------------

}
