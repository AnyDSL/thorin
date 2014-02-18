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

#include <unordered_map>

namespace thorin {

class CCodeGen : public Printer {
public:
    CCodeGen(World& world, std::ostream& stream, LangType lang)
        : Printer(stream)
        , world_(world)
        , lang_(lang)
    {}

    void emit();
    std::ostream& emit_tuple_decl(const Type *type);
    std::ostream& emit_type(const Type* type);
    std::ostream& emit(Def def);
private:
    World& world_;
    LangType lang_;
    std::unordered_map<int, std::string> primops_;
    std::unordered_set<int> gparams_;
};

std::ostream& CCodeGen::emit_type(const Type* type) {
    if (type == nullptr) {
        return stream() << "NULL";
    } else if (type->isa<Frame>()) {
        return stream();
    } else if (type->isa<Mem>()) {
        return stream() << "void";
    } else if (type->isa<Pi>()) {
        THORIN_UNREACHABLE;
    } else if (auto sigma = type->isa<Sigma>()) {
        if (primops_.count(sigma->gid()))
            return stream() << primops_[sigma->gid()];
        stream() << "typedef struct tuple_" << sigma->gid() << " {";
        ++indent;
        size_t i = 0;
        for (auto elem : sigma->elems()) {
            newline();
            emit_type(elem) << " e" << i++ << ";";
        }
        --indent; newline();
        stream() << "} tuple_" << sigma->gid() << ";";
        return stream();
    } else if (type->isa<Generic>()) {
        THORIN_UNREACHABLE;
    } else if (type->isa<GenericRef>()) {
        THORIN_UNREACHABLE;
    } else if (auto array = type->isa<IndefArray>()) {
        emit_type(array->elem_type());
        return stream();
    } else if (auto array = type->isa<DefArray>()) { // DefArray is mapped to tuple
        if (primops_.count(array->gid()))
            return stream() << primops_[array->gid()];
        stream() << "typedef struct array_" << array->gid() << " {";
        ++indent;
        for (size_t i = 0, dim = array->dim(); i < dim; ++i) {
            newline();
            emit_type(array->elem_type()) << " e" << i << ";";
        }
        --indent; newline();
        stream() << "} array_" << array->gid() << ";";
        return stream();
    } else if (auto ptr = type->isa<Ptr>()) {
        emit_type(ptr->referenced_type());
        stream() << '*';
        if (ptr->is_vector())
            stream() << ptr->referenced_type()->length();
        return stream();
    } else if (auto primtype = type->isa<PrimType>()) {
        switch (primtype->primtype_kind()) {
            case PrimType_bool:                     stream() << "bool";     break;
            case PrimType_ps8:  case PrimType_qs8:  stream() << "char";     break;
            case PrimType_pu8:  case PrimType_qu8:  stream() << "uchar";    break;
            case PrimType_ps16: case PrimType_qs16: stream() << "short";    break;
            case PrimType_pu16: case PrimType_qu16: stream() << "ushort";   break;
            case PrimType_ps32: case PrimType_qs32: stream() << "int";      break;
            case PrimType_pu32: case PrimType_qu32: stream() << "uint";     break;
            case PrimType_ps64: case PrimType_qs64: stream() << "long";     break;
            case PrimType_pu64: case PrimType_qu64: stream() << "ulong";    break;
            case PrimType_pf32: case PrimType_qf32: stream() << "float";    break;
            case PrimType_pf64: case PrimType_qf64: stream() << "double";   break;
        }
        if (primtype->is_vector())
            stream() << primtype->length();
        return stream();
    }
    THORIN_UNREACHABLE;
}


std::ostream& CCodeGen::emit_tuple_decl(const Type *type) {
    if (auto pi = type->isa<Pi>())
        for (auto type : pi->elems()) emit_tuple_decl(type);

    if (type->isa<Sigma>() || type->isa<DefArray>()) {
        if (!primops_.count(type->gid())) {
            emit_type(type);
            newline();
            primops_[type->gid()] = (type->isa<Sigma>()?"tuple_":"array_") +
                std::to_string(type->gid());
        }
    }

    return stream();
}


void CCodeGen::emit() {
    auto scopes = top_level_scopes(world_);

    // emit lambda and tuple declarations
    for (auto scope : scopes) {
        // tuple declarations
        for (auto lambda : scope->rpo()) {
            Schedule schedule = schedule_smart(*scope);

            for (auto param : lambda->params()) {
                emit_tuple_decl(param->type());
                primops_[param->gid()] = param->unique_name();
            }

            for (auto primop : schedule[lambda])
                emit_tuple_decl(primop->type());
        }

        auto lambda = scope->entry();
        if (lambda->is_builtin() || lambda->attribute().is(Lambda::Intrinsic))
            continue;

        // retrieve return param
        const Param *ret_param = nullptr;
        for (auto param : lambda->params()) {
            emit_tuple_decl(param->type());
            if (param->order() != 0) {
                assert(!ret_param);
                ret_param = param;
            }
        }
        assert(ret_param);

        const Pi *ret_fn_type = ret_param->type()->as<Pi>();
        if (lambda->attribute().is(Lambda::KernelEntry)) {
            if (lang_==OPENCL) stream() << "__global ";
            emit_type(ret_fn_type->elems().back()) << " " << lambda->name << "(";
        } else {
            emit_type(ret_fn_type->elems().back()) << " " << lambda->unique_name() << "(";
        }
        size_t i = 0;
        // emit all first-order params
        for (auto param : lambda->params()) {
            if (param->order() == 0 && !param->type()->isa<Mem>()) {
                if (i++ > 0) stream() << ", ";
                if (lang_==OPENCL && param->type()->isa<Ptr>()) {
                    stream() << "__global ";
                    gparams_.insert(param->gid());
                }
                emit_type(param->type());
                primops_[param->gid()] = param->unique_name();
            }
        }
        stream() << ");";
        newline();
    }
    newline();

    // emit all globals
    for (auto primop : world_.primops()) {
        if (auto global = primop->isa<Global>())
            emit(global);
    }

    // emit connected functions first
    std::stable_sort(scopes.begin(), scopes.end(), [] (Scope* s1, Scope* s2) { return s1->entry()->is_connected_to_builtin(); });

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

        const Pi *ret_fn_type = ret_param->type()->as<Pi>();
        if (lambda->attribute().is(Lambda::KernelEntry)) {
            if (lang_==OPENCL) stream() << "__global ";
            emit_type(ret_fn_type->elems().back()) << " " << lambda->name << "(";
        } else {
            emit_type(ret_fn_type->elems().back()) << " " << lambda->unique_name() << "(";
        }
        size_t i = 0;
        // emit and store all first-order params
        for (auto param : lambda->params()) {
            if (param->order() == 0 && !param->type()->isa<Mem>()) {
                if (i++ > 0) stream() << ", ";
                if (lang_==OPENCL && param->type()->isa<Ptr>()) {
                    stream() << "__global ";
                    gparams_.insert(param->gid());
                }
                emit_type(param->type()) << " " << param->unique_name();
            }
        }
        stream() << ") {";
        ++indent;

        for (auto lambda : scope.rpo()) {
            // dump declarations for variables set in gotos
            if (!lambda->is_cascading() && scope.entry() != lambda)
                for (auto param : lambda->params())
                    if (!param->type()->isa<Mem>()) {
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
                if (!primop->type()->isa<Pi>() && !primop->type()->isa<Frame>()
                        && (!primop->type()->isa<Mem>() || primop->isa<Store>())) {
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
                            if (lambda->arg(0)->type()->isa<Mem>())
                                break;
                            else
                                emit(lambda->arg(0));
                            break;
                    case 2: {
                                if (lambda->arg(0)->type()->isa<Mem>()) {
                                    emit(lambda->arg(1));
                                    break;
                                } else if (lambda->arg(1)->type()->isa<Mem>()) {
                                    emit(lambda->arg(0));
                                    break;
                                }
                                // FALLTHROUGH
                            }
                    default: {
                                THORIN_UNREACHABLE;
                             }
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
            } else {
                Lambda* to_lambda = lambda->to()->as_lambda();

                // emit inlined tuples before the call operation
                for (auto arg : lambda->args()) {
                    if (arg->isa<ArrayAgg>() || arg->isa<Tuple>()) {
                        if (!primops_.count(arg->gid())) {
                            emit(arg);
                            newline();
                        }
                    }
                }
                if (to_lambda->is_basicblock()) {    // ordinary jump
                    assert(to_lambda->num_params()==lambda->num_args());
                    size_t size = to_lambda->num_params();
                    for (size_t i = 0; i != size; ++i) {
                        if (!to_lambda->param(i)->type()->isa<Mem>()) {
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
                            if (to_lambda->attribute().is(Lambda::Intrinsic))
                                stream() << to_lambda->name << "(";
                            else
                                stream() << to_lambda->unique_name() << "(";
                            // emit all first-order args
                            size_t i = 0;
                            for (auto arg : lambda->args()) {
                                if (arg->order() == 0 && !arg->type()->isa<Mem>()) {
                                    if (i++ > 0) stream() << ", ";
                                    emit(arg);
                                }
                            }
                            stream() << ");";
                        } else {                        // call + continuation
                            Lambda* succ = ret_arg->as_lambda();
                            const Param* param = succ->param(0)->type()->isa<Mem>() ? nullptr : succ->param(0);
                            if (param == nullptr && succ->num_params() == 2)
                                param = succ->param(1);

                            if (param) {
                                emit_type(param->type()) << " ";
                                emit(param) << " = ";
                            }
                            if (to_lambda->attribute().is(Lambda::Intrinsic))
                                stream() << to_lambda->name << "(";
                            else
                                stream() << to_lambda->unique_name() << "(";
                            // emit all first-order args
                            size_t i = 0;
                            for (auto arg : lambda->args()) {
                                if (arg->order() == 0 && !arg->type()->isa<Mem>()) {
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
    }
    primops_.clear();
    gparams_.clear();
}

std::ostream& CCodeGen::emit(Def def) {
    if (auto lambda = def->isa<Lambda>()) {
        return stream() << "goto l" << lambda->gid() << ";";
    }

    if (primops_.count(def->gid()))
        return stream() << primops_[def->gid()];

    if (auto bin = def->isa<BinOp>()) {
        emit_type(def->type()) << " " << def->unique_name() << " = ";
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
        primops_[def->gid()] = def->unique_name();
        return stream();
    }

    if (def->isa<ConvOp>()) {
        THORIN_UNREACHABLE;
    }

    if (auto array = def->isa<ArrayAgg>()) {
        if (array->is_const()) {
            // DefArray is mapped to Tuple
            emit_type(array->type()) << " " << array->unique_name() << " = {";
            for (size_t i = 0, e = array->size(); i != e; ++i) {
                if (i) stream() << ", ";
                emit(array->op(i));
            }
            stream() << "};";
            primops_[def->gid()] = def->unique_name();
            return stream();
        }
        THORIN_UNREACHABLE;
    }

    if (auto tuple = def->isa<Tuple>()) {
        emit_type(tuple->type()) << " " << tuple->unique_name() << " = {";
        for (size_t i = 0, e = tuple->ops().size(); i != e; ++i) {
            if (i) stream() << ", ";
            emit(tuple->op(i));
        }
        stream() << "};";
        primops_[def->gid()] = def->unique_name();
        return stream();
    }

    if (auto aggop = def->isa<AggOp>()) {
        if (!primops_.count(aggop->agg()->gid())) {
            emit(aggop->agg());
            newline();
        }
        if (aggop->agg_type()->isa<Sigma>()) {
            if (aggop->isa<Extract>()) {
                emit_type(aggop->type()) << " " << aggop->unique_name() << " = ";
                emit(aggop->agg()) << ".e";
                emit(aggop->index()) << ";";
                primops_[def->gid()] = def->unique_name();
            } else {
                emit(aggop->agg()) << ".e";
                emit(aggop->index()) << " = ";
                emit(aggop->as<Insert>()->value()) << ";";
                primops_[def->gid()] = aggop->agg()->unique_name();
            }
        } else if (aggop->agg_type()->isa<ArrayType>()) {
            if (aggop->isa<Extract>()) {
                emit_type(aggop->type()) << " " << aggop->unique_name() << " = ";
                emit(aggop->agg()) << "[";
                emit(aggop->index()) << "];";
                primops_[def->gid()] = def->unique_name();
            } else {
                emit(aggop->agg()) << "[";
                emit(aggop->index()) << "] = ";
                emit(aggop->as<Insert>()->value()) << ";";
                primops_[def->gid()] = aggop->agg()->unique_name();
            }
        } else {
            THORIN_UNREACHABLE;
        }

        return stream();
    }

    if (auto primlit = def->isa<PrimLit>()) {
        switch (primlit->primtype_kind()) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: stream() << primlit->T##_value(); break;
#include "thorin/tables/primtypetable.h"
            default: THORIN_UNREACHABLE; break;
        }
        return stream();
    }

    if (def->isa<Undef>()) {   // bottom and any
        return stream() << "42";
    }

    if (auto load = def->isa<Load>()) {
        emit_type(load->type()) << " " << load->unique_name() << " = *";
        emit(load->ptr()) << ";";

        primops_[def->gid()] = def->unique_name();
        return stream();
    }

    if (auto store = def->isa<Store>()) {
        stream() << "*";
        emit(store->ptr()) << " = ";
        emit(store->val()) << ";";

        primops_[def->gid()] = def->unique_name();
        return stream();
    }

    if (def->isa<Slot>())
        THORIN_UNREACHABLE;

    if (def->isa<Enter>() || def->isa<Leave>())
        return stream();

    if (def->isa<Vector>()) {
        THORIN_UNREACHABLE;
    }

    if (auto lea = def->isa<LEA>()) {
        if (lang_==OPENCL && gparams_.count(lea->ptr()->gid()))
            stream() << "__global ";
        emit_type(lea->type()) << " " << lea->unique_name() << " = ";
        emit(lea->ptr()) << " + ";
        emit(lea->index()) << ";";

        primops_[def->gid()] = def->unique_name();
        return stream();
    }

    THORIN_UNREACHABLE;
}

//------------------------------------------------------------------------------

void emit_c(World& world, std::ostream& stream, LangType lang) { CCodeGen(world, stream, lang).emit(); }

//------------------------------------------------------------------------------

}
