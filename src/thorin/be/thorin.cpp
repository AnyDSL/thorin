#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/bb_schedule.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/printer.h"

namespace thorin {

class CodeGen : public Printer {
public:
    CodeGen(bool fancy, bool colored = false)
        : Printer(std::cout, fancy, colored)
    {}

    std::ostream& emit_type_vars(Type);
    std::ostream& emit_type_args(Type);
    std::ostream& emit_type_elems(Type);
    std::ostream& emit_type(Type);
    std::ostream& emit_name(Def);
    std::ostream& emit_def(Def);
    std::ostream& emit_primop(const PrimOp*);
    std::ostream& emit_assignment(const PrimOp*);
    std::ostream& emit_head(const Lambda*);
    std::ostream& emit_jump(const Lambda*);
};

//------------------------------------------------------------------------------

std::ostream& CodeGen::emit_type_vars(Type type) {
    if (type->num_type_vars() != 0)
        return dump_list([&](TypeVar type_var) { emit_type(type_var); }, type->type_vars(), "[", "]");
    return stream();
}

std::ostream& CodeGen::emit_type_args(Type type) {
    return dump_list([&](Type type) { emit_type(type); }, type->args(), "(", ")");
}

std::ostream& CodeGen::emit_type_elems(Type type) {
    if (auto struct_app = type.isa<StructAppType>())
        return dump_list([&](Type type) { emit_type(type); }, struct_app->elems(), "{", "}");
    return emit_type_args(type);
}

std::ostream& CodeGen::emit_type(Type type) {
    if (type.empty()) {
        return stream() << "<NULL>";
    } else if (type.isa<FrameType>()) {
        return stream() << "frame";
    } else if (type.isa<MemType>()) {
        return stream() << "mem";
    } else if (auto fn = type.isa<FnType>()) {
        stream() << "fn";
        emit_type_vars(fn);
        return emit_type_args(fn);
    } else if (auto tuple = type.isa<TupleType>()) {
        emit_type_vars(tuple);
        return emit_type_args(tuple);
    } else if (auto struct_abs = type.isa<StructAbsType>()) {
        stream() << struct_abs->name();
        return emit_type_vars(struct_abs);
        // TODO emit args - but don't do this inline: structs may be recursive
        //return emit_type_args(struct_abs);
    } else if (auto struct_app = type.isa<StructAppType>()) {
        stream() << struct_app->struct_abs_type()->name();
        return emit_type_elems(struct_app);
    } else if (auto type_var = type.isa<TypeVar>()) {
        return stream() << '<' << type_var->gid() << '>';
    } else if (auto array = type.isa<IndefiniteArrayType>()) {
        stream() << '[';
        emit_type(array->elem_type());
        return stream() << ']';
    } else if (auto array = type.isa<DefiniteArrayType>()) {
        stream() << '[' << array->dim() << " x ";
        emit_type(array->elem_type());
        return stream() << ']';
    } else if (auto ptr = type.isa<PtrType>()) {
        if (ptr->is_vector())
            stream() << '<' << ptr->length() << " x ";
        emit_type(ptr->referenced_type());
        stream() << '*';
        if (ptr->is_vector())
            stream() << '>';
        auto device = ptr->device();
        if (device != -1)
            stream() << '[' << device << ']';
        switch (ptr->addr_space()) {
            case AddressSpace::Global:   stream() << "[Global]";   break;
            case AddressSpace::Texture:  stream() << "[Tex]";      break;
            case AddressSpace::Shared:   stream() << "[Shared]";   break;
            case AddressSpace::Constant: stream() << "[Constant]"; break;
            default: /* ignore unknown address space */            break;
        }
        return stream();
    } else if (auto primtype = type.isa<PrimType>()) {
        if (primtype->is_vector())
            stream() << "<" << primtype->length() << " x ";
            switch (primtype->primtype_kind()) {
#define THORIN_ALL_TYPE(T, M) case Node_PrimType_##T: stream() << #T; break;
#include "thorin/tables/primtypetable.h"
                default: THORIN_UNREACHABLE;
            }
        if (primtype->is_vector())
            stream() << ">";
        return stream();
    }
    THORIN_UNREACHABLE;
}

std::ostream& CodeGen::emit_def(Def def) {
    if (auto primop = def->isa<PrimOp>())
        return emit_primop(primop);
    return emit_name(def);
}

std::ostream& CodeGen::emit_name(Def def) {
    if (is_fancy()) // elide white = 0 and black = 7
        color(def->gid() % 6 + 30 + 1);

    stream() << def->unique_name();

    if (is_fancy())
        reset_color();

    return stream();
}

std::ostream& CodeGen::emit_primop(const PrimOp* primop) {
    if (primop->is_proxy())
        stream() << "<proxy>";
    else if (auto primlit = primop->isa<PrimLit>()) {
        emit_type(primop->type()) << ' ';
        auto kind = primlit->primtype_kind();

        // print i8 as ints
        if (kind == PrimType_qs8)
            stream() << (int) primlit->qs8_value();
        else if (kind == PrimType_ps8)
            stream() << (int) primlit->ps8_value();
        else if (kind == PrimType_qu8)
            stream() << (unsigned) primlit->qu8_value();
        else if (kind == PrimType_pu8)
            stream() << (unsigned) primlit->pu8_value();
        else {
            switch (kind) {
#define THORIN_ALL_TYPE(T, M) case PrimType_##T: stream() << primlit->T##_value(); break;
#include "thorin/tables/primtypetable.h"
                default: THORIN_UNREACHABLE;
            }
        }
    } else if (primop->isa<Global>()) {
        emit_name(primop);
    } else if (primop->is_const()) {
        if (primop->empty()) {
            stream() << primop->op_name() << ' ';
            emit_type(primop->type());
        } else {
            stream() << '(';
            if (primop->isa<PrimLit>())
                emit_type(primop->type()) << ' ';
            stream() << primop->op_name();
            dump_list([&](Def def) { emit_def(def); }, primop->ops(), " ", ")");
        }
    } else
        emit_name(primop);

    return stream();
}

std::ostream& CodeGen::emit_assignment(const PrimOp* primop) {
    emit_type(primop->type()) << " ";
    emit_name(primop) << " = ";

    auto ops = primop->ops();
    if (auto vectorop = primop->isa<VectorOp>()) {
        if (!vectorop->cond()->is_allset()) {
            stream() << "@ ";
            emit_name(vectorop->cond()) << " ";
        }
        ops = ops.slice_from_begin(1);
    }

    stream() << primop->op_name() << " ";
    dump_list([&](Def def) { emit_def(def); }, ops);
    return newline();
}

std::ostream& CodeGen::emit_head(const Lambda* lambda) {
    emit_name(lambda);
    emit_type_vars(lambda->type());
    dump_list([&](const Param* param) { emit_type(param->type()) << " "; emit_name(param); }, lambda->params(), "(", ")");

    if (lambda->is_external())
        stream() << " extern ";
    if (lambda->cc() == CC::Device)
        stream() << " device ";

    return up();
}

std::ostream& CodeGen::emit_jump(const Lambda* lambda) {
    if (!lambda->empty()) {
        emit_def(lambda->to());
        dump_list([&](Def def) { emit_def(def); }, lambda->args(), " ", "");
    }
    return down();
}

//------------------------------------------------------------------------------

void emit_thorin(const Scope& scope, bool fancy, bool nocolor) {
    CodeGen cg(fancy, nocolor);
    auto domtree = scope.domtree();
    Schedule schedule = schedule_smart(scope);
    auto bbs = bb_schedule(scope);
    for (auto lambda : bbs) {
        int depth = fancy ? domtree->depth(lambda) : 0;
        cg.indent += depth;
        cg.newline();
        cg.emit_head(lambda);

        for (auto op : schedule[lambda])
            cg.emit_assignment(op);

        cg.emit_jump(lambda);
        cg.indent -= depth;
    }
    cg.newline();
}

void emit_thorin(World& world, bool fancy, bool nocolor) {
    CodeGen cg(fancy, nocolor);
    cg.stream() << "module '" << world.name() << "'\n\n";

    for (auto primop : world.primops()) {
        if (auto global = primop->isa<Global>())
            cg.emit_assignment(global);
    }

    top_level_scopes<false>(world, [&] (const Scope& scope) { emit_thorin(scope, fancy, nocolor); });
}

void emit_type(Type type)                  { CodeGen(false).emit_type(type);         }
void emit_def(Def def)                     { CodeGen(false).emit_def(def);           }
void emit_head(const Lambda* lambda)       { CodeGen(false).emit_head(lambda);       }
void emit_jump(const Lambda* lambda)       { CodeGen(false).emit_jump(lambda);       }
void emit_assignment(const PrimOp* primop) { CodeGen(false).emit_assignment(primop); }

//------------------------------------------------------------------------------

}
