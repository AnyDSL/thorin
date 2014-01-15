#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/util/printer.h"

namespace thorin {

class CodeGen : public Printer {
public:
    CodeGen(bool fancy, bool colored = false)
        : Printer(std::cout, fancy, colored)
    {}

    std::ostream& emit_type(const Type*);
    std::ostream& emit_name(Def);
    std::ostream& emit_def(Def);
    std::ostream& emit_primop(const PrimOp*);
    std::ostream& emit_assignment(const PrimOp*);
    std::ostream& emit_head(const Lambda*);
    std::ostream& emit_jump(const Lambda*);
};

//------------------------------------------------------------------------------

std::ostream& CodeGen::emit_type(const Type* type) {
    if (type == nullptr) {
        return stream() << "<NULL>";
    } else if (type->isa<Frame>()) {
        return stream() << "frame";
    } else if (type->isa<Mem>()) {
        return stream() << "mem";
    } else if (auto pi = type->isa<Pi>()) {
        return dump_list([&](const Type* type) { emit_type(type); }, pi->elems(), "fn(", ")");
    } else if (auto sigma = type->isa<Sigma>()) {
        return dump_list([&](const Type* type) { emit_type(type); }, sigma->elems(), "(", ")");
    } else if (auto generic = type->isa<Generic>()) {
        return stream() << '<' << generic->index() << '>';
    } else if (auto genref = type->isa<GenericRef>()) {
        stream() << '<';
        emit_name(genref->lambda()) << ", ";
        return emit_type(genref->generic()) << '>';
    } else if (auto array = type->isa<IndefArray>()) {
        stream() << '[';
        emit_type(array->elem_type());
        return stream() << ']';
    } else if (auto array = type->isa<DefArray>()) {
        stream() << '[' << array->dim() << " x ";
        emit_type(array->elem_type());
        return stream() << ']';
    } else if (auto ptr = type->isa<Ptr>()) {
        if (ptr->is_vector())
            stream() << '<' << ptr->length() << " x ";
        emit_type(ptr->referenced_type());
        stream() << '*';
        if (ptr->is_vector())
            stream() << '>';
        return stream();
    } else if (auto primtype = type->isa<PrimType>()) {
        if (primtype->is_vector())
            stream() << "<" << primtype->length() << " x ";
            switch (primtype->primtype_kind()) {
#define THORIN_ALL_TYPE(T) case Node_PrimType_##T: stream() << #T; break;
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
        switch (primlit->primtype_kind()) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: stream() << primlit->T##_value(); break;
#include "thorin/tables/primtypetable.h"
            default: THORIN_UNREACHABLE; break;
        }
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
    dump_list([&](const Param* param) { emit_type(param->type()) << " "; emit_name(param); }, lambda->params(), "(", ")");

    if (lambda->attribute().is(Lambda::Extern))
        stream() << " extern ";
    if (lambda->attribute().is(Lambda::Intrinsic))
        stream() << " intrinsic ";

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

void emit_thorin(World& world, bool fancy, bool nocolor) {
    CodeGen cg(fancy, nocolor);
    cg.stream() << "module '" << world.name() << "'\n\n";

    for (auto primop : world.primops()) {
        if (auto global = primop->isa<Global>())
            cg.emit_assignment(global);
    }

    Scope scope(world);
    const DomTree domtree(scope);
    Schedule schedule = schedule_smart(scope);
    for (auto lambda : scope) {
        //if (scope.exit() == lambda)
            //continue; // HACK
        int depth = fancy ? domtree.depth(lambda) : 0;
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

void emit_type(const Type* type)           { CodeGen(false).emit_type(type);         }
void emit_def(Def def)                     { CodeGen(false).emit_def(def);           }
void emit_head(const Lambda* lambda)       { CodeGen(false).emit_head(lambda);       }
void emit_jump(const Lambda* lambda)       { CodeGen(false).emit_jump(lambda);       }
void emit_assignment(const PrimOp* primop) { CodeGen(false).emit_assignment(primop); }

//------------------------------------------------------------------------------

} // namespace thorin
