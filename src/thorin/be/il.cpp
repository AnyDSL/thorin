#include "thorin/lambda.h"
#include "thorin/literal.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/top_level_scopes.h"
#include "thorin/util/hash.h"
#include "thorin/util/printer.h"

namespace thorin {

typedef HashSet<const DefNode*> Vars;

void free_vars(const DomTree& domtree, Schedule& schedule, Lambda* lambda, Vars& vars) {
    for (auto lamb : domtree.lookup(lambda)->children()) {
        free_vars(domtree, schedule, lamb->lambda(), vars);
    }
    for (auto op : lambda->args()) {
        if (op->isa<PrimOp>() && !op->is_const())
            vars.insert(op);
    }
    std::vector<const PrimOp*>& ops = schedule[lambda];
    for (auto i = ops.rbegin(); i != ops.rend(); ++i) {
        vars.erase(*i);
        for (auto op : (*i)->ops()) {
            if (op->isa<PrimOp>() && !op->is_const())
                vars.insert(op);
        }
    }
}

void defined_vars(const DomTree& domtree, Schedule& schedule, Lambda* lambda, Vars& vars) {
    std::vector<const PrimOp*>& ops = schedule[lambda];
    for (auto i = ops.rbegin(); i != ops.rend(); ++i) {
        vars.erase(*i);
    }
    auto idom = domtree.idom(lambda);
    if (idom && idom != lambda) {
        defined_vars(domtree, schedule, idom, vars);
    }
}

class IlPrinter : public Printer {
public:
    IlPrinter(bool fancy)
        : Printer(std::cout, fancy)
    {}

    std::ostream& emit_type(const Type*);
    std::ostream& emit_name(Def);
    std::ostream& emit_def(Def);
    std::ostream& emit_primop(const PrimOp*);
    std::ostream& emit_assignment(const PrimOp*);
    std::ostream& emit_head(const Lambda*, bool nodefs);
    std::ostream& emit_jump(const Lambda*, bool nodefs);

    void print_lambda(const DomTree& domtree, Schedule& schedule, Lambda* lambda, Vars& def_vars);
    DefSet pass_;
};

//------------------------------------------------------------------------------

std::ostream& IlPrinter::emit_type(const Type* type) {
    if (type == nullptr)
        return stream() << "null";
    else if (type->isa<Frame>())
        return stream() << "frame";
    else if (type->isa<Mem>())
        return stream() << "mem";
    else if (auto pi = type->isa<Pi>()) {
        if (pi->elems().empty())
          return stream() << "unit -> unit";
        else
          return dump_list([&] (const Type* type) { emit_type(type); }, pi->elems(), "", " -> unit", " * ");
    }
    else if (auto sigma = type->isa<Sigma>())
        return dump_list([&] (const Type* type) { emit_type(type); }, sigma->elems(), "", "", " * ");
    else if (type->isa<Generic>())
        return stream() << "TODO";
    else if (type->isa<GenericRef>())
        return stream() << "TODO";
    else if (auto ptr = type->isa<Ptr>()) {
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

std::ostream& IlPrinter::emit_def(Def def) {
    if (auto primop = def->isa<PrimOp>())
        return emit_primop(primop);
    return emit_name(def);
}

std::ostream& IlPrinter::emit_name(Def def) {
    if (is_fancy()) // elide white = 0 and black = 7
        color(def->gid() % 6 + 30 + 1);

    if (def->isa<PrimOp>())
      stream() << "v";
    stream() << def->unique_name();

    if (is_fancy())
        reset_color();

    return stream();
}

std::ostream& IlPrinter::emit_primop(const PrimOp* primop) {
    if (auto primlit = primop->isa<PrimLit>()) {
        //emit_type(primop->type()) << ' ';
        switch (primlit->primtype_kind()) {
#define THORIN_ALL_TYPE(T) case PrimType_##T: stream() << primlit->T##_value(); break;
#include "thorin/tables/primtypetable.h"
            default: THORIN_UNREACHABLE; break;
        }
    } else if (primop->is_const()) {
        if (primop->empty()) {
            stream() << primop->op_name() << " ";
            emit_type(primop->type());
        } else {
            emit_type(primop->type());
            dump_list([&] (Def def) { emit_def(def); }, primop->ops(), "(", ")");
        }
    } else
        emit_name(primop);

    return stream();
}

std::ostream& IlPrinter::emit_assignment(const PrimOp* primop) {
    stream() << "val ";
    emit_name(primop) << " : ";
    emit_type(primop->type()) << " = ";

    ArrayRef<Def> ops = primop->ops();
    if (primop->isa<Select>()) {
    } else if (auto vectorop = primop->isa<VectorOp>()) {
        if (!vectorop->cond()->is_allset()) {
            stream() << "@ ";
            emit_name(vectorop->cond()) << " ";
        }
        ops = ops.slice_from_begin(1);
    }

    stream() << primop->op_name() << " ";
    dump_list([&] (Def def) { emit_def(def); }, ops, "(", ")");
    return newline();
}

std::ostream& IlPrinter::emit_head(const Lambda* lambda, bool nodefs) {
    stream() << "fun ";
    emit_name(lambda) << " ";
    dump_list([&] (const Param* param) { emit_name(param) << " : "; emit_type(param->type()); }, lambda->params(), "(", ")");

//    if (lambda->attr().is_extern())
//       stream() << " extern ";

    stream() << " = ";
    if (!nodefs)
        stream() << "let";

    return up();
}

std::ostream& IlPrinter::emit_jump(const Lambda* lambda, bool nodefs) {
    if (!lambda->empty()) {
        if (!nodefs)
            stream() << "in ";
        emit_def(lambda->to());
        dump_list([&] (Def def) { emit_def(def); }, lambda->args(), "(", ")");
        if (!nodefs)
            stream() << " end";
    }
    return down();
}


void IlPrinter::print_lambda(const DomTree& domtree, Schedule& schedule, Lambda* lambda, Vars& def_vars) {
    if (visit(pass_, lambda))
        return;
    emit_head(lambda, schedule[lambda].empty());
    bool first = true;
    Vars this_def_vars;
    for (auto op : schedule[lambda]) {
        for (auto lamb : domtree.lookup(lambda)->children()) {
            Vars vars;
            free_vars(domtree, schedule, lamb->lambda(), vars);
            for (auto dop : def_vars) {
                vars.erase(dop);
            }
            if (vars.empty())
            print_lambda(domtree, schedule, lamb->lambda(), def_vars);
        }
        if (!first) {
            //newline();
        } else {
            first = false;
        }
        emit_assignment(op);
        def_vars.insert(op);
        this_def_vars.insert(op);
    }

    emit_jump(lambda, schedule[lambda].empty());
    for (auto dop : this_def_vars) {
        def_vars.erase(dop);
    }
    //newline();
}

//------------------------------------------------------------------------------

void emit_il(World& world, bool fancy) {
    IlPrinter cg(fancy);

    for (auto scope : top_level_scopes(world)) {
        auto lambda = scope->entry();
        const DomTree domtree(*scope);
        Schedule schedule = schedule_smart(*scope);
        cg.pass_.clear();
        Vars def_vars;
        cg.print_lambda(domtree, schedule, lambda, def_vars);
        cg.newline();
    }
}

//------------------------------------------------------------------------------

} // namespace thorin
