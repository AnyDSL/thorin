#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/schedule.h"
#include "anydsl2/util/printer.h"

namespace anydsl2 {

typedef std::unordered_set<const Def*> Vars;

void free_vars(Scope& scope, Schedule& schedule, Lambda* lambda, Vars& vars) {
    for (auto lamb : scope.domtree().node(lambda)->children()) {
      free_vars(scope, schedule, lamb->lambda(), vars);
    }
    std::vector<const PrimOp*>& ops = schedule[lambda->sid()];
    for (auto i = ops.rbegin(); i != ops.rend(); ++i) {
      vars.erase(*i);
      for (auto op : (*i)->ops()) {
        if (op->isa<PrimOp>() && !op->is_const())
          vars.insert(op);
      }
    }
}

void defined_vars(Scope& scope, Schedule& schedule, Lambda* lambda, Vars& vars) {
    std::vector<const PrimOp*>& ops = schedule[lambda->sid()];
    for (auto i = ops.rbegin(); i != ops.rend(); ++i) {
        vars.erase(*i);
    }
    auto idom = scope.domtree().idom(lambda);
    if (idom && idom != lambda) {
        defined_vars(scope, schedule, idom, vars);
    }
}

class IlPrinter : public Printer {
public:
    IlPrinter(bool fancy)
        : Printer(std::cout, fancy)
    {}

    std::ostream& emit_type(const Type*);
    std::ostream& emit_name(const Def*);
    std::ostream& emit_def(const Def*);
    std::ostream& emit_primop(const PrimOp*);
    std::ostream& emit_assignment(const PrimOp*);
    std::ostream& emit_head(const Lambda*, bool nodefs);
    std::ostream& emit_jump(const Lambda*, bool nodefs);

    void print_lambda(Scope& scope, Schedule& schedule, Lambda* lambda, Vars& def_vars);
    int pass_;
};

//------------------------------------------------------------------------------

std::ostream& IlPrinter::emit_type(const Type* type) {
    if (type == nullptr)
        return stream() << "null";
    else if (auto frame = type->isa<Frame>())
        return stream() << "frame";
    else if (auto mem = type->isa<Mem>())
        return stream() << "mem";
    else if (auto pi = type->isa<Pi>())
        return dump_list([&] (const Type* type) { emit_type(type); }, pi->elems(), "", " ->#", " * ");
    else if (auto sigma = type->isa<Sigma>())
        return dump_list([&] (const Type* type) { emit_type(type); }, sigma->elems(), "", "", " * ");
    else if (auto generic = type->isa<Generic>())
        return stream() << "TODO";
    else if (auto genref = type->isa<GenericRef>())
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
#define ANYDSL2_UF_TYPE(T) case Node_PrimType_##T: stream() << #T; break;
#include "anydsl2/tables/primtypetable.h"
            default: ANYDSL2_UNREACHABLE;
        }
        if (primtype->is_vector())
            stream() << ">";
        return stream();
    }
    ANYDSL2_UNREACHABLE;
}

std::ostream& IlPrinter::emit_def(const Def* def) {
    if (auto primop = def->isa<PrimOp>())
        return emit_primop(primop);
    return emit_name(def);
}

std::ostream& IlPrinter::emit_name(const Def* def) {
    if (is_fancy()) // elide white = 0 and black = 7
        color(def->gid() % 6 + 30 + 1);

    stream() << def->unique_name();

    if (is_fancy())
        reset_color();

    return stream();
}

std::ostream& IlPrinter::emit_primop(const PrimOp* primop) {
    if (auto primlit = primop->isa<PrimLit>()) {
        emit_type(primop->type()) << ' ';
        switch (primlit->primtype_kind()) {
#define ANYDSL2_UF_TYPE(T) case PrimType_##T: stream() << primlit->T##_value(); break;
#include "anydsl2/tables/primtypetable.h"
            default: ANYDSL2_UNREACHABLE; break;
        }
    } else if (primop->is_const()) {
        if (primop->empty()) {
            stream() << primop->op_name() << " ";
            emit_type(primop->type());
        } else {
            emit_type(primop->type());
            dump_list([&] (const Def* def) { emit_def(def); }, primop->ops(), "(", ")");
        }
    } else
        emit_name(primop);

    return stream();
}

std::ostream& IlPrinter::emit_assignment(const PrimOp* primop) {
    stream() << "val ";
    emit_name(primop) << " : ";
    emit_type(primop->type()) << " = ";

    ArrayRef<const Def*> ops = primop->ops();
    if (auto vectorop = primop->isa<VectorOp>()) {
        if (!vectorop->cond()->is_allset()) {
            stream() << "@ ";
            emit_name(vectorop->cond()) << " ";
        }
        ops = ops.slice_back(1);
    }

    stream() << primop->op_name() << " ";
    dump_list([&] (const Def* def) { emit_def(def); }, ops);
    return newline();
}

std::ostream& IlPrinter::emit_head(const Lambda* lambda, bool nodefs) {
    stream() << "fun ";
    emit_name(lambda) << " ";
    dump_list([&] (const Param* param) { emit_name(param) << " : "; emit_type(param->type()); }, lambda->params(), "(", ")");

    if (lambda->attr().is_extern())
        stream() << " extern ";

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
        dump_list([&] (const Def* def) { emit_def(def); }, lambda->args(), "(", ")");
    }
    return down();
}


void IlPrinter::print_lambda(Scope& scope, Schedule& schedule, Lambda* lambda, Vars& def_vars) {
            if (lambda->visit(pass_))
              return;
            emit_head(lambda, schedule[lambda->sid()].empty());
            bool first = true;
            for (auto op : schedule[lambda->sid()]) {
                for (auto lamb : scope.domtree().node(lambda)->children()) {
                  Vars vars;
                  free_vars(scope, schedule, lamb->lambda(), vars);
                  for (auto dop : def_vars) {
                      vars.erase(dop);
                  }
                  if (vars.empty())
                    print_lambda(scope, schedule, lamb->lambda(), def_vars);
                }
                if (!first) {
                  //newline();
                } else {
                  first = false;
                }
                emit_assignment(op);
                def_vars.insert(op);
            }

            emit_jump(lambda, schedule[lambda->sid()].empty());
            //newline();
}

//------------------------------------------------------------------------------

void emit_il(World& world, bool fancy) {
    IlPrinter cg(fancy);

    for (auto lambda : top_level_lambdas(world)) {
        Scope scope(lambda);
        Schedule schedule = schedule_smart(scope);
        cg.pass_ = world.new_pass();
        Vars def_vars;
        cg.print_lambda(scope, schedule, lambda, def_vars);
        cg.newline();
    }
}

//------------------------------------------------------------------------------

} // namespace anydsl2
