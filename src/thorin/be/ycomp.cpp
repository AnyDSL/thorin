#include <sstream>
#include "thorin/lambda.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/bb_schedule.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/printer.h"

using namespace std::placeholders;

namespace thorin {

typedef std::function<void ()> Emitter;

class YCompGen : public Printer {
private:
    DefSet emitted_defs;

    static void EMIT_NOOP() { }

    std::ostream& emit_type_vars(Type);
    std::ostream& emit_type_args(Type);
    std::ostream& emit_type_elems(Type);
    std::ostream& emit_type(Type);
    std::ostream& emit_name(Def);

    std::ostream& emit_operands(Def def);
    std::ostream& emit_lambda_graph_begin(const Lambda*);
    std::ostream& emit_lambda_graph_params(const Lambda*);
    std::ostream& emit_lambda_graph_continuation(const Lambda*);
    std::ostream& emit_lambda_graph_end(const Lambda*);

    std::ostream& emit_def(Def);
    std::ostream& emit_primop(const PrimOp*);
    std::ostream& emit_param(const Param*);
    std::ostream& emit_lambda(const Lambda*);
    std::ostream& emit_lambda_graph(const Lambda*);
    std::ostream& emit_lambda_graph(const Lambda*, const std::vector<const PrimOp*>& schedule);

    template<typename T, typename U>
    std::ostream& write_edge(T source, U target, bool control_flow,
        Emitter label = EMIT_NOOP);

    template<typename T>
    std::ostream& write_node(T gid, Emitter label,
        Emitter info1 = EMIT_NOOP,
        Emitter info2 = EMIT_NOOP,
        Emitter info3 = EMIT_NOOP);
public:
    YCompGen(bool fancy, bool colored = false, int indent = 0)
        : Printer(std::cout, fancy, colored) {
        this->indent = indent;
    }

    void emit_scope(const Scope& scope);
    void emit_world(const World& world);
};

//------------------------------------------------------------------------------

std::ostream& YCompGen::emit_type_vars(Type type) {
    if (type->num_type_vars() != 0)
        return dump_list([&](TypeVar type_var) { emit_type(type_var); }, type->type_vars(), "[", "]");
    return stream();
}

std::ostream& YCompGen::emit_type_args(Type type) {
    return dump_list([&](Type type) { emit_type(type); }, type->args(), "(", ")");
}

std::ostream& YCompGen::emit_type_elems(Type type) {
    if (auto struct_app = type.isa<StructAppType>())
        return dump_list([&](Type type) { emit_type(type); }, struct_app->elems(), "{", "}");
    return emit_type_args(type);
}

std::ostream& YCompGen::emit_type(Type type) {
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

std::ostream& YCompGen::emit_operands(Def def) {
    int i = 0;
    Emitter emit_label = EMIT_NOOP;
    if (def->size() > 1) {
        emit_label = [&] { stream() << i++; };
    }
    dump_list([&](Def operand) {
            write_edge(def->gid(), operand->gid(), false, emit_label);
        }, def->ops(), "", "", "");
    return stream();
}

std::ostream& YCompGen::emit_def(Def def) {
    if (emitted_defs.contains(def)) {
        return stream();
    }
    if (auto primop = def->isa<PrimOp>()) {
        emit_primop(primop);
    } else if (auto lambda = def->isa<Lambda>()) {
        emit_lambda_graph(lambda);
    } else if (auto param = def->isa<Param>()) {
        emit_param(param);
    } else {
        // XXX what is it?
        write_node(def->gid(), std::bind(&YCompGen::emit_name, this, def),
            std::bind(&YCompGen::emit_type, this, def->type()));
        emit_operands(def);
    }
    emitted_defs.insert(def);
    return stream();
}

std::ostream& YCompGen::emit_name(Def def) {
    if (is_fancy()) // elide white = 0 and black = 7
        color(def->gid() % 6 + 30 + 1);

    std::ostream& s = stream() << def->unique_name();

    if (is_fancy())
        reset_color();

    return s;
}

std::ostream& YCompGen::emit_primop(const PrimOp* primop) {
    if (emitted_defs.contains(primop)) {
        return stream();
    }
    emitted_defs.insert(primop);
    auto emit_label = [&] {
        if (primop->is_proxy()) {
            stream() << "<proxy>";
        } else if (auto primlit = primop->isa<PrimLit>()) {
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
            stream() << primop->op_name() << " ";
            emit_name(primop);
        } else if (primop->is_const()) {
            if (primop->empty()) {
                stream() << primop->op_name() << ' ';
                emit_type(primop->type());
            } else {
                stream() << '(';
                if (primop->isa<PrimLit>()) {
                    emit_type(primop->type()) << ' ';
                }
                stream() << primop->op_name() << ')';
            }
        } else {
            stream() << primop->op_name() << " ";
            emit_name(primop);
        }
        //auto ops = primop->ops();
        if (auto vectorop = primop->isa<VectorOp>()) {
            if (!vectorop->cond()->is_allset()) {
                stream() << "@ ";
                emit_name(vectorop->cond()) << " ";
            }
            //ops = ops.slice_from_begin(1);
        }
    };
    write_node(primop->gid(), emit_label,
        [&] { emit_type(primop->type()); });
    emit_operands(primop);
    return stream();
}

template<typename T, typename U>
std::ostream& YCompGen::write_edge(T source, U target, bool control_flow,
        Emitter emit_label) {
    newline() << "edge: { sourcename: \"" << source << "\" targetname: \"" << target << "\"" << " class: ";
    if (control_flow) {
        stream() << 13 << " color: blue";
    } else {
        stream() << 16;
    }
    stream() << " label: \"";
    emit_label();
    return stream() << "\"}";
}

template<typename T>
std::ostream& YCompGen::write_node(T id, Emitter emit_label,
        Emitter emit_info1, Emitter emit_info2, Emitter emit_info3) {
    newline() << "node: { title: \"" << id << "\" label: \"";
    emit_label();
    stream() << "\" info1: \"";
    emit_info1();
    stream() << "\" info2: \"";
    emit_info2();
    stream() << "\" info3: \"";
    emit_info3();
    return stream() << "\"}";
}

std::ostream& YCompGen::emit_param(const Param* param) {
    if (emitted_defs.contains(param)) {
        return stream();
    }
    emitted_defs.insert(param);
    write_node(param->gid(), [&] { stream() << "Param "; emit_name(param); },
        [&] { emit_type(param->type()); });
    emit_operands(param);
    return stream();
}

std::ostream& YCompGen::emit_lambda(const Lambda* lambda) {
    if (emitted_defs.contains(lambda)) {
        return stream();
    }
    emitted_defs.insert(lambda);
    write_node(lambda->gid(),
        [&] { stream() << "λ "; emit_name(lambda); },
        [&] { emit_type_vars(lambda->type()); });
    if (!lambda->empty()) {
        write_edge(lambda->gid(), lambda->to()->gid(), true);
        int i = 0;
        for (auto def : lambda->args()) {
            write_edge(lambda->gid(), def->gid(), false, [&] { stream() << i++;});
        }
    }
    return stream();
}

std::ostream& YCompGen::emit_lambda_graph_begin(const Lambda* lambda) {
    newline() << "graph: {";
    up() << "title: \"" << lambda->gid() << "\"";
    newline() << "label: \"λ ";
    emit_name(lambda);
    stream() << "\"";
    newline() << "info1: \"";
    emit_type_vars(lambda->type());
    dump_list([&](const Param* param) { emit_type(param->type()); }, lambda->params(), "(", ")");

    if (lambda->is_external())
        stream() << " extern";
    if (lambda->cc() == CC::Device)
        stream() << " device";
    return stream() << "\"";
}

std::ostream& YCompGen::emit_lambda_graph_params(const Lambda* lambda) {
    for (auto param : lambda->params()) {
        emit_param(param);
    }
    return stream();
}

std::ostream& YCompGen::emit_lambda_graph_continuation(const Lambda* lambda) {
    if (!lambda->empty()) {
        write_node("cont"+std::to_string(lambda->gid()),
            [&] { stream() << "continue"; });
        write_edge("cont"+std::to_string(lambda->gid()), lambda->to()->gid(), true);
        int i = 0;
        for (auto def : lambda->args()) {
            write_edge("cont"+std::to_string(lambda->gid()), def->gid(), false,
                [&] { stream() << i++; });
        }
        // write_edge(lambda->gid(), lambda->to()->gid(), true, [&] {;});
        // int i = 0;
        // for (auto def : lambda->args()) {
        //     write_edge(lambda->gid(), def->gid(), false, [&] { stream() << i; i++; });
        // }
    }
    return stream();
}

std::ostream& YCompGen::emit_lambda_graph_end(const Lambda* lambda) {
    return down() << "}";
}
std::ostream& YCompGen::emit_lambda_graph(const Lambda* lambda, const std::vector<const PrimOp*>& schedule) {
    emitted_defs.insert(lambda);
    emit_lambda_graph_begin(lambda);
    emit_lambda_graph_params(lambda);
    emit_lambda_graph_continuation(lambda);
    for (auto primop : schedule) {
        emit_primop(primop);
    }
    return emit_lambda_graph_end(lambda);
}
std::ostream& YCompGen::emit_lambda_graph(const Lambda* lambda) {
    emitted_defs.insert(lambda);
    emit_lambda_graph_begin(lambda);
    emit_lambda_graph_params(lambda);
    emit_lambda_graph_continuation(lambda);
    return emit_lambda_graph_end(lambda);
}

void YCompGen::emit_scope(const Scope& scope) {
    auto schedule = schedule_smart(scope);
    // auto bbs = bb_schedule(scope);
    newline() << "graph: {";
    up() << "title: \"scope" << scope.id() << "\"";
    newline() << "label: \"scope " << scope.id() << "\"";
    for (auto lambda : scope) {
        emit_lambda_graph(lambda, schedule[lambda]);
    }
    //for (auto def : scope.in_scope()) {
    //    cg.emit_def(def);
    //}
    down() << "}";
    newline();
}

void YCompGen::emit_world(const World& world) {
    stream() << "graph: {";
    up() << "layoutalgorithm: mindepth //$ \"Compilergraph\"";
    newline() << "orientation: bottom_to_top";
    newline() << "graph: {";
    up() << "label: \"module " << world.name() << "\"";

    Scope::for_each<false>(world, [&] (const Scope& scope) { emit_scope(scope); });

    for (auto primop : world.primops()) {
        emit_def(primop);
        //if (auto global = primop->isa<Global>())
//            cg.emit_primop(global);
    }

    down() << "}";
    down() << "}";
}

//------------------------------------------------------------------------------

void emit_ycomp(const Scope& scope, int indent = 0, bool fancy = true, bool colored = false) {
    YCompGen cg(fancy, colored, indent);
    cg.emit_scope(scope);
}

void emit_ycomp(const World& world, bool fancy, bool colored) {
    YCompGen cg(fancy, colored);
    cg.emit_world(world);
}

//------------------------------------------------------------------------------


}
