#include "thorin/continuation.h"
#include "thorin/primop.h"
#include "thorin/type.h"
#include "thorin/world.h"
#include "thorin/analyses/schedule.h"
#include "thorin/analyses/scope.h"
#include "thorin/util/printer.h"

namespace thorin {

typedef std::function<void()> Emitter;

class YCompGen : public Printer {
public:
    YCompGen(bool scheduled, std::ostream& ostream)
        : Printer(ostream)
        , scheduled_(scheduled)
    {}

    void emit_scope_cfg(const Scope& scope);
    void emit_world_cfg(const World& world);
    void emit_scope(const Scope& scope);
    void emit_world(const World& world);

private:
    DefSet emitted_defs;
    bool scheduled_;

    static void EMIT_NOOP() { }

    std::ostream& emit_operands(const Def* def);
    std::ostream& emit_continuation_graph_begin(const Continuation*);
    std::ostream& emit_continuation_graph_params(const Continuation*);
    std::ostream& emit_continuation_graph_continuation(const Continuation*);
    std::ostream& emit_continuation_graph_end();

    std::ostream& emit_def(const Def*);
    std::ostream& emit_primop(const PrimOp*);
    std::ostream& emit_param(const Param*);
    std::ostream& emit_continuation(const Continuation*);
    std::ostream& emit_continuation_graph(const Continuation*);
    std::ostream& emit_block(const Schedule::Block&);

    template<bool forward>
    std::ostream& emit_cfnode(const CFG<forward>&, const CFNode*);

    std::ostream& emit_type(const Type* type) { return stream() << type; }
    std::ostream& emit_name(const Def* def) { return stream() << def; }

    template<typename T, typename U>
    std::ostream& write_edge(T source, U target, bool control_flow,
        Emitter label = EMIT_NOOP);

    template<typename T>
    std::ostream& write_node(T gid, Emitter label,
        Emitter info1 = EMIT_NOOP,
        Emitter info2 = EMIT_NOOP,
        Emitter info3 = EMIT_NOOP);
};

//------------------------------------------------------------------------------


std::ostream& YCompGen::emit_operands(const Def* def) {
    int i = 0;
    Emitter emit_label = EMIT_NOOP;
    if (def->size() > 1) {
        emit_label = [&] { stream() << i++; };
    }
    dump_list([&](const Def* op) {
        write_edge(def->gid(), op->gid(), false, emit_label);
    }, def->ops(), "", "", "");
    return stream();
}

std::ostream& YCompGen::emit_def(const Def* def) {
    if (emitted_defs.contains(def)) {
        return stream();
    }
    if (auto primop = def->isa<PrimOp>()) {
        emit_primop(primop);
    } else if (auto continuation = def->isa<Continuation>()) {
        emit_continuation_graph(continuation);
    } else if (auto param = def->isa<Param>()) {
        emit_param(param);
    } else {
        // default, but should not happen...
        write_node(def->gid(),
            [&] { emit_name(def); },
            [&] { emit_type(def->type()); });
        emit_operands(def);
    }
    emitted_defs.insert(def);
    return stream();
}

std::ostream& YCompGen::emit_primop(const PrimOp* primop) {
    if (emitted_defs.contains(primop)) {
        return stream();
    }
    emitted_defs.insert(primop);
    auto emit_label = [&] {
        if (auto primlit = primop->isa<PrimLit>()) {
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
                    emit_type(primop->type());
                    stream() << ' ';
                }
                stream() << primop->op_name() << ')';
            }
        } else {
            stream() << primop->op_name() << " ";
            emit_name(primop);
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

std::ostream& YCompGen::emit_continuation(const Continuation* continuation) {
    if (emitted_defs.contains(continuation)) {
        return stream();
    }
    emitted_defs.insert(continuation);
    write_node(continuation->gid(),
        [&] { stream() << "λ "; emit_name(continuation); },
        [&] { emit_type(continuation->type()); });
    if (!continuation->empty()) {
        write_edge(continuation->gid(), continuation->callee()->gid(), true);
        int i = 0;
        for (auto def : continuation->args()) {
            write_edge(continuation->gid(), def->gid(), false, [&] { stream() << i++;});
        }
    }
    return stream();
}

std::ostream& YCompGen::emit_continuation_graph_begin(const Continuation* continuation) {
    newline() << "graph: {";
    up() << "title: \"" << continuation->gid() << "\"";
    newline() << "label: \"λ ";
    emit_name(continuation);
    stream() << "\"";
    newline() << "info1: \"";
    emit_type(continuation->type());

    if (continuation->is_external())
        stream() << " extern";
    if (continuation->cc() == CC::Device)
        stream() << " device";
    return stream() << "\"";
}

std::ostream& YCompGen::emit_continuation_graph_params(const Continuation* continuation) {
    for (auto param : continuation->params())
        emit_param(param);
    return stream();
}

std::ostream& YCompGen::emit_continuation_graph_continuation(const Continuation* continuation) {
    if (!continuation->empty()) {
        write_node("cont"+std::to_string(continuation->gid()),
            [&] { stream() << "continue"; });
        write_edge("cont"+std::to_string(continuation->gid()), continuation->callee()->gid(), true);
        int i = 0;
        for (auto def : continuation->args()) {
            write_edge("cont"+std::to_string(continuation->gid()), def->gid(), false,
                [&] { stream() << i++; });
        }
        // write_edge(continuation->gid(), continuation->callee()->gid(), true, [&] {;});
        // int i = 0;
        // for (auto def : continuation->args()) {
        //     write_edge(continuation->gid(), def->gid(), false, [&] { stream() << i; i++; });
        // }
    }
    return stream();
}

std::ostream& YCompGen::emit_continuation_graph_end() { return down() << "}"; }

std::ostream& YCompGen::emit_block(const Schedule::Block& block) {
    auto continuation = block.continuation();
    emitted_defs.insert(continuation);
    emit_continuation_graph_begin(continuation);
    emit_continuation_graph_params(continuation);
    emit_continuation_graph_continuation(continuation);
    for (auto primop : block)
        emit_primop(primop);

    return emit_continuation_graph_end();
}
std::ostream& YCompGen::emit_continuation_graph(const Continuation* continuation) {
    emitted_defs.insert(continuation);
    emit_continuation_graph_begin(continuation);
    emit_continuation_graph_params(continuation);
    emit_continuation_graph_continuation(continuation);
    return emit_continuation_graph_end();
}

template<bool forward>
std::ostream& YCompGen::emit_cfnode(const CFG<forward>& cfg, const CFNode* node) {
    write_node(node->to_string(),
        [&] { stream() << node; },
        [&] { stream() << node; }); // TODO fix this: twice the same thing

    for (auto succ : cfg.succs(node))
        write_edge(node->to_string(), succ->to_string(), true);

    return stream();
}

void YCompGen::emit_scope_cfg(const Scope& scope) {
    newline() << "graph: {";
    up() << "title: \"scope" << scope.id() << "\"";
    newline() << "label: \"scope " << scope.id() << "\"";

    auto& cfg = scope.f_cfg();
    for (auto node : cfg.reverse_post_order()) {
        emit_cfnode(cfg, node);
    }

    down() << "}";
    newline();
}

void YCompGen::emit_world_cfg(const World& world) {
    stream() << "graph: {";
    up() << "layoutalgorithm: mindepth //$ \"Compilergraph\"";
    newline() << "orientation: top_to_bottom";
    newline() << "graph: {";
    up() << "label: \"module " << world.name() << "\"";

    Scope::for_each<false>(world, [&] (const Scope& scope) { emit_scope_cfg(scope); });

    down() << "}";
    down() << "}";
}


void YCompGen::emit_scope(const Scope& scope) {
    newline() << "graph: {";
    up() << "title: \"scope" << scope.id() << "\"";
    newline() << "label: \"scope " << scope.id() << "\"";
    if (scheduled_) {
        for (auto& block : schedule(scope))
            emit_block(block);
    } else {
        for (auto continuation : scope)
            emit_continuation_graph(continuation);
        for (auto def : scope.defs())
           emit_def(def);
    }
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
        if (!scheduled_) {
            if (auto global = primop->isa<Global>()) {
                emit_primop(global);
            }
        }
    }

    down() << "}";
    down() << "}";
}

//------------------------------------------------------------------------------

// TODO option for outputting primops?
void emit_ycomp_cfg(const Scope& scope, std::ostream& ostream) {
    YCompGen cg(false, ostream);
    cg.emit_scope_cfg(scope);
}

void emit_ycomp_cfg(const World& world, std::ostream& ostream) {
    YCompGen cg(false, ostream);
    cg.emit_world_cfg(world);
}

void emit_ycomp(const Scope& scope, bool scheduled, std::ostream& ostream) {
    YCompGen cg(scheduled, ostream);
    cg.emit_scope(scope);
}

void emit_ycomp(const World& world, bool scheduled, std::ostream& ostream) {
    YCompGen cg(scheduled, ostream);
    cg.emit_world(world);
}

//------------------------------------------------------------------------------


}
